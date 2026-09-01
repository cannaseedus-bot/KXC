// ggml-xcfe.cpp — KHΛNARY XCFE backend for ggml (structure mirrors ggml-blas).
//
// Registered backend "XCFE"; reuses CPU host buffers. Dispatch chain per op:
//
//   MUL_MAT  : GL(xcfe_gl_ops) → DML(dml_gemm) → D3D11(d3d11_infer) →
//              XVM Fiber(25 GFLOPS) → OpenCL(21 GFLOPS) → WebGL2 → CPU ref
//   NORM/GELU: GL → D3D11 → XVM Fiber → OpenCL → WebGL2 → CPU ref
//   ADD      : GL → D3D11 add → CPU inline
//
// Each fallback is loaded lazily. Env-var gates:
//   KHANARY_D3D11=0       disable d3d11_infer.dll
//   KHANARY_XVM_FIBER=<path>  load xvm_fiber.dll from path (exports xcfe_xvm_gemm_bt_f32 / xcfe_xvm_op_f32)
//   KHANARY_OPENCL_DLL=<path> load xcfe_cl.dll (exports xcfe_cl_gemm_bt_f32 / xcfe_cl_op_f32)
//   KHANARY_WEBGL2_DLL=<path> load xcfe_webgl2.dll (exports xcfe_webgl2_gemm_bt_f32 / xcfe_webgl2_op_f32)

#include "ggml-impl.h"
#include "ggml-xcfe.h"
#include "ggml-backend-impl.h"

#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

struct ggml_backend_xcfe_context {
    int n_threads = GGML_DEFAULT_N_THREADS;
};

// ── GL compute (xcfe_gl_ops.dll) ──────────────────────────────────────────────
typedef int (*xcfe_gl_run_fn)(const char*, const float* const*, int, float*,
                              const int64_t*, int, const float*, int);
static xcfe_gl_run_fn xcfe_gl_fn() {
#ifdef _WIN32
    static xcfe_gl_run_fn fn = [](){
        HMODULE h = LoadLibraryA("xcfe_gl_ops.dll");
        return h ? (xcfe_gl_run_fn) GetProcAddress(h, "xcfe_gl_run") : nullptr;
    }();
    return fn;
#else
    return nullptr;
#endif
}

// ── DirectML fast path (dml_gemm.dll) ────────────────────────────────────────
// C[M,N] = A[M,K] @ B^T  (B is [N,K]) — exactly ggml MUL_MAT convention.
typedef int (*dml_gemm_bt_fn)(const float*, const float*, float*, unsigned, unsigned, unsigned);
static dml_gemm_bt_fn xcfe_dml_bt() {
#ifdef _WIN32
    static dml_gemm_bt_fn fn = [](){
        HMODULE h = LoadLibraryA("dml_gemm.dll");
        return h ? (dml_gemm_bt_fn) GetProcAddress(h, "dml_gemm_bt_f32") : nullptr;
    }();
    return fn;
#else
    return nullptr;
#endif
}

// ── D3D11 native fallback (d3d11_infer.dll) ───────────────────────────────────
// Ops: gemm_bt / layernorm / gelu / add / add_bias — all via GPU buffer IDs.
// Goes directly to igd10iumd64.dll (same path as D3D11 games on HD 4600).
struct XcfeD3D11 {
    bool (*init)()                                           = nullptr;
    int  (*buf_alloc)(uint32_t)                              = nullptr;
    void (*buf_upload)(int, const void*, uint32_t)           = nullptr;
    void (*buf_download)(int, float*, uint32_t)              = nullptr;
    void (*buf_free)(int)                                    = nullptr;
    void (*gemm_bt)(int,int,int, uint32_t,uint32_t,uint32_t) = nullptr;
    void (*layernorm)(int,int,int,int, uint32_t,uint32_t)    = nullptr;
    void (*gelu)(int,int, uint32_t)                          = nullptr;
    void (*add)(int,int, uint32_t)                           = nullptr;
    void (*add_bias)(int,int, uint32_t,uint32_t)             = nullptr;
    bool ready = false;
};
static XcfeD3D11& xcfe_d3d11() {
    static XcfeD3D11 d = [](){
        XcfeD3D11 r;
#ifdef _WIN32
        const char* dis = getenv("KHANARY_D3D11");
        if (dis && dis[0] == '0') return r;
        HMODULE h = LoadLibraryA("d3d11_infer.dll");
        if (!h) return r;
        r.init        = (bool(*)(void))                                       GetProcAddress(h, "d3d11_infer_init");
        r.buf_alloc   = (int(*)(uint32_t))                                    GetProcAddress(h, "d3d11_alloc");
        r.buf_upload  = (void(*)(int,const void*,uint32_t))                   GetProcAddress(h, "d3d11_upload");
        r.buf_download= (void(*)(int,float*,uint32_t))                        GetProcAddress(h, "d3d11_download");
        r.buf_free    = (void(*)(int))                                         GetProcAddress(h, "d3d11_free");
        r.gemm_bt     = (void(*)(int,int,int,uint32_t,uint32_t,uint32_t))     GetProcAddress(h, "d3d11_gemm_bt");
        r.layernorm   = (void(*)(int,int,int,int,uint32_t,uint32_t))          GetProcAddress(h, "d3d11_layernorm");
        r.gelu        = (void(*)(int,int,uint32_t))                            GetProcAddress(h, "d3d11_gelu");
        r.add         = (void(*)(int,int,uint32_t))                            GetProcAddress(h, "d3d11_add");
        r.add_bias    = (void(*)(int,int,uint32_t,uint32_t))                  GetProcAddress(h, "d3d11_add_bias");
        if (r.init && r.buf_alloc && r.gemm_bt && r.gelu && r.add && r.layernorm)
            r.ready = r.init();
#endif
        return r;
    }();
    return d;
}

// Alloc/upload/kernel/download/free helper — returns 0 on success.
static int d3d11_run_gemm_bt(const float* A, const float* B, float* C,
                               uint32_t M, uint32_t N, uint32_t K) {
    auto& d = xcfe_d3d11();
    if (!d.ready) return 1;
    int bA = d.buf_alloc(M*K), bB = d.buf_alloc(N*K), bC = d.buf_alloc(M*N);
    if (!bA || !bB || !bC) {
        if (bA) d.buf_free(bA); if (bB) d.buf_free(bB); if (bC) d.buf_free(bC);
        return 1;
    }
    d.buf_upload(bA, A, M*K); d.buf_upload(bB, B, N*K);
    d.gemm_bt(bA, bB, bC, M, N, K);
    d.buf_download(bC, C, M*N);
    d.buf_free(bA); d.buf_free(bB); d.buf_free(bC);
    return 0;
}

static int d3d11_run_norm(const float* x, const float* gamma, const float* beta,
                           float* y, uint32_t S, uint32_t E) {
    auto& d = xcfe_d3d11();
    if (!d.ready || !d.layernorm) return 1;
    int bX = d.buf_alloc(S*E), bG = d.buf_alloc(E), bB = d.buf_alloc(E), bY = d.buf_alloc(S*E);
    if (!bX || !bG || !bB || !bY) {
        if (bX) d.buf_free(bX); if (bG) d.buf_free(bG);
        if (bB) d.buf_free(bB); if (bY) d.buf_free(bY);
        return 1;
    }
    static const std::vector<float> ones_e(4096, 1.f), zeros_e(4096, 0.f);
    d.buf_upload(bX, x, S*E);
    d.buf_upload(bG, gamma ? gamma : (E <= 4096 ? ones_e.data()  : nullptr), E);
    d.buf_upload(bB, beta  ? beta  : (E <= 4096 ? zeros_e.data() : nullptr), E);
    d.layernorm(bX, bG, bB, bY, S, E);
    d.buf_download(bY, y, S*E);
    d.buf_free(bX); d.buf_free(bG); d.buf_free(bB); d.buf_free(bY);
    return 0;
}

static int d3d11_run_gelu(const float* x, float* y, uint32_t n) {
    auto& d = xcfe_d3d11();
    if (!d.ready || !d.gelu) return 1;
    int bX = d.buf_alloc(n), bY = d.buf_alloc(n);
    if (!bX || !bY) { if (bX) d.buf_free(bX); if (bY) d.buf_free(bY); return 1; }
    d.buf_upload(bX, x, n);
    d.gelu(bX, bY, n);
    d.buf_download(bY, y, n);
    d.buf_free(bX); d.buf_free(bY);
    return 0;
}

static int d3d11_run_add(const float* a, const float* b_in, float* dst, uint32_t n) {
    auto& d = xcfe_d3d11();
    if (!d.ready || !d.add) return 1;
    int bA = d.buf_alloc(n), bB = d.buf_alloc(n);
    if (!bA || !bB) { if (bA) d.buf_free(bA); if (bB) d.buf_free(bB); return 1; }
    d.buf_upload(bA, a, n); d.buf_upload(bB, b_in, n);
    d.add(bA, bB, n);       // bB = bB + bA = src1 + src0
    d.buf_download(bB, dst, n);
    d.buf_free(bA); d.buf_free(bB);
    return 0;
}

// ── XVM Fiber fallback (25 GFLOPS) ───────────────────────────────────────────
// Set KHANARY_XVM_FIBER=<path to xvm_fiber.dll>.
// DLL must export: xcfe_xvm_gemm_bt_f32(A,B,C,M,N,K)->int
//                  xcfe_xvm_op_f32(op,src,n,dst,n)->int  (op="gelu","norm","add")
typedef int (*xvm_gemm_fn)(const float*, const float*, float*, uint32_t, uint32_t, uint32_t);
typedef int (*xvm_op_fn)(const char*, const float*, uint32_t, float*, uint32_t);
struct XcfeXVM { xvm_gemm_fn gemm = nullptr; xvm_op_fn op = nullptr; };
static XcfeXVM& xcfe_xvm() {
    static XcfeXVM x = [](){
        XcfeXVM r;
#ifdef _WIN32
        const char* path = getenv("KHANARY_XVM_FIBER");
        if (!path) return r;
        HMODULE h = LoadLibraryA(path);
        if (!h) return r;
        r.gemm = (xvm_gemm_fn) GetProcAddress(h, "xcfe_xvm_gemm_bt_f32");
        r.op   = (xvm_op_fn)   GetProcAddress(h, "xcfe_xvm_op_f32");
#endif
        return r;
    }();
    return x;
}

// ── OpenCL fallback (21 GFLOPS) ──────────────────────────────────────────────
// Set KHANARY_OPENCL_DLL=<path to xcfe_cl.dll>.
// DLL must export: xcfe_cl_gemm_bt_f32(A,B,C,M,N,K)->int
//                  xcfe_cl_op_f32(op,src,n,dst,n)->int
typedef int (*cl_gemm_fn)(const float*, const float*, float*, uint32_t, uint32_t, uint32_t);
typedef int (*cl_op_fn)(const char*, const float*, uint32_t, float*, uint32_t);
struct XcfeCL { cl_gemm_fn gemm = nullptr; cl_op_fn op = nullptr; };
static XcfeCL& xcfe_cl() {
    static XcfeCL x = [](){
        XcfeCL r;
#ifdef _WIN32
        const char* path = getenv("KHANARY_OPENCL_DLL");
        if (!path) return r;
        HMODULE h = LoadLibraryA(path);
        if (!h) return r;
        r.gemm = (cl_gemm_fn) GetProcAddress(h, "xcfe_cl_gemm_bt_f32");
        r.op   = (cl_op_fn)   GetProcAddress(h, "xcfe_cl_op_f32");
#endif
        return r;
    }();
    return x;
}

// ── WebGL2 fallback ───────────────────────────────────────────────────────────
// Set KHANARY_WEBGL2_DLL=<path to xcfe_webgl2.dll>.
// DLL must export: xcfe_webgl2_gemm_bt_f32(A,B,C,M,N,K)->int
//                  xcfe_webgl2_op_f32(op,src,n,dst,n)->int
typedef int (*webgl2_gemm_fn)(const float*, const float*, float*, uint32_t, uint32_t, uint32_t);
typedef int (*webgl2_op_fn)(const char*, const float*, uint32_t, float*, uint32_t);
struct XcfeWebGL2 { webgl2_gemm_fn gemm = nullptr; webgl2_op_fn op = nullptr; };
static XcfeWebGL2& xcfe_webgl2() {
    static XcfeWebGL2 x = [](){
        XcfeWebGL2 r;
#ifdef _WIN32
        const char* path = getenv("KHANARY_WEBGL2_DLL");
        if (!path) return r;
        HMODULE h = LoadLibraryA(path);
        if (!h) return r;
        r.gemm = (webgl2_gemm_fn) GetProcAddress(h, "xcfe_webgl2_gemm_bt_f32");
        r.op   = (webgl2_op_fn)   GetProcAddress(h, "xcfe_webgl2_op_f32");
#endif
        return r;
    }();
    return x;
}

// ── Glyph name table (all 17 ops the GL backend knows) ────────────────────────
static const char* xcfe_glyph_name(ggml_op op) {
    switch (op) {
        case GGML_OP_MUL_MAT:  return "mul_mat";
        case GGML_OP_GET_ROWS: return "get_rows";
        case GGML_OP_NORM:     return "norm";
        case GGML_OP_RMS_NORM: return "rms_norm";
        case GGML_OP_ADD:      return "add";
        case GGML_OP_SUB:      return "sub";
        case GGML_OP_MUL:      return "mul";
        case GGML_OP_SOFT_MAX: return "soft_max";
        case GGML_OP_ROPE:     return "rope";
        case GGML_OP_CONCAT:   return "concat";
        case GGML_OP_CPY:      return "cpy";
        case GGML_OP_UNARY:    return "unary";
        default:               return nullptr;
    }
}

// Pop→Sek: dispatch through xcfe_gl_run (OpenGL 4.3 compute).
static int xcfe_glyph_dispatch(ggml_tensor* dst) {
    xcfe_gl_run_fn gl = xcfe_gl_fn();
    if (!gl || !dst->data) return 1;

    const char* glyph = xcfe_glyph_name(dst->op);
    if (!glyph) return 1;

    float   params[8] = {};
    int     n_params  = 0;
    int64_t ne_out[4] = { dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3] };
    const float* inputs[2] = {};
    int n_inputs = 0;

    if (dst->op == GGML_OP_MUL_MAT) {
        if (!dst->src[0] || !dst->src[1]) return 1;
        inputs[0] = (const float*) dst->src[1]->data;
        inputs[1] = (const float*) dst->src[0]->data;
        n_inputs  = 2;
        ne_out[2] = dst->src[0]->ne[0];
        if (dst->src[0]->type == GGML_TYPE_Q4_0) glyph = "mul_mat_q4";
    } else {
        if (!dst->src[0]) return 1;
        inputs[0] = (const float*) dst->src[0]->data;
        n_inputs  = 1;
        if (dst->src[1]) { inputs[1] = (const float*) dst->src[1]->data; n_inputs = 2; }
        const float* op = (const float*) dst->op_params;
        switch (dst->op) {
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:  params[0] = op[0]; n_params = 1; break;
            case GGML_OP_SOFT_MAX:  params[0] = op[0]; n_params = 1; break;
            case GGML_OP_ROPE: {
                params[0] = (float)(*(const int*) op);
                params[1] = op[3]; params[2] = op[4];
                n_params  = 3;
                break;
            }
            case GGML_OP_UNARY: {
                // Pass the unary op code as param[0] so the GL shader can dispatch.
                params[0] = (float) ggml_get_unary_op(dst);
                n_params  = 1;
                break;
            }
            default: break;
        }
    }

    return gl(glyph, inputs, n_inputs, (float*) dst->data, ne_out, 4, params, n_params);
}

// ── CPU inline fallbacks ──────────────────────────────────────────────────────

static void cpu_elementwise(ggml_tensor* dst) {
    const float* a = (const float*) dst->src[0]->data;
    const float* b = dst->src[1] ? (const float*) dst->src[1]->data : nullptr;
    float*       d = (float*)       dst->data;
    const int64_t n = ggml_nelements(dst);
    switch (dst->op) {
        case GGML_OP_CPY:
            for (int64_t i = 0; i < n; i++) d[i] = a[i]; break;
        case GGML_OP_ADD:
            for (int64_t i = 0; i < n; i++) d[i] = a[i]+b[i]; break;
        case GGML_OP_SUB:
            for (int64_t i = 0; i < n; i++) d[i] = a[i]-b[i]; break;
        case GGML_OP_MUL:
            for (int64_t i = 0; i < n; i++) d[i] = a[i]*b[i]; break;
        default: break;
    }
}

static void cpu_norm(ggml_tensor* dst) {
    const float* x   = (const float*) dst->src[0]->data;
    float*       y   = (float*)        dst->data;
    const int64_t E  = dst->ne[0];
    const int64_t S  = ggml_nelements(dst) / E;
    const float   eps = ((const float*) dst->op_params)[0];
    for (int64_t s = 0; s < S; s++) {
        const float* row = x + s*E;
        float*       out = y + s*E;
        float sum = 0.f;
        for (int64_t i = 0; i < E; i++) sum += row[i];
        float mean = sum / (float)E;
        float var  = 0.f;
        for (int64_t i = 0; i < E; i++) { float d = row[i]-mean; var += d*d; }
        float istd = 1.f / sqrtf(var/(float)E + eps);
        for (int64_t i = 0; i < E; i++) out[i] = (row[i]-mean)*istd;
    }
}

static void cpu_rms_norm(ggml_tensor* dst) {
    const float* x   = (const float*) dst->src[0]->data;
    float*       y   = (float*)        dst->data;
    const int64_t E  = dst->ne[0];
    const int64_t S  = ggml_nelements(dst) / E;
    const float   eps = ((const float*) dst->op_params)[0];
    for (int64_t s = 0; s < S; s++) {
        const float* row = x + s*E;
        float*       out = y + s*E;
        float ss = 0.f;
        for (int64_t i = 0; i < E; i++) ss += row[i]*row[i];
        float scale = 1.f / sqrtf(ss/(float)E + eps);
        for (int64_t i = 0; i < E; i++) out[i] = row[i]*scale;
    }
}

static void cpu_gelu(ggml_tensor* dst) {
    static const float SQRT_2_OVER_PI = 0.7978845608f;
    static const float COEFF          = 0.044715f;
    const float* x = (const float*) dst->src[0]->data;
    float*       y = (float*)        dst->data;
    const int64_t n = ggml_nelements(dst);
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        float k = SQRT_2_OVER_PI * (v + COEFF*v*v*v);
        // clamp to avoid tanhf overflow on extreme inputs
        if      (k >  10.f) k =  10.f;
        else if (k < -10.f) k = -10.f;
        y[i] = 0.5f*v*(1.f + tanhf(k));
    }
}

static void cpu_silu(ggml_tensor* dst) {
    const float* x = (const float*) dst->src[0]->data;
    float*       y = (float*)        dst->data;
    const int64_t n = ggml_nelements(dst);
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        y[i] = v / (1.f + expf(-v));
    }
}

static void cpu_unary(ggml_tensor* dst) {
    switch (ggml_get_unary_op(dst)) {
        case GGML_UNARY_OP_GELU: cpu_gelu(dst); return;
        case GGML_UNARY_OP_SILU: cpu_silu(dst); return;
        default: break;
    }
}

// ── Q4_0 dequant GEMM (CPU, last resort for quantized weights) ───────────────
struct xcfe_block_q4_0 { uint16_t d; uint8_t qs[16]; };

static float xcfe_fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15u) & 1u;
    uint32_t exp  = (h >> 10u) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    if (exp == 0u)  return 0.0f;
    if (exp == 31u) return sign ? -1e38f : 1e38f;
    uint32_t f32bits = (sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u);
    float v; memcpy(&v, &f32bits, sizeof(v)); return v;
}

static void xcfe_gemm_q4_0_cpu(ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];
    const int64_t K  = src0->ne[0];
    const int64_t N  = src0->ne[1];
    const int64_t M  = src1->ne[1];
    const int64_t nb = (K + 31) / 32;
    const auto*   W  = (const xcfe_block_q4_0*) src0->data;
    const auto*   A  = (const float*)            src1->data;
    auto*         D  = (float*)                  dst->data;
    for (int64_t m = 0; m < M; m++)
        for (int64_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int64_t b = 0; b < nb; b++) {
                const xcfe_block_q4_0& blk = W[n * nb + b];
                float scale = xcfe_fp16_to_f32(blk.d);
                for (int j = 0; j < 16; j++) {
                    int64_t k0 = b * 32 + j * 2, k1 = k0 + 1;
                    if (k0 < K) acc += A[m*K+k0] * scale * (float)((blk.qs[j] & 0xF) - 8);
                    if (k1 < K) acc += A[m*K+k1] * scale * (float)((blk.qs[j] >> 4)  - 8);
                }
            }
            D[m * N + n] = acc;
        }
}

// ── MUL_MAT dispatch: GL → DML → D3D11 → XVM Fiber → OpenCL → WebGL2 → CPU ──
static void ggml_backend_xcfe_gemm_f32(struct ggml_tensor* dst) {
    const struct ggml_tensor* src0 = dst->src[0];
    const struct ggml_tensor* src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t K = ne00;
    const int64_t N = ne01;
    const int64_t M = ne11;

    // Fast path: pure 2D contiguous F32 — try DML → D3D11 → XVM → OpenCL → WebGL2.
    if (ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
        nb00 == sizeof(float) && nb10 == sizeof(float) && nb0 == sizeof(float) &&
        src0->type == GGML_TYPE_F32) {

        const float* A = (const float*) src1->data;  // activations [M,K]
        const float* B = (const float*) src0->data;  // weights [N,K]
        float*       C = (float*)       dst->data;

        // 1. DirectML
        dml_gemm_bt_fn dml = xcfe_dml_bt();
        if (dml && dml(A, B, C, (unsigned)M, (unsigned)N, (unsigned)K) == 0) return;

        // 2. D3D11 cs_5_0 (native igd10iumd64.dll path)
        if (d3d11_run_gemm_bt(A, B, C, (uint32_t)M, (uint32_t)N, (uint32_t)K) == 0) return;

        // 3. XVM Fiber (25 GFLOPS)
        auto& xvm = xcfe_xvm();
        if (xvm.gemm && xvm.gemm(A, B, C, (uint32_t)M, (uint32_t)N, (uint32_t)K) == 0) return;

        // 4. OpenCL (21 GFLOPS)
        auto& cl = xcfe_cl();
        if (cl.gemm && cl.gemm(A, B, C, (uint32_t)M, (uint32_t)N, (uint32_t)K) == 0) return;

        // 5. WebGL2
        auto& wgl = xcfe_webgl2();
        if (wgl.gemm && wgl.gemm(A, B, C, (uint32_t)M, (uint32_t)N, (uint32_t)K) == 0) return;
    }

    // 6. CPU reference (handles batched + non-contiguous)
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;
    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const int64_t i03 = i3 / r3, i02 = i2 / r2;
            const char* a_base = (const char*) src0->data + i03*nb03 + i02*nb02;
            const char* b_base = (const char*) src1->data + i3 *nb13  + i2 *nb12;
            char*       d_base = (      char*) dst->data  + i3 *nb3   + i2 *nb2;
            for (int64_t m = 0; m < M; m++) {
                const float* b = (const float*)(b_base + m*nb11);
                float*       d = (      float*)(d_base + m*nb1);
                for (int64_t n = 0; n < N; n++) {
                    const float* a = (const float*)(a_base + n*nb01);
                    float acc = 0.0f;
                    for (int64_t k = 0; k < K; k++) acc += a[k] * b[k];
                    *(float*)((char*)d + n*nb0) = acc;
                }
            }
        }
    }
}

// NORM/RMS_NORM dispatch: GL → D3D11 → XVM Fiber → OpenCL → WebGL2 → CPU
static void xcfe_dispatch_norm(ggml_tensor* dst) {
    if (xcfe_glyph_dispatch(dst) == 0) return;

    const float* x  = (const float*) dst->src[0]->data;
    float*       y  = (float*)        dst->data;
    const uint32_t E = (uint32_t) dst->ne[0];
    const uint32_t S = (uint32_t)(ggml_nelements(dst->src[0]) / E);

    // D3D11 layernorm (with identity gamma=1, beta=0 for plain NORM)
    const bool want_rms = (dst->op == GGML_OP_RMS_NORM);
    if (!want_rms && d3d11_run_norm(x, nullptr, nullptr, y, S, E) == 0) return;

    // XVM / OpenCL / WebGL2 op stubs
    auto& xvm = xcfe_xvm();
    const char* op_name = want_rms ? "rms_norm" : "norm";
    if (xvm.op && xvm.op(op_name, x, (uint32_t)ggml_nelements(dst->src[0]), y, E) == 0) return;
    auto& cl = xcfe_cl();
    if (cl.op  && cl.op(op_name,  x, (uint32_t)ggml_nelements(dst->src[0]), y, E) == 0) return;
    auto& wgl = xcfe_webgl2();
    if (wgl.op && wgl.op(op_name, x, (uint32_t)ggml_nelements(dst->src[0]), y, E) == 0) return;

    if (want_rms) cpu_rms_norm(dst);
    else          cpu_norm(dst);
}

// UNARY (GELU/SILU) dispatch: GL → D3D11 → XVM Fiber → OpenCL → WebGL2 → CPU
static void xcfe_dispatch_unary(ggml_tensor* dst) {
    if (xcfe_glyph_dispatch(dst) == 0) return;

    const ggml_unary_op uop = ggml_get_unary_op(dst);
    const float*   x = (const float*) dst->src[0]->data;
    float*         y = (float*)        dst->data;
    const uint32_t n = (uint32_t) ggml_nelements(dst);

    if (uop == GGML_UNARY_OP_GELU) {
        if (d3d11_run_gelu(x, y, n) == 0) return;
        auto& xvm = xcfe_xvm();
        if (xvm.op && xvm.op("gelu", x, n, y, n) == 0) return;
        auto& cl = xcfe_cl();
        if (cl.op  && cl.op("gelu",  x, n, y, n) == 0) return;
        auto& wgl = xcfe_webgl2();
        if (wgl.op && wgl.op("gelu", x, n, y, n) == 0) return;
    }

    cpu_unary(dst);
}

// ADD dispatch: GL → D3D11 add → CPU inline
static void xcfe_dispatch_add(ggml_tensor* dst) {
    if (xcfe_glyph_dispatch(dst) == 0) return;

    if (dst->src[0] && dst->src[1]) {
        const float* a = (const float*) dst->src[0]->data;
        const float* b = (const float*) dst->src[1]->data;
        float*       d = (float*)        dst->data;
        const uint32_t n = (uint32_t) ggml_nelements(dst);
        if (d3d11_run_add(a, b, d, n) == 0) return;
    }

    cpu_elementwise(dst);
}

// ── Backend interface ─────────────────────────────────────────────────────────

static const char* ggml_backend_xcfe_get_name(ggml_backend_t backend) {
    return "XCFE";
    GGML_UNUSED(backend);
}

static void ggml_backend_xcfe_free(ggml_backend_t backend) {
    delete (ggml_backend_xcfe_context*) backend->context;
    delete backend;
}

static enum ggml_status ggml_backend_xcfe_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor* node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) continue;

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                // GL → DML → D3D11 → XVM Fiber → OpenCL → WebGL2 → CPU
                if (xcfe_glyph_dispatch(node) != 0) {
                    if (node->src[0]->type == GGML_TYPE_Q4_0) xcfe_gemm_q4_0_cpu(node);
                    else                                       ggml_backend_xcfe_gemm_f32(node);
                }
                break;

            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
                // GL → D3D11 layernorm → XVM Fiber → OpenCL → WebGL2 → CPU
                xcfe_dispatch_norm(node);
                break;

            case GGML_OP_UNARY:
                // GL → D3D11 gelu → XVM Fiber → OpenCL → WebGL2 → CPU
                xcfe_dispatch_unary(node);
                break;

            case GGML_OP_ADD:
                // GL → D3D11 add → CPU
                xcfe_dispatch_add(node);
                break;

            case GGML_OP_CPY:
            case GGML_OP_SUB:
            case GGML_OP_MUL:
                if (xcfe_glyph_dispatch(node) != 0) cpu_elementwise(node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}

static struct ggml_backend_i xcfe_backend_i = {
    /* .get_name                = */ ggml_backend_xcfe_get_name,
    /* .free                    = */ ggml_backend_xcfe_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_xcfe_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_xcfe_guid(void) {
    static ggml_guid guid = { 0x78, 0x63, 0x66, 0x65, 0x4b, 0x48, 0x4c, 0x21,
                               0x67, 0x67, 0x6d, 0x6c, 0x78, 0x63, 0x66, 0x65 };
    return &guid;
}

ggml_backend_t ggml_backend_xcfe_init(void) {
    auto* ctx     = new ggml_backend_xcfe_context;
    auto* backend = new ggml_backend {
        /* .guid    = */ ggml_backend_xcfe_guid(),
        /* .iface   = */ xcfe_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_xcfe_reg(), 0),
        /* .context = */ ctx,
    };
    return backend;
}

bool ggml_backend_is_xcfe(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_xcfe_guid());
}

void ggml_backend_xcfe_set_n_threads(ggml_backend_t backend_xcfe, int n_threads) {
    GGML_ASSERT(ggml_backend_is_xcfe(backend_xcfe));
    ((ggml_backend_xcfe_context*) backend_xcfe->context)->n_threads = n_threads;
}

// ── Device interface ──────────────────────────────────────────────────────────

static const char* ggml_backend_xcfe_device_get_name(ggml_backend_dev_t dev) {
    return "XCFE";
    GGML_UNUSED(dev);
}

static const char* ggml_backend_xcfe_device_get_description(ggml_backend_dev_t dev) {
    return "KHANARY XCFE: GL4.3→DML→D3D11(cs_5_0)→XVM Fiber(25G)→OpenCL(21G)→WebGL2→CPU";
    GGML_UNUSED(dev);
}

static void ggml_backend_xcfe_device_get_memory(ggml_backend_dev_t dev, size_t* free, size_t* total) {
    *free = 0; *total = 0;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_xcfe_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_IGPU;
    GGML_UNUSED(dev);
}

static void ggml_backend_xcfe_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props* props) {
    props->name        = ggml_backend_xcfe_device_get_name(dev);
    props->description = ggml_backend_xcfe_device_get_description(dev);
    props->type        = ggml_backend_xcfe_device_get_type(dev);
    ggml_backend_xcfe_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id   = NULL;
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ false,
    };
}

static ggml_backend_t ggml_backend_xcfe_device_init_backend(ggml_backend_dev_t dev, const char* params) {
    return ggml_backend_xcfe_init();
    GGML_UNUSED(dev); GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_xcfe_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();
    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_xcfe_device_buffer_from_host_ptr(
        ggml_backend_dev_t dev, void* ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
    GGML_UNUSED(dev); GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_xcfe_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor* op) {
    const struct ggml_tensor* src0 = op->src[0];
    const struct ggml_tensor* src1 = op->src[1];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT: {
            const int64_t ne10      = src1->ne[0];
            const int64_t ne0       = op->ne[0];
            const int64_t ne1       = op->ne[1];
            const int64_t min_batch = 32;
            const bool src0_ok = (src0->type == GGML_TYPE_F32) ||
                                  (src0->type == GGML_TYPE_Q4_0 && ne10 % 32 == 0);
            return ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
                   src0_ok && src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch);
        }

        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            return src0 != nullptr && src0->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 && ggml_is_contiguous(src0);

        case GGML_OP_UNARY: {
            const ggml_unary_op uop = ggml_get_unary_op(op);
            return (uop == GGML_UNARY_OP_GELU || uop == GGML_UNARY_OP_SILU) &&
                   src0 != nullptr && src0->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 && ggml_is_contiguous(src0);
        }

        case GGML_OP_CPY:
            return src0 != nullptr && src0->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 && ggml_is_contiguous(src0);

        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
            return src0 != nullptr && src1 != nullptr &&
                   src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(src0) && ggml_is_contiguous(src1);

        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_xcfe_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_xcfe_device_i = {
    /* .get_name             = */ ggml_backend_xcfe_device_get_name,
    /* .get_description      = */ ggml_backend_xcfe_device_get_description,
    /* .get_memory           = */ ggml_backend_xcfe_device_get_memory,
    /* .get_type             = */ ggml_backend_xcfe_device_get_type,
    /* .get_props            = */ ggml_backend_xcfe_device_get_props,
    /* .init_backend         = */ ggml_backend_xcfe_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_xcfe_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_xcfe_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_xcfe_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xcfe_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ── Registry interface ────────────────────────────────────────────────────────

static const char* ggml_backend_xcfe_reg_get_name(ggml_backend_reg_t reg) {
    return "XCFE";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_xcfe_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_xcfe_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device ggml_backend_xcfe_device = {
        /* .iface   = */ ggml_backend_xcfe_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &ggml_backend_xcfe_device;
    GGML_UNUSED(reg); GGML_UNUSED(index);
}

static void* ggml_backend_xcfe_get_proc_address(ggml_backend_reg_t reg, const char* name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0)
        return (void*) ggml_backend_xcfe_set_n_threads;
    return NULL;
    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_xcfe_reg_i = {
    /* .get_name         = */ ggml_backend_xcfe_reg_get_name,
    /* .get_device_count = */ ggml_backend_xcfe_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xcfe_reg_get_device,
    /* .get_proc_address = */ ggml_backend_xcfe_get_proc_address,
};

ggml_backend_reg_t ggml_backend_xcfe_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xcfe_reg_i,
        /* .context     = */ NULL,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xcfe_reg)
