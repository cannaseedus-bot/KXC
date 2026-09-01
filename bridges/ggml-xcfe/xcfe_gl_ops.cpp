// xcfe_gl_ops.cpp — KHANARY GL tensor runtime: OpenGL 4.3 compute shader backend.
//
// Contract (see ggml-xcfe.cpp):
//   int xcfe_gl_run(const char* op, const float* const* inputs, int n_inputs,
//                   float* out, const int64_t* ne_out, int n_dims,
//                   const float* params, int n_params);
//     returns 0 = GPU handled; nonzero = caller falls back to its CPU reference.
//
// Driver: native OpenGL 4.3 compute shaders.  opengl32.dll loaded lazily —
// zero external DLL dependencies beyond system libraries.  Headless WGL context
// via a hidden HWND (no visible window).
//
// Uniform / SSBO binding layout (matches the retired WGSL build):
//   binding 0  UBO  Params   (std140, 48 bytes padded to 64)
//   binding 1  SSBO in0      (read-only)
//   binding 2  SSBO in1      (read-only, dummy 16 bytes for unary ops)
//   binding 3  SSBO buf_out  (read-write)
//
// Ops: mul_mat, get_rows, norm, rms_norm, gelu, gelu_quick, silu, relu, tanh,
//      sigmoid, add, sub, mul, soft_max, rope, concat, cpy.

#include <windows.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <map>

// ─── GL type aliases (no gl.h needed) ────────────────────────────────────────
typedef unsigned int  GLenum;
typedef int           GLint;
typedef unsigned int  GLuint;
typedef int64_t       GLsizeiptr;
typedef ptrdiff_t     GLintptr;
typedef char          GLchar;
typedef unsigned int  GLbitfield;

// ─── GL constants ────────────────────────────────────────────────────────────
#define GL_COMPUTE_SHADER              0x91B9
#define GL_SHADER_STORAGE_BUFFER       0x90D2
#define GL_UNIFORM_BUFFER              0x8A11
#define GL_DYNAMIC_DRAW                0x88E8
#define GL_COMPILE_STATUS              0x8B81
#define GL_LINK_STATUS                 0x8B82
#define GL_INFO_LOG_LENGTH             0x8B84
// Use ALL_BARRIER_BITS: covers shader→client readback (GL_BUFFER_UPDATE_BARRIER_BIT).
#define GL_ALL_BARRIER_BITS            0xFFFFFFFFu

// WGL attribs for wglCreateContextAttribsARB
#define WGL_CONTEXT_MAJOR_VERSION_ARB  0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB  0x2092
#define WGL_CONTEXT_PROFILE_MASK_ARB   0x9126
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001

// ─── GL function pointer typedefs ────────────────────────────────────────────
typedef HGLRC  (WINAPI * PFN_wglCreateContext)(HDC);
typedef BOOL   (WINAPI * PFN_wglDeleteContext)(HGLRC);
typedef BOOL   (WINAPI * PFN_wglMakeCurrent)(HDC, HGLRC);
typedef PROC   (WINAPI * PFN_wglGetProcAddress)(LPCSTR);
typedef HGLRC  (WINAPI * PFN_wglGetCurrentContext)(void);
typedef HGLRC  (WINAPI * PFN_wglCreateContextAttribsARB)(HDC, HGLRC, const int*);

typedef GLuint (*PFN_glCreateShader)(GLenum);
typedef void   (*PFN_glShaderSource)(GLuint, GLint, const GLchar* const*, const GLint*);
typedef void   (*PFN_glCompileShader)(GLuint);
typedef void   (*PFN_glGetShaderiv)(GLuint, GLenum, GLint*);
typedef void   (*PFN_glGetShaderInfoLog)(GLuint, GLint, GLint*, GLchar*);
typedef void   (*PFN_glDeleteShader)(GLuint);
typedef GLuint (*PFN_glCreateProgram)(void);
typedef void   (*PFN_glAttachShader)(GLuint, GLuint);
typedef void   (*PFN_glLinkProgram)(GLuint);
typedef void   (*PFN_glGetProgramiv)(GLuint, GLenum, GLint*);
typedef void   (*PFN_glGetProgramInfoLog)(GLuint, GLint, GLint*, GLchar*);
typedef void   (*PFN_glDeleteProgram)(GLuint);
typedef void   (*PFN_glUseProgram)(GLuint);
typedef void   (*PFN_glGenBuffers)(GLint, GLuint*);
typedef void   (*PFN_glDeleteBuffers)(GLint, const GLuint*);
typedef void   (*PFN_glBindBuffer)(GLenum, GLuint);
typedef void   (*PFN_glBufferData)(GLenum, GLsizeiptr, const void*, GLenum);
typedef void   (*PFN_glBindBufferBase)(GLenum, GLuint, GLuint);
typedef void   (*PFN_glGetBufferSubData)(GLenum, GLintptr, GLsizeiptr, void*);
typedef void   (*PFN_glDispatchCompute)(GLuint, GLuint, GLuint);
typedef void   (*PFN_glMemoryBarrier)(GLbitfield);

// ─── Loaded GL API ───────────────────────────────────────────────────────────
static struct GlApi {
    HMODULE                         hmod              = nullptr;
    PFN_wglCreateContext            wglCreateContext  = nullptr;
    PFN_wglDeleteContext            wglDeleteContext  = nullptr;
    PFN_wglMakeCurrent              wglMakeCurrent    = nullptr;
    PFN_wglGetProcAddress           wglGetProcAddress = nullptr;
    PFN_wglGetCurrentContext        wglGetCurrentContext = nullptr;
    PFN_wglCreateContextAttribsARB  wglCreateContextAttribsARB = nullptr;
    PFN_glCreateShader              glCreateShader        = nullptr;
    PFN_glShaderSource              glShaderSource        = nullptr;
    PFN_glCompileShader             glCompileShader       = nullptr;
    PFN_glGetShaderiv               glGetShaderiv         = nullptr;
    PFN_glGetShaderInfoLog          glGetShaderInfoLog    = nullptr;
    PFN_glDeleteShader              glDeleteShader        = nullptr;
    PFN_glCreateProgram             glCreateProgram       = nullptr;
    PFN_glAttachShader              glAttachShader        = nullptr;
    PFN_glLinkProgram               glLinkProgram         = nullptr;
    PFN_glGetProgramiv              glGetProgramiv        = nullptr;
    PFN_glGetProgramInfoLog         glGetProgramInfoLog   = nullptr;
    PFN_glDeleteProgram             glDeleteProgram       = nullptr;
    PFN_glUseProgram                glUseProgram          = nullptr;
    PFN_glGenBuffers                glGenBuffers          = nullptr;
    PFN_glDeleteBuffers             glDeleteBuffers       = nullptr;
    PFN_glBindBuffer                glBindBuffer          = nullptr;
    PFN_glBufferData                glBufferData          = nullptr;
    PFN_glBindBufferBase            glBindBufferBase      = nullptr;
    PFN_glGetBufferSubData          glGetBufferSubData    = nullptr;
    PFN_glDispatchCompute           glDispatchCompute     = nullptr;
    PFN_glMemoryBarrier             glMemoryBarrier       = nullptr;
} gl;

static HWND         g_hwnd        = nullptr;
static HDC          g_hdc         = nullptr;
static HGLRC        g_ctx         = nullptr;
static bool         g_ready       = false;
static const char * g_device_desc = "OpenGL 4.3 compute (gl43_compute)";

// ─── Context initialisation ───────────────────────────────────────────────────
static bool glproc(const char* name, void** fn) {
    *fn = (void*) gl.wglGetProcAddress(name);
    if (!*fn) *fn = (void*) GetProcAddress(gl.hmod, name);
    if (!*fn) {
        fprintf(stderr, "[xcfe_gl_ops] GL proc not found: %s\n", name);
        return false;
    }
    return true;
}

static bool gl_ensure_context() {
    if (g_ready) {
        // Re-make-current in case we are on a different thread than init.
        if (gl.wglGetCurrentContext() != g_ctx)
            gl.wglMakeCurrent(g_hdc, g_ctx);
        return true;
    }

    // Load opengl32.dll and bootstrap WGL.
    gl.hmod = LoadLibraryA("opengl32.dll");
    if (!gl.hmod) { fprintf(stderr, "[xcfe_gl_ops] opengl32.dll not found\n"); return false; }

#define WGL_GET(fn) do { \
    gl.fn = (PFN_##fn) GetProcAddress(gl.hmod, #fn); \
    if (!gl.fn) { fprintf(stderr, "[xcfe_gl_ops] missing %s\n", #fn); return false; } \
} while(0)
    WGL_GET(wglCreateContext);
    WGL_GET(wglDeleteContext);
    WGL_GET(wglMakeCurrent);
    WGL_GET(wglGetProcAddress);
    WGL_GET(wglGetCurrentContext);
#undef WGL_GET

    // Hidden window for the device context.
    WNDCLASSA wc = {};
    wc.style         = CS_OWNDC;
    wc.lpfnWndProc   = DefWindowProcA;
    wc.hInstance     = GetModuleHandleA(nullptr);
    wc.lpszClassName = "xcfe_gl_cls";
    RegisterClassA(&wc); // ignore "already registered" — harmless on repeated DLL loads

    g_hwnd = CreateWindowExA(0, "xcfe_gl_cls", "", WS_POPUP | WS_DISABLED,
                              0, 0, 1, 1, nullptr, nullptr, wc.hInstance, nullptr);
    if (!g_hwnd) {
        fprintf(stderr, "[xcfe_gl_ops] CreateWindow failed (%lu)\n", GetLastError());
        return false;
    }

    g_hdc = GetDC(g_hwnd);
    if (!g_hdc) { fprintf(stderr, "[xcfe_gl_ops] GetDC failed\n"); return false; }

    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize      = sizeof(pfd);
    pfd.nVersion   = 1;
    pfd.dwFlags    = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.iLayerType = PFD_MAIN_PLANE;
    int pf = ChoosePixelFormat(g_hdc, &pfd);
    if (!pf || !SetPixelFormat(g_hdc, pf, &pfd)) {
        fprintf(stderr, "[xcfe_gl_ops] pixel format failed (%lu)\n", GetLastError());
        return false;
    }

    // Legacy context (required to load wglCreateContextAttribsARB).
    HGLRC legacy = gl.wglCreateContext(g_hdc);
    if (!legacy) { fprintf(stderr, "[xcfe_gl_ops] wglCreateContext failed (%lu)\n", GetLastError()); return false; }
    if (!gl.wglMakeCurrent(g_hdc, legacy)) {
        fprintf(stderr, "[xcfe_gl_ops] wglMakeCurrent (legacy) failed\n");
        gl.wglDeleteContext(legacy);
        return false;
    }

    // Try for a 4.3 core profile; Intel HD 4600 supports it.
    gl.wglCreateContextAttribsARB = (PFN_wglCreateContextAttribsARB)
        gl.wglGetProcAddress("wglCreateContextAttribsARB");
    if (gl.wglCreateContextAttribsARB) {
        const int attribs[] = {
            WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
            WGL_CONTEXT_MINOR_VERSION_ARB, 3,
            WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
            0
        };
        HGLRC core = gl.wglCreateContextAttribsARB(g_hdc, nullptr, attribs);
        if (core) {
            gl.wglMakeCurrent(g_hdc, core);
            gl.wglDeleteContext(legacy);
            g_ctx = core;
        } else {
            g_ctx = legacy; // compat profile is fine for compute
        }
    } else {
        g_ctx = legacy;
    }

    // Load all GL 4.3 function pointers (requires an active context).
#define GL_LOAD(fn) do { void* _p; if (!glproc(#fn, &_p)) return false; gl.fn = (PFN_##fn)_p; } while(0)
    GL_LOAD(glCreateShader);
    GL_LOAD(glShaderSource);
    GL_LOAD(glCompileShader);
    GL_LOAD(glGetShaderiv);
    GL_LOAD(glGetShaderInfoLog);
    GL_LOAD(glDeleteShader);
    GL_LOAD(glCreateProgram);
    GL_LOAD(glAttachShader);
    GL_LOAD(glLinkProgram);
    GL_LOAD(glGetProgramiv);
    GL_LOAD(glGetProgramInfoLog);
    GL_LOAD(glDeleteProgram);
    GL_LOAD(glUseProgram);
    GL_LOAD(glGenBuffers);
    GL_LOAD(glDeleteBuffers);
    GL_LOAD(glBindBuffer);
    GL_LOAD(glBufferData);
    GL_LOAD(glBindBufferBase);
    GL_LOAD(glGetBufferSubData);
    GL_LOAD(glDispatchCompute);
    GL_LOAD(glMemoryBarrier);
#undef GL_LOAD

    fprintf(stderr, "[xcfe_gl_ops] device up: %s\n", g_device_desc);
    g_ready = true;
    return true;
}

// ─── GLSL compute shader sources ─────────────────────────────────────────────
//
// Binding layout (matches WGSL build):
//   binding 0  UBO  Params  (std140)
//   binding 1  SSBO in0     (readonly)
//   binding 2  SSBO in1     (readonly)
//   binding 3  SSBO buf_out (read-write)
//
// Note: "out" is a reserved GLSL keyword — output buffer is named buf_out.

static const char GLSL_HDR[] =
    "#version 430 core\n"
    "layout(std140, binding=0) uniform Params {\n"
    "  uint ne0, ne1, ne2, ne3;\n"
    "  uint n_dims, dim, aux, pad;\n"
    "  float eps, scale, freq_base, freq_scale;\n"
    "} p;\n"
    "layout(std430, binding=1) readonly buffer In0    { float in0[];    };\n"
    "layout(std430, binding=2) readonly buffer In1    { float in1[];    };\n"
    "layout(std430, binding=3)          buffer BufOut { float buf_out[]; };\n";

static const char GLSL_ERF[] =
    "float erf_approx(float x) {\n"
    "  float ax = abs(x);\n"
    "  float t = 1.0/(1.0 + 0.3275911*ax);\n"
    "  float poly = ((((1.061405429*t - 1.453152027)*t + 1.421413741)*t\n"
    "                  - 0.284496736)*t + 0.254829592)*t;\n"
    "  float y = 1.0 - poly*exp(-ax*ax);\n"
    "  return x >= 0.0 ? y : -y;\n"
    "}\n";

static const char GLSL_NORM[] =
    "layout(local_size_x=256) in;\n"
    "void main() {\n"
    "  uint gid = gl_GlobalInvocationID.x;\n"
    "  uint rows = p.ne1 * p.ne2 * p.ne3;\n"
    "  if (gid >= rows) return;\n"
    "  uint base = gid * p.ne0;\n"
    "  float sum = 0.0;\n"
    "  for (uint j = 0u; j < p.ne0; j++) { float v = in0[base+j]; sum += v*v; }\n"
    "  float inv = 1.0 / sqrt(sum / float(p.ne0) + p.eps);\n"
    "  for (uint j = 0u; j < p.ne0; j++) {\n"
    "    float w = (p.aux != 0u) ? in1[base+j] : 1.0;\n"
    "    buf_out[base+j] = in0[base+j] * inv * w;\n"
    "  }\n"
    "}\n";

static const char GLSL_RMS_NORM[] =
    "layout(local_size_x=256) in;\n"
    "void main() {\n"
    "  uint gid = gl_GlobalInvocationID.x;\n"
    "  uint rows = p.ne1 * p.ne2 * p.ne3;\n"
    "  if (gid >= rows) return;\n"
    "  uint base = gid * p.ne0;\n"
    "  float sum = 0.0;\n"
    "  for (uint j = 0u; j < p.ne0; j++) { float v = in0[base+j]; sum += v*v; }\n"
    "  float inv = 1.0 / sqrt(sum / float(p.ne0) + p.eps);\n"
    "  for (uint j = 0u; j < p.ne0; j++) {\n"
    "    float w = (p.aux != 0u) ? in1[j] : 1.0;\n"
    "    buf_out[base+j] = in0[base+j] * inv * w;\n"
    "  }\n"
    "}\n";

static const char GLSL_SOFT_MAX[] =
    "layout(local_size_x=256) in;\n"
    "void main() {\n"
    "  uint gid = gl_GlobalInvocationID.x;\n"
    "  uint rows = p.ne1 * p.ne2 * p.ne3;\n"
    "  if (gid >= rows) return;\n"
    "  uint base = gid * p.ne0;\n"
    "  float mx = -1e30;\n"
    "  for (uint j = 0u; j < p.ne0; j++) {\n"
    "    float m = (p.aux != 0u) ? in1[base+j] : 0.0;\n"
    "    mx = max(mx, in0[base+j] * p.scale + m);\n"
    "  }\n"
    "  float sm = 0.0;\n"
    "  for (uint j = 0u; j < p.ne0; j++) {\n"
    "    float m = (p.aux != 0u) ? in1[base+j] : 0.0;\n"
    "    float e = exp(in0[base+j] * p.scale + m - mx);\n"
    "    buf_out[base+j] = e;\n"
    "    sm += e;\n"
    "  }\n"
    "  for (uint j = 0u; j < p.ne0; j++) buf_out[base+j] /= sm;\n"
    "}\n";

static const char GLSL_GET_ROWS[] =
    "layout(local_size_x=256) in;\n"
    "void main() {\n"
    "  uint gid = gl_GlobalInvocationID.x;\n"
    "  uint n = p.ne0 * p.ne1;\n"
    "  if (gid >= n) return;\n"
    "  uint c = gid % p.ne0;\n"
    "  uint r = gid / p.ne0;\n"
    "  uint idx = uint(in1[r]);\n"
    "  buf_out[gid] = (idx < p.pad) ? in0[idx * p.ne0 + c] : 0.0;\n"
    "}\n";

static const char GLSL_ROPE[] =
    "layout(local_size_x=256) in;\n"
    "void main() {\n"
    "  uint gid = gl_GlobalInvocationID.x;\n"
    "  uint rows = p.ne1 * p.ne2 * p.ne3;\n"
    "  if (gid >= rows) return;\n"
    "  uint i1  = gid % p.ne1;\n"
    "  uint pos = (p.aux != 0u) ? uint(in1[i1]) : i1;\n"
    "  uint base = gid * p.ne0;\n"
    "  uint j = 0u;\n"
    "  for (; j + 1u < p.n_dims; j += 2u) {\n"
    "    float theta = float(pos) * p.freq_scale\n"
    "                * pow(p.freq_base, -2.0 * float(j / 2u) / float(p.n_dims));\n"
    "    float c = cos(theta), s = sin(theta);\n"
    "    float x0 = in0[base+j], x1 = in0[base+j+1u];\n"
    "    buf_out[base+j]    = x0*c - x1*s;\n"
    "    buf_out[base+j+1u] = x0*s + x1*c;\n"
    "  }\n"
    "  for (; j < p.ne0; j++) buf_out[base+j] = in0[base+j];\n"
    "}\n";

static const char GLSL_CONCAT[] =
    "layout(local_size_x=256) in;\n"
    "void main() {\n"
    "  uint ne0a = p.n_dims, ne0b = p.ne0 - ne0a;\n"
    "  uint ne1a = p.pad,    ne1b = p.ne1 - ne1a;\n"
    "  uint total = (ne1a + ne1b) * p.ne0;\n"
    "  uint gid = gl_GlobalInvocationID.x;\n"
    "  if (gid >= total) return;\n"
    "  uint r = gid / p.ne0, c = gid % p.ne0;\n"
    "  if (p.dim == 1u) {\n"
    "    buf_out[gid] = (r < ne1a) ? in0[r * ne0a + c] : in1[(r - ne1a) * ne0b + c];\n"
    "  } else {\n"
    "    if (c < ne0a) buf_out[gid] = (r < ne1a) ? in0[r * ne0a + c] : 0.0;\n"
    "    else          buf_out[gid] = (r >= ne1a) ? in1[r * ne0b + (c - ne0a)] : 0.0;\n"
    "  }\n"
    "}\n";

// Tiled F32 GEMM ported from ggml-opencl mul_mm_f32_f32_l4_lm.cl
// BM=64 BN=64 BK=16 TM=4 TN=8 → 128 threads/workgroup (BM/TM * BN/TN = 16*8)
// smem_a[BK][BM] = activations tile; smem_b[BK][BN] = weights tile (each 4 KB)
// Computes: out[m*N+n] = sum_k in0[m*K+k]*in1[n*K+k]
// where in0=activations[M][K], in1=weights[N][K], out=result[M][N]
static const char GLSL_MUL_MAT[] =
    "shared float smem_a[1024];\n"          // BK*BM = 16*64
    "shared float smem_b[1024];\n"          // BK*BN = 16*64
    "layout(local_size_x=128) in;\n"
    "void main() {\n"
    "  uint M=p.ne1, N=p.ne0, K=p.ne2;\n"
    "  uint ir=gl_WorkGroupID.x, ic=gl_WorkGroupID.y;\n"
    "  uint tid=gl_LocalInvocationID.x;\n"
    "  uint th_r=tid%16u, th_c=tid/16u;\n"  // TM-block (0..15) and TN-block (0..7)
    "  uint lr=tid%4u, lc=tid/4u;\n"         // load: lr=K-vec-idx (0..3), lc=row (0..31)
    "  float sums[32], ca[4], cb[8];\n"
    "  for(int i=0;i<32;i++) sums[i]=0.0;\n"
    "  for(uint blk=0u;blk<K;blk+=16u) {\n"
    "    for(uint l=0u;l<64u;l+=32u) {\n"   // 2 passes, loadstride=LS*LV/BK=128*4/16=32
    "      uint m=lc+l, mg=ir*64u+m;\n"
    "      uint kb=blk+lr*4u;\n"
    "      bool vm=mg<M;\n"
    "      smem_a[(lr*4u+0u)*64u+m]=(vm&&(kb+0u)<K)?in0[mg*K+kb+0u]:0.0;\n"
    "      smem_a[(lr*4u+1u)*64u+m]=(vm&&(kb+1u)<K)?in0[mg*K+kb+1u]:0.0;\n"
    "      smem_a[(lr*4u+2u)*64u+m]=(vm&&(kb+2u)<K)?in0[mg*K+kb+2u]:0.0;\n"
    "      smem_a[(lr*4u+3u)*64u+m]=(vm&&(kb+3u)<K)?in0[mg*K+kb+3u]:0.0;\n"
    "    }\n"
    "    for(uint l=0u;l<64u;l+=32u) {\n"
    "      uint n=lc+l, ng=ic*64u+n;\n"
    "      uint kb=blk+lr*4u;\n"
    "      bool vn=ng<N;\n"
    "      smem_b[(lr*4u+0u)*64u+n]=(vn&&(kb+0u)<K)?in1[ng*K+kb+0u]:0.0;\n"
    "      smem_b[(lr*4u+1u)*64u+n]=(vn&&(kb+1u)<K)?in1[ng*K+kb+1u]:0.0;\n"
    "      smem_b[(lr*4u+2u)*64u+n]=(vn&&(kb+2u)<K)?in1[ng*K+kb+2u]:0.0;\n"
    "      smem_b[(lr*4u+3u)*64u+n]=(vn&&(kb+3u)<K)?in1[ng*K+kb+3u]:0.0;\n"
    "    }\n"
    "    barrier(); memoryBarrierShared();\n"
    "    for(uint ki=0u;ki<16u;ki++) {\n"
    "      for(uint j=0u;j<4u;j++) ca[j]=smem_a[ki*64u+th_r*4u+j];\n"
    "      for(uint j=0u;j<8u;j++) cb[j]=smem_b[ki*64u+th_c*8u+j];\n"
    "      for(uint cc=0u;cc<8u;cc++)\n"
    "        for(uint cr=0u;cr<4u;cr++)\n"
    "          sums[cc*4u+cr]=fma(ca[cr],cb[cc],sums[cc*4u+cr]);\n"
    "    }\n"
    "    barrier(); memoryBarrierShared();\n"
    "  }\n"
    "  uint dr=ir*64u+th_r*4u, dc=ic*64u+th_c*8u;\n"
    "  for(uint cc=0u;cc<8u;cc++)\n"
    "    for(uint cr=0u;cr<4u;cr++) {\n"
    "      uint m=dr+cr, n=dc+cc;\n"
    "      if(m<M&&n<N) buf_out[m*N+n]=sums[cc*4u+cr];\n"
    "    }\n"
    "}\n";

// Q4_0 dequant + matmul: in1 binding uses uint (packed Q4_0 bytes) instead of float.
// Q4_0 block layout: [2-byte f16 scale][16-byte nibbles], 32 elements per block.
//   Low nibble of qs[j] → element 2j in block; high nibble → element 2j+1.
//   Dequant: value = scale * (nibble - 8)
static const char GLSL_HDR_Q4[] =
    "#version 430 core\n"
    "layout(std140, binding=0) uniform Params {\n"
    "  uint ne0, ne1, ne2, ne3;\n"
    "  uint n_dims, dim, aux, pad;\n"
    "  float eps, scale, freq_base, freq_scale;\n"
    "} p;\n"
    "layout(std430, binding=1) readonly buffer In0    { float in0[];    };\n"
    "layout(std430, binding=2) readonly buffer In1Q4  { uint  in1q4[];  };\n"
    "layout(std430, binding=3)          buffer BufOut { float buf_out[]; };\n";

static const char GLSL_MUL_MAT_Q4_0[] =
    "layout(local_size_x=16, local_size_y=16) in;\n"
    "uint q4_byte(uint b) {\n"
    "  return (in1q4[b >> 2u] >> ((b & 3u) << 3u)) & 0xFFu;\n"
    "}\n"
    "float q4_dequant(uint col, uint k) {\n"
    "  uint bpr = (p.ne2 + 31u) / 32u;\n"
    "  uint blk = (col * bpr + k / 32u) * 18u;\n"
    "  float d = unpackHalf2x16(q4_byte(blk) | (q4_byte(blk + 1u) << 8u)).x;\n"
    "  uint qs_b = q4_byte(blk + 2u + (k % 32u) / 2u);\n"
    "  uint nibble = ((k & 1u) == 0u) ? (qs_b & 0xFu) : (qs_b >> 4u);\n"
    "  return d * (float(nibble) - 8.0);\n"
    "}\n"
    "void main() {\n"
    "  uint col = gl_GlobalInvocationID.x;\n"
    "  uint row = gl_GlobalInvocationID.y;\n"
    "  if (row >= p.ne1 || col >= p.ne0) return;\n"
    "  float acc = 0.0;\n"
    "  for (uint k = 0u; k < p.ne2; k++)\n"
    "    acc += in0[row * p.ne2 + k] * q4_dequant(col, k);\n"
    "  buf_out[row * p.ne0 + col] = acc;\n"
    "}\n";

// ─── Shader build helpers ────────────────────────────────────────────────────
static std::string make_unary(const char* expr, const char* helper = nullptr) {
    std::string s = GLSL_HDR;
    if (helper) s += helper;
    s += "layout(local_size_x=256) in;\n"
         "void main() {\n"
         "  uint gid = gl_GlobalInvocationID.x;\n"
         "  uint n = p.ne0 * p.ne1 * p.ne2 * p.ne3;\n"
         "  if (gid >= n) return;\n"
         "  float x = in0[gid];\n"
         "  buf_out[gid] = ";
    s += expr;
    s += ";\n}\n";
    return s;
}

static std::string make_binary(const char* expr) {
    std::string s = GLSL_HDR;
    s += "layout(local_size_x=256) in;\n"
         "void main() {\n"
         "  uint gid = gl_GlobalInvocationID.x;\n"
         "  uint n = p.ne0 * p.ne1 * p.ne2 * p.ne3;\n"
         "  if (gid >= n) return;\n"
         "  buf_out[gid] = ";
    s += expr;
    s += ";\n}\n";
    return s;
}

static std::string make_src(const char* body) {
    return std::string(GLSL_HDR) + body;
}

static std::map<std::string, std::string>& shader_sources() {
    static std::map<std::string, std::string> m = {
        { "cpy",        make_unary("x") },
        { "gelu",       make_unary("0.5*x*(1.0+erf_approx(x*0.70710678118654752440))", GLSL_ERF) },
        { "gelu_quick", make_unary("x/(1.0+exp(-1.702*x))") },
        { "silu",       make_unary("x/(1.0+exp(-x))") },
        { "relu",       make_unary("max(x,0.0)") },
        { "tanh",       make_unary("tanh(x)") },
        { "sigmoid",    make_unary("1.0/(1.0+exp(-x))") },
        { "add",        make_binary("in0[gid]+in1[gid]") },
        { "sub",        make_binary("in0[gid]-in1[gid]") },
        { "mul",        make_binary("in0[gid]*in1[gid]") },
        { "norm",       make_src(GLSL_NORM) },
        { "rms_norm",   make_src(GLSL_RMS_NORM) },
        { "soft_max",   make_src(GLSL_SOFT_MAX) },
        { "get_rows",   make_src(GLSL_GET_ROWS) },
        { "rope",       make_src(GLSL_ROPE) },
        { "concat",     make_src(GLSL_CONCAT) },
        { "mul_mat",    make_src(GLSL_MUL_MAT) },
        { "mul_mat_q4", std::string(GLSL_HDR_Q4) + GLSL_MUL_MAT_Q4_0 },
    };
    return m;
}

// ─── Program compilation + cache ─────────────────────────────────────────────
static GLuint compile_program(const std::string& src) {
    const char* cstr = src.c_str();
    GLint len = (GLint) src.size();

    GLuint sh = gl.glCreateShader(GL_COMPUTE_SHADER);
    gl.glShaderSource(sh, 1, &cstr, &len);
    gl.glCompileShader(sh);

    GLint ok = 0;
    gl.glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint loglen = 0;
        gl.glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &loglen);
        std::vector<char> log(loglen + 1);
        gl.glGetShaderInfoLog(sh, loglen, nullptr, log.data());
        fprintf(stderr, "[xcfe_gl_ops] shader compile error:\n%s\n", log.data());
        gl.glDeleteShader(sh);
        return 0;
    }

    GLuint prog = gl.glCreateProgram();
    gl.glAttachShader(prog, sh);
    gl.glLinkProgram(prog);
    gl.glDeleteShader(sh);

    gl.glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint loglen = 0;
        gl.glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &loglen);
        std::vector<char> log(loglen + 1);
        gl.glGetProgramInfoLog(prog, loglen, nullptr, log.data());
        fprintf(stderr, "[xcfe_gl_ops] program link error:\n%s\n", log.data());
        gl.glDeleteProgram(prog);
        return 0;
    }

    return prog;
}

static GLuint op_program(const char* op) {
    static std::map<std::string, GLuint> cache;
    auto it = cache.find(op);
    if (it != cache.end()) return it->second;

    auto& srcs = shader_sources();
    auto sit = srcs.find(op);
    if (sit == srcs.end()) return 0;

    GLuint prog = compile_program(sit->second);
    if (prog) cache[op] = prog;
    return prog;
}

// ─── Params struct (matches std140 UBO layout, 48 bytes + 16 pad = 64) ───────
struct Params {
    uint32_t ne0, ne1, ne2, ne3;
    uint32_t n_dims, dim, aux, pad;
    float    eps, scale, freq_base, freq_scale;
};  // 48 bytes; UBO upload is 64 bytes (std140 base alignment)

static uint64_t round_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// ─── Exported entry points ────────────────────────────────────────────────────
extern "C" __declspec(dllexport) const char* xcfe_gl_device_name() {
    return g_device_desc;
}

extern "C" __declspec(dllexport) int xcfe_gl_run(
    const char*         op,
    const float* const* inputs,
    int                 n_inputs,
    float*              out,
    const int64_t*      ne_out,
    int                 n_dims,
    const float*        params,
    int                 n_params)
{
    // OpenGL 4.3 compute is not available on Intel HD 4600 (headless context silently fails).
    // Always return nonzero so ggml-xcfe falls back to its CPU reference implementations.
    // MUL_MAT is handled separately via dml_gemm.dll / CPU gemm in ggml-xcfe.cpp.
    (void)op; (void)inputs; (void)n_inputs; (void)out; (void)ne_out; (void)n_dims; (void)params; (void)n_params;
    return 1;

    if (!gl_ensure_context()) return 1;

    const uint64_t ne0 = (ne_out && n_dims > 0) ? (uint64_t) ne_out[0] : 1;
    const uint64_t ne1 = (ne_out && n_dims > 1) ? (uint64_t) ne_out[1] : 1;
    const uint64_t ne2 = (ne_out && n_dims > 2) ? (uint64_t) ne_out[2] : 1;
    const uint64_t ne3 = (ne_out && n_dims > 3) ? (uint64_t) ne_out[3] : 1;

    GLuint prog = op_program(op);
    if (!prog) return 1;

    // Build Params — same op-specific wiring as the WGSL build.
    Params p = {};
    p.ne0 = (uint32_t) ne0; p.ne1 = (uint32_t) ne1;
    p.ne2 = (uint32_t) ne2; p.ne3 = (uint32_t) ne3;
    p.eps = 1e-5f; p.scale = 1.0f; p.freq_base = 10000.0f; p.freq_scale = 1.0f;
    p.aux = (uint32_t) (n_inputs >= 2 ? 1 : 0);

    if (params && n_params > 0) {
        if (strcmp(op, "mul_mat") == 0) {
            p.ne2 = (uint32_t) params[0];
        } else if (strcmp(op, "norm") == 0 || strcmp(op, "rms_norm") == 0) {
            p.eps = params[0];
        } else if (strcmp(op, "soft_max") == 0) {
            p.scale = params[0];
        } else if (strcmp(op, "rope") == 0 && n_params >= 3) {
            p.n_dims     = (uint32_t) params[0];
            p.freq_base  = params[1];
            p.freq_scale = params[2];
        } else if (strcmp(op, "concat") == 0 && n_params >= 3) {
            p.dim    = (uint32_t) params[0];
            p.n_dims = (uint32_t) params[1];
            p.pad    = (uint32_t) params[2];
        } else if (strcmp(op, "get_rows") == 0) {
            p.pad = (uint32_t) params[0];
        }
    }

    // Buffer sizing (same logic as the WGSL build).
    const uint64_t out_elems = ne0 * ne1 * ne2 * ne3;
    const uint64_t out_bytes = round_up(out_elems * sizeof(float), 16);
    uint64_t in0_bytes = out_bytes, in1_bytes = 16;

    if (strcmp(op, "mul_mat") == 0) {
        p.aux = 0;
        in0_bytes = round_up((uint64_t) p.ne1 * p.ne2 * sizeof(float), 16);
        in1_bytes = round_up((uint64_t) p.ne0 * p.ne2 * sizeof(float), 16);
    } else if (strcmp(op, "mul_mat_q4") == 0) {
        p.aux = 0;
        in0_bytes = round_up((uint64_t) p.ne1 * p.ne2 * sizeof(float), 16);
        // Q4_0: N rows × ceil(K/32) blocks × 18 bytes per block
        in1_bytes = round_up((uint64_t) p.ne0 * ((p.ne2 + 31u) / 32u) * 18u, 16);
    } else if (strcmp(op, "get_rows") == 0) {
        in0_bytes = round_up((uint64_t) p.pad * ne0 * sizeof(float), 16);
        in1_bytes = round_up(ne1 * sizeof(float), 16);
    } else if (strcmp(op, "concat") == 0) {
        const uint64_t ne0a = p.n_dims, ne1a = p.pad;
        const uint64_t ne0b = ne0 > ne0a ? ne0 - ne0a : 0;
        const uint64_t ne1b = ne1 > ne1a ? ne1 - ne1a : 0;
        in0_bytes = round_up(ne0a * ne1a * sizeof(float), 16);
        in1_bytes = round_up(ne0b * ne1b * sizeof(float), 16);
    } else if (strcmp(op, "norm") == 0 || strcmp(op, "rms_norm") == 0) {
        in0_bytes = out_bytes;
        // weight vector is ne0 elements (one per feature), not the full ne0*ne1*... tensor
        in1_bytes = (n_inputs >= 2) ? round_up(ne0 * sizeof(float), 16) : 16;
    } else if (strcmp(op, "soft_max") == 0) {
        in0_bytes = out_bytes;
        in1_bytes = (n_inputs >= 2) ? out_bytes : 16;
    } else if (strcmp(op, "rope") == 0) {
        in0_bytes = out_bytes;
        in1_bytes = (n_inputs >= 2) ? round_up(ne1 * sizeof(float), 16) : 16;
    } else if (strcmp(op, "add") == 0 || strcmp(op, "sub") == 0 ||
               strcmp(op, "mul") == 0) {
        in0_bytes = out_bytes;
        in1_bytes = out_bytes;
    } else {
        in0_bytes = out_bytes;
        in1_bytes = 16;
    }

    // Clamp in1_bytes to at least 16 to avoid zero-size UBO or SSBO.
    if (in1_bytes < 16) in1_bytes = 16;

    // ── Create GL buffers ────────────────────────────────────────────────────
    GLuint bufs[4];
    gl.glGenBuffers(4, bufs);
    GLuint bufP = bufs[0], buf0 = bufs[1], buf1 = bufs[2], bufO = bufs[3];

    // UBO: Params padded to 64 bytes.
    uint8_t params_buf[64] = {};
    memcpy(params_buf, &p, sizeof(p));
    gl.glBindBuffer(GL_UNIFORM_BUFFER, bufP);
    gl.glBufferData(GL_UNIFORM_BUFFER, 64, params_buf, GL_DYNAMIC_DRAW);

    // SSBO in0.
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf0);
    gl.glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr) in0_bytes,
                    inputs[0], GL_DYNAMIC_DRAW);

    // SSBO in1 — use dummy from p when second input is absent.
    const void* in1_data = (n_inputs >= 2) ? (const void*) inputs[1]
                                            : (const void*) params_buf;
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf1);
    gl.glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr) in1_bytes,
                    in1_data, GL_DYNAMIC_DRAW);

    // SSBO out — uninitialized, written by the shader.
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufO);
    gl.glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr) out_bytes,
                    nullptr, GL_DYNAMIC_DRAW);

    // ── Bind to binding points ───────────────────────────────────────────────
    gl.glBindBufferBase(GL_UNIFORM_BUFFER,        0, bufP);
    gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buf0);
    gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buf1);
    gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bufO);

    // ── Dispatch ─────────────────────────────────────────────────────────────
    gl.glUseProgram(prog);

    uint32_t gx = 1, gy = 1, gz = 1;
    if (strcmp(op, "mul_mat") == 0) {
        // ceil(N/BN) × ceil(M/BM) workgroups, 128 threads each (BN=BM=64)
        gx = (uint32_t) ((ne0 + 63) / 64);
        gy = (uint32_t) ((ne1 + 63) / 64);
    } else if (strcmp(op, "mul_mat_q4") == 0) {
        gx = (uint32_t) ((ne0 + 15) / 16);
        gy = (uint32_t) ((ne1 + 15) / 16);
    } else if (strcmp(op, "norm")     == 0 || strcmp(op, "rms_norm") == 0 ||
               strcmp(op, "soft_max") == 0 || strcmp(op, "rope")     == 0) {
        gx = (uint32_t) ((ne1 * ne2 * ne3 + 255) / 256);
    } else {
        gx = (uint32_t) ((out_elems + 255) / 256);
    }
    if (gx < 1) gx = 1;

    gl.glDispatchCompute(gx, gy, gz);

    // GL_ALL_BARRIER_BITS covers GL_BUFFER_UPDATE_BARRIER_BIT, which is required
    // to make shader writes visible to subsequent glGetBufferSubData reads.
    gl.glMemoryBarrier(GL_ALL_BARRIER_BITS);

    // ── Readback ─────────────────────────────────────────────────────────────
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufO);
    gl.glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                          (GLsizeiptr) (out_elems * sizeof(float)), out);

    // ── Cleanup ──────────────────────────────────────────────────────────────
    gl.glDeleteBuffers(4, bufs);
    return 0;
}
