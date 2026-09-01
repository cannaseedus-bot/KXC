// d3d11_infer_dll.cpp — Native D3D11 cs_5_0 inference DLL
//
// Goes directly to igd10iumd64.dll (Intel HD 4600 native D3D11 path —
// the same driver WoW and other games use). No DirectML, no OpenCL,
// no D3D12 compat shim. Pure D3D11 compute shaders compiled at init time.
//
// Exports (C ABI):
//   d3d11_infer_init()                        — create device, compile shaders
//   d3d11_infer_shutdown()                    — release all resources
//   d3d11_alloc(n_floats)    → buf_id         — allocate GPU float buffer
//   d3d11_upload(id, data, n)                 — CPU→GPU
//   d3d11_download(id, out, n)                — GPU→CPU
//   d3d11_free(id)                            — release GPU buffer
//   d3d11_gemm(A,B,C, M,N,K)                 — C[M,N]=A[M,K]*B[K,N]
//   d3d11_embed(tokens,wte,wpe,out, S,E)      — token+pos embedding
//   d3d11_layernorm(x,gamma,beta,y, S,E)      — layer normalisation
//   d3d11_attention(qkv,out, S,E,H)           — causal multi-head attention
//   d3d11_gelu(x,y, numel)                    — GELU activation
//   d3d11_add_bias(y,b, rows,N)               — broadcast bias add

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "D3DCompiler.lib")

// ── Global D3D11 state ───────────────────────────────────────────────────────

static ID3D11Device*        g_dev  = nullptr;
static ID3D11DeviceContext* g_ctx  = nullptr;

static ID3D11ComputeShader* g_cs_gemm     = nullptr;
static ID3D11ComputeShader* g_cs_gemm_bt  = nullptr;
static ID3D11ComputeShader* g_cs_embed    = nullptr;
static ID3D11ComputeShader* g_cs_ln       = nullptr;
static ID3D11ComputeShader* g_cs_attn     = nullptr;
static ID3D11ComputeShader* g_cs_gelu     = nullptr;
static ID3D11ComputeShader* g_cs_add_bias = nullptr;
static ID3D11ComputeShader* g_cs_add      = nullptr;

struct GpuBuf {
    ID3D11Buffer*              buf = nullptr;
    ID3D11ShaderResourceView*  srv = nullptr;
    ID3D11UnorderedAccessView* uav = nullptr;
    uint32_t n = 0;
};
static std::unordered_map<int, GpuBuf> g_bufs;
static int g_next_id = 1;

// ── HLSL source (embedded) ───────────────────────────────────────────────────

static const char* SRC_GEMM = R"HLSL(
#define TS 16
StructuredBuffer<float>   A : register(t0);
StructuredBuffer<float>   B : register(t1);
RWStructuredBuffer<float> C : register(u0);
cbuffer GemmCB : register(b0) { uint M; uint N; uint K; uint _pad; };
groupshared float As[TS][TS];
groupshared float Bs[TS][TS];
[numthreads(TS,TS,1)]
void main(uint3 dtid:SV_DispatchThreadID, uint3 lid:SV_GroupThreadID) {
    uint row=dtid.y, col=dtid.x; float acc=0.f;
    for (uint t=0; t<(K+TS-1)/TS; ++t) {
        uint aC=t*TS+lid.x, bR=t*TS+lid.y;
        As[lid.y][lid.x] = (row<M && aC<K) ? A[row*K+aC] : 0.f;
        Bs[lid.y][lid.x] = (bR<K  && col<N) ? B[bR*N+col] : 0.f;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint k=0; k<TS; ++k) acc += As[lid.y][k]*Bs[k][lid.x];
        GroupMemoryBarrierWithGroupSync();
    }
    if (row<M && col<N) C[row*N+col] = acc;
}
)HLSL";

// C[M,N] = A[M,K] × B^T  where B is stored row-major as [N,K] (ggml MUL_MAT convention).
static const char* SRC_GEMM_BT = R"HLSL(
#define TS 16
StructuredBuffer<float>   A : register(t0);
StructuredBuffer<float>   B : register(t1);
RWStructuredBuffer<float> C : register(u0);
cbuffer GemmCB : register(b0) { uint M; uint N; uint K; uint _pad; };
groupshared float As[TS][TS];
groupshared float Bs[TS][TS];
[numthreads(TS,TS,1)]
void main(uint3 dtid:SV_DispatchThreadID, uint3 lid:SV_GroupThreadID) {
    uint row=dtid.y, col=dtid.x; float acc=0.f;
    for (uint t=0; t<(K+TS-1)/TS; ++t) {
        uint k0=t*TS;
        As[lid.y][lid.x] = (row<M && k0+lid.x<K) ? A[row*K+k0+lid.x] : 0.f;
        Bs[lid.y][lid.x] = (col<N && k0+lid.y<K) ? B[col*K+k0+lid.y] : 0.f;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint k=0; k<TS; ++k) acc += As[lid.y][k]*Bs[k][lid.x];
        GroupMemoryBarrierWithGroupSync();
    }
    if (row<M && col<N) C[row*N+col] = acc;
}
)HLSL";

static const char* SRC_EMBED = R"HLSL(
cbuffer EmbedParams : register(b0) { uint seq_len; uint n_embd; uint2 pad; };
StructuredBuffer<int>     tokens : register(t0);
StructuredBuffer<float>   wte    : register(t1);
StructuredBuffer<float>   wpe    : register(t2);
RWStructuredBuffer<float> h_out  : register(u0);
[numthreads(256,1,1)]
void main(uint3 gid:SV_GroupID, uint3 lid:SV_GroupThreadID) {
    uint i=gid.x; if (i>=seq_len) return;
    uint tok=(uint)tokens[i];
    for (uint d=lid.x; d<n_embd; d+=256)
        h_out[i*n_embd+d] = wte[tok*n_embd+d] + wpe[i*n_embd+d];
}
)HLSL";

// Inference-only layernorm: outputs y only (no xhat/inv_std).
static const char* SRC_LN = R"HLSL(
cbuffer LNParams : register(b0) { uint n_embd; uint seq_len; float eps; uint pad; };
StructuredBuffer<float>   x_in  : register(t0);
StructuredBuffer<float>   gamma : register(t1);
StructuredBuffer<float>   beta  : register(t2);
RWStructuredBuffer<float> y_out : register(u0);
groupshared float gs_s[256], gs_s2[256];
[numthreads(256,1,1)]
void main(uint3 gid:SV_GroupID, uint3 lid:SV_GroupThreadID) {
    uint s=gid.x, tid=lid.x, base=s*n_embd;
    float lsum=0.f, lsum2=0.f;
    for (uint i=tid; i<n_embd; i+=256) { float v=x_in[base+i]; lsum+=v; lsum2+=v*v; }
    gs_s[tid]=lsum; gs_s2[tid]=lsum2;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride=128; stride>=1; stride>>=1) {
        if (tid<stride) { gs_s[tid]+=gs_s[tid+stride]; gs_s2[tid]+=gs_s2[tid+stride]; }
        GroupMemoryBarrierWithGroupSync();
    }
    float mean = gs_s[0]/(float)n_embd;
    float istd = 1.0f/sqrt(gs_s2[0]/(float)n_embd - mean*mean + eps);
    for (uint i=tid; i<n_embd; i+=256) {
        float xh = (x_in[base+i]-mean)*istd;
        y_out[base+i] = gamma[i]*xh + beta[i];
    }
}
)HLSL";

static const char* SRC_ATTN = R"HLSL(
cbuffer AttnParams : register(b0) { uint seq_len; uint n_embd; uint head_dim; float scale; };
StructuredBuffer<float>   qkv      : register(t0);
RWStructuredBuffer<float> attn_out : register(u0);
RWStructuredBuffer<float> P_buf    : register(u1);
[numthreads(128,1,1)]
void main(uint3 gid:SV_GroupID, uint3 lid:SV_GroupThreadID) {
    uint h=gid.x, i=lid.x, S=seq_len, E=n_embd, D=head_dim;
    if (i>=S) return;
    uint p_row = h*S*S + i*S;
    float mx = -1e30f;
    for (uint j=0; j<=i; ++j) {
        float dot=0.f;
        for (uint d=0; d<D; ++d)
            dot += qkv[i*3*E + h*D+d] * qkv[j*3*E + E + h*D+d];
        dot *= scale; P_buf[p_row+j]=dot;
        if (dot>mx) mx=dot;
    }
    for (uint j=i+1; j<S; ++j) P_buf[p_row+j]=-1e30f;
    float sum_e=0.f;
    for (uint j=0; j<=i; ++j) { float e=exp(P_buf[p_row+j]-mx); P_buf[p_row+j]=e; sum_e+=e; }
    for (uint j=0; j<=i; ++j) P_buf[p_row+j]/=sum_e;
    for (uint j=i+1; j<S; ++j) P_buf[p_row+j]=0.f;
    for (uint d=0; d<D; ++d) {
        float acc=0.f;
        for (uint j=0; j<=i; ++j) acc += P_buf[p_row+j] * qkv[j*3*E + 2*E + h*D+d];
        attn_out[i*E + h*D+d] = acc;
    }
}
)HLSL";

static const char* SRC_GELU = R"HLSL(
static const float SQRT_2_OVER_PI = 0.7978845608f;
static const float COEFF = 0.044715f;
cbuffer GeluParams : register(b0) { uint numel; uint x_offset; uint2 pad; };
StructuredBuffer<float>   x_in : register(t0);
RWStructuredBuffer<float> y    : register(u0);
[numthreads(256,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
    uint i=tid.x; if (i>=numel) return;
    float x=x_in[i+x_offset];
    float k=SQRT_2_OVER_PI*(x + COEFF*x*x*x);
    y[i] = 0.5f*x*(1.0f + tanh(clamp(k,-10.f,10.f)));
}
)HLSL";

static const char* SRC_ADD_BIAS = R"HLSL(
cbuffer BiasParams : register(b0) { uint rows; uint N; uint2 pad; };
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float>   b : register(t0);
[numthreads(256,1,1)]
void main(uint3 t:SV_DispatchThreadID) {
    uint i=t.x; if (i>=rows*N) return;
    y[i] += b[i%N];
}
)HLSL";

static const char* SRC_ADD = R"HLSL(
cbuffer AddParams : register(b0) { uint numel; uint3 pad; };
StructuredBuffer<float>   a : register(t0);
RWStructuredBuffer<float> b : register(u0);
[numthreads(256,1,1)]
void main(uint3 t:SV_DispatchThreadID) {
    uint i=t.x; if (i>=numel) return;
    b[i] += a[i];
}
)HLSL";

// ── Internal helpers ─────────────────────────────────────────────────────────

static ID3D11ComputeShader* compile_cs(const char* src, const char* name) {
    ID3DBlob* blob = nullptr, *err = nullptr;
    HRESULT hr = D3DCompile(src, strlen(src), name, nullptr, nullptr,
                            "main", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &err);
    if (FAILED(hr)) {
        if (err) { fprintf(stderr, "[d3d11_infer] %s: %s\n", name, (char*)err->GetBufferPointer()); err->Release(); }
        return nullptr;
    }
    if (err) err->Release();
    ID3D11ComputeShader* cs = nullptr;
    g_dev->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &cs);
    blob->Release();
    return cs;
}

static GpuBuf& gbuf(int id) { return g_bufs.at(id); }

// Make a temporary constant buffer from stack data and bind it.
static ID3D11Buffer* make_cb(const void* data, UINT byte_size) {
    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth      = (byte_size + 15) & ~15u;
    bd.Usage          = D3D11_USAGE_IMMUTABLE;
    bd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA sd = {}; sd.pSysMem = data;
    ID3D11Buffer* cb = nullptr;
    g_dev->CreateBuffer(&bd, &sd, &cb);
    return cb;
}

static void null_srvs(UINT n)  { static ID3D11ShaderResourceView*  z[8]={}; g_ctx->CSSetShaderResources(0, n, z); }
static void null_uavs(UINT n)  { static ID3D11UnorderedAccessView* z[8]={}; g_ctx->CSSetUnorderedAccessViews(0, n, z, nullptr); }
static void null_cbs(UINT n)   { static ID3D11Buffer*               z[8]={}; g_ctx->CSSetConstantBuffers(0, n, z); }

// ── Public C API ─────────────────────────────────────────────────────────────

extern "C" {

__declspec(dllexport) bool d3d11_infer_init() {
    if (g_dev) return true;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                   0, &fl, 1, D3D11_SDK_VERSION, &g_dev, nullptr, &g_ctx);
    if (FAILED(hr)) { fprintf(stderr, "[d3d11_infer] D3D11CreateDevice hr=0x%08X\n", (unsigned)hr); return false; }

    g_cs_gemm     = compile_cs(SRC_GEMM,     "k_matmul");
    g_cs_gemm_bt  = compile_cs(SRC_GEMM_BT, "k_matmul_bt");
    g_cs_embed    = compile_cs(SRC_EMBED,    "k_embed");
    g_cs_ln       = compile_cs(SRC_LN,       "k_layernorm");
    g_cs_attn     = compile_cs(SRC_ATTN,     "k_attention");
    g_cs_gelu     = compile_cs(SRC_GELU,     "k_gelu");
    g_cs_add_bias = compile_cs(SRC_ADD_BIAS, "k_add_bias");
    g_cs_add      = compile_cs(SRC_ADD,      "k_add");

    if (!g_cs_gemm||!g_cs_gemm_bt||!g_cs_embed||!g_cs_ln||!g_cs_attn||!g_cs_gelu||!g_cs_add_bias||!g_cs_add) {
        fprintf(stderr, "[d3d11_infer] shader compile failed\n"); return false;
    }
    fprintf(stderr, "[d3d11_infer] init OK — D3D11 native path (igd10iumd64.dll)\n");
    return true;
}

__declspec(dllexport) void d3d11_infer_shutdown() {
    for (auto& [id,b] : g_bufs) {
        if (b.uav) b.uav->Release();
        if (b.srv) b.srv->Release();
        if (b.buf) b.buf->Release();
    }
    g_bufs.clear();
    auto rel = [](auto*& p){ if(p){p->Release();p=nullptr;} };
    rel(g_cs_gemm); rel(g_cs_gemm_bt); rel(g_cs_embed); rel(g_cs_ln); rel(g_cs_attn);
    rel(g_cs_gelu); rel(g_cs_add_bias); rel(g_cs_add);
    rel(g_ctx); rel(g_dev);
}

__declspec(dllexport) int d3d11_alloc(uint32_t n_floats) {
    GpuBuf b; b.n = n_floats;
    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth           = n_floats * 4;
    bd.Usage               = D3D11_USAGE_DEFAULT;
    bd.BindFlags           = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    bd.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bd.StructureByteStride = 4;
    if (FAILED(g_dev->CreateBuffer(&bd, nullptr, &b.buf))) return 0;
    D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
    sd.Format = DXGI_FORMAT_UNKNOWN;
    sd.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
    sd.BufferEx.NumElements = n_floats;
    g_dev->CreateShaderResourceView(b.buf, &sd, &b.srv);
    D3D11_UNORDERED_ACCESS_VIEW_DESC ud = {};
    ud.Format = DXGI_FORMAT_UNKNOWN;
    ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    ud.Buffer.NumElements = n_floats;
    g_dev->CreateUnorderedAccessView(b.buf, &ud, &b.uav);
    int id = g_next_id++;
    g_bufs[id] = b;
    return id;
}

// Upload float or int data (4 bytes per element either way).
__declspec(dllexport) void d3d11_upload(int buf_id, const void* data, uint32_t n_elements) {
    auto& b = gbuf(buf_id);
    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth      = n_elements * 4;
    bd.Usage          = D3D11_USAGE_STAGING;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    D3D11_SUBRESOURCE_DATA sd = {}; sd.pSysMem = data;
    ID3D11Buffer* stg = nullptr;
    g_dev->CreateBuffer(&bd, &sd, &stg);
    g_ctx->CopyResource(b.buf, stg);
    stg->Release();
}

__declspec(dllexport) void d3d11_download(int buf_id, float* out, uint32_t n) {
    auto& b = gbuf(buf_id);
    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth      = n * 4;
    bd.Usage          = D3D11_USAGE_STAGING;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Buffer* stg = nullptr; g_dev->CreateBuffer(&bd, nullptr, &stg);
    g_ctx->CopyResource(stg, b.buf);
    D3D11_MAPPED_SUBRESOURCE ms = {};
    g_ctx->Map(stg, 0, D3D11_MAP_READ, 0, &ms);
    memcpy(out, ms.pData, n * 4);
    g_ctx->Unmap(stg, 0);
    stg->Release();
}

__declspec(dllexport) void d3d11_free(int buf_id) {
    auto it = g_bufs.find(buf_id); if (it==g_bufs.end()) return;
    auto& b = it->second;
    if (b.uav) b.uav->Release();
    if (b.srv) b.srv->Release();
    if (b.buf) b.buf->Release();
    g_bufs.erase(it);
}

// C[M,N] = A[M,K] × B[K,N]  (both row-major)
__declspec(dllexport) void d3d11_gemm(int A, int B, int C,
                                       uint32_t M, uint32_t N, uint32_t K) {
    struct { uint32_t M,N,K,pad; } cb={M,N,K,0};
    auto* cbuf = make_cb(&cb, sizeof(cb));
    g_ctx->CSSetShader(g_cs_gemm, nullptr, 0);
    ID3D11ShaderResourceView*  srvs[] = { gbuf(A).srv, gbuf(B).srv };
    ID3D11UnorderedAccessView* uavs[] = { gbuf(C).uav };
    g_ctx->CSSetShaderResources(0, 2, srvs);
    g_ctx->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
    g_ctx->CSSetConstantBuffers(0, 1, &cbuf);
    g_ctx->Dispatch((N+15)/16, (M+15)/16, 1);
    cbuf->Release(); null_srvs(2); null_uavs(1); null_cbs(1);
}

// C[M,N] = A[M,K] × B^T  where B is [N,K] (ggml MUL_MAT: dst = src1 @ src0^T)
__declspec(dllexport) void d3d11_gemm_bt(int A, int B, int C,
                                          uint32_t M, uint32_t N, uint32_t K) {
    struct { uint32_t M,N,K,pad; } cb={M,N,K,0};
    auto* cbuf = make_cb(&cb, sizeof(cb));
    g_ctx->CSSetShader(g_cs_gemm_bt, nullptr, 0);
    ID3D11ShaderResourceView*  srvs[] = { gbuf(A).srv, gbuf(B).srv };
    ID3D11UnorderedAccessView* uavs[] = { gbuf(C).uav };
    g_ctx->CSSetShaderResources(0, 2, srvs);
    g_ctx->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
    g_ctx->CSSetConstantBuffers(0, 1, &cbuf);
    g_ctx->Dispatch((N+15)/16, (M+15)/16, 1);
    cbuf->Release(); null_srvs(2); null_uavs(1); null_cbs(1);
}

// out[S,E] = wte[tokens[i]] + wpe[i]
__declspec(dllexport) void d3d11_embed(int tokens, int wte, int wpe, int out,
                                        uint32_t S, uint32_t E) {
    struct { uint32_t seq_len,n_embd; uint32_t pad[2]; } cb={S,E,{0,0}};
    auto* cbuf = make_cb(&cb, sizeof(cb));
    g_ctx->CSSetShader(g_cs_embed, nullptr, 0);
    ID3D11ShaderResourceView*  srvs[] = { gbuf(tokens).srv, gbuf(wte).srv, gbuf(wpe).srv };
    ID3D11UnorderedAccessView* uavs[] = { gbuf(out).uav };
    g_ctx->CSSetShaderResources(0, 3, srvs);
    g_ctx->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
    g_ctx->CSSetConstantBuffers(0, 1, &cbuf);
    g_ctx->Dispatch(S, 1, 1);
    cbuf->Release(); null_srvs(3); null_uavs(1); null_cbs(1);
}

// y[S,E] = gamma * (x[S,E] - mean) / std + beta
__declspec(dllexport) void d3d11_layernorm(int x, int gamma, int beta, int y,
                                            uint32_t S, uint32_t E) {
    struct { uint32_t n_embd,seq_len; float eps; uint32_t pad; } cb={E,S,1e-5f,0};
    auto* cbuf = make_cb(&cb, sizeof(cb));
    g_ctx->CSSetShader(g_cs_ln, nullptr, 0);
    ID3D11ShaderResourceView*  srvs[] = { gbuf(x).srv, gbuf(gamma).srv, gbuf(beta).srv };
    ID3D11UnorderedAccessView* uavs[] = { gbuf(y).uav };
    g_ctx->CSSetShaderResources(0, 3, srvs);
    g_ctx->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
    g_ctx->CSSetConstantBuffers(0, 1, &cbuf);
    g_ctx->Dispatch(S, 1, 1);
    cbuf->Release(); null_srvs(3); null_uavs(1); null_cbs(1);
}

// causal multi-head attention: qkv[S,3E] → out[S,E], H=n_heads
__declspec(dllexport) void d3d11_attention(int qkv, int out,
                                            uint32_t S, uint32_t E, uint32_t H) {
    uint32_t D = E / H;
    struct { uint32_t seq_len,n_embd,head_dim; float scale; } cb={S,E,D,1.f/sqrtf((float)D)};
    auto* cbuf = make_cb(&cb, sizeof(cb));
    int pbuf = d3d11_alloc(H * S * S);
    g_ctx->CSSetShader(g_cs_attn, nullptr, 0);
    ID3D11ShaderResourceView*  srvs[] = { gbuf(qkv).srv };
    ID3D11UnorderedAccessView* uavs[] = { gbuf(out).uav, gbuf(pbuf).uav };
    g_ctx->CSSetShaderResources(0, 1, srvs);
    g_ctx->CSSetUnorderedAccessViews(0, 2, uavs, nullptr);
    g_ctx->CSSetConstantBuffers(0, 1, &cbuf);
    g_ctx->Dispatch(H, 1, 1);
    cbuf->Release(); null_srvs(1); null_uavs(2); null_cbs(1);
    d3d11_free(pbuf);
}

// y[numel] = gelu(x[numel])
__declspec(dllexport) void d3d11_gelu(int x, int y, uint32_t numel) {
    struct { uint32_t numel,offset; uint32_t pad[2]; } cb={numel,0,{0,0}};
    auto* cbuf = make_cb(&cb, sizeof(cb));
    g_ctx->CSSetShader(g_cs_gelu, nullptr, 0);
    ID3D11ShaderResourceView*  srvs[] = { gbuf(x).srv };
    ID3D11UnorderedAccessView* uavs[] = { gbuf(y).uav };
    g_ctx->CSSetShaderResources(0, 1, srvs);
    g_ctx->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
    g_ctx->CSSetConstantBuffers(0, 1, &cbuf);
    g_ctx->Dispatch((numel+255)/256, 1, 1);
    cbuf->Release(); null_srvs(1); null_uavs(1); null_cbs(1);
}

// y[rows,N] += b[N]  (broadcast bias)
__declspec(dllexport) void d3d11_add_bias(int y, int b, uint32_t rows, uint32_t N) {
    struct { uint32_t rows,N; uint32_t pad[2]; } cb={rows,N,{0,0}};
    auto* cbuf = make_cb(&cb, sizeof(cb));
    g_ctx->CSSetShader(g_cs_add_bias, nullptr, 0);
    ID3D11ShaderResourceView*  srvs[] = { gbuf(b).srv };
    ID3D11UnorderedAccessView* uavs[] = { gbuf(y).uav };
    g_ctx->CSSetShaderResources(0, 1, srvs);
    g_ctx->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
    g_ctx->CSSetConstantBuffers(0, 1, &cbuf);
    g_ctx->Dispatch((rows*N+255)/256, 1, 1);
    cbuf->Release(); null_srvs(1); null_uavs(1); null_cbs(1);
}

// b[numel] += a[numel]  (residual add)
__declspec(dllexport) void d3d11_add(int a, int b, uint32_t numel) {
    struct { uint32_t numel; uint32_t pad[3]; } cb={numel,{0,0,0}};
    auto* cbuf = make_cb(&cb, sizeof(cb));
    g_ctx->CSSetShader(g_cs_add, nullptr, 0);
    ID3D11ShaderResourceView*  srvs[] = { gbuf(a).srv };
    ID3D11UnorderedAccessView* uavs[] = { gbuf(b).uav };
    g_ctx->CSSetShaderResources(0, 1, srvs);
    g_ctx->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
    g_ctx->CSSetConstantBuffers(0, 1, &cbuf);
    g_ctx->Dispatch((numel+255)/256, 1, 1);
    cbuf->Release(); null_srvs(1); null_uavs(1); null_cbs(1);
}

} // extern "C"
