// gemm_bench.cpp — 2x2 GEMM head-to-head on the HD 4600.
//
//   backend  x  kernel   =  { D3D11, OpenCL } x { naive, tiled(16x16) }
//
// One process, identical shapes, identical CPU reference. CORRECTNESS FIRST:
// every kernel is verified against a CPU GEMM (max relative error) on a
// rectangular shape before any GFLOPS number is trusted. Then timed on square
// 1024^3 / 512^3 (matching the earlier standalone OpenCL bench).
//
// D3D11 tiled kernel is the same 16x16 groupshared algorithm the trainer ships
// in shaders/gpt2_matmul_fwd.hlsl (CSMain), minus bias/offset/accumulate.
//
// Build (from an x64 VS BuildTools shell):
//   vcvars64 && cl /std:c++17 /EHsc /O2 gemm_bench.cpp
// OpenCL is loaded dynamically from the ICD (OpenCL.dll) — no SDK .lib needed.
//
// Answers the standing directive: "commit to the OpenCL leg only if it wins."

#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#include <random>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")

// ------------------------------------------------------------------ timing
static double qpc_now() {
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

// ------------------------------------------------------------------ CPU ref
static void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) acc += A[(size_t)r * K + k] * B[(size_t)k * N + c];
            C[(size_t)r * N + c] = acc;
        }
}
static double max_rel_err(const std::vector<float>& x, const std::vector<float>& ref) {
    double m = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = std::fabs((double)x[i] - (double)ref[i]);
        double denom = std::fabs((double)ref[i]) + 1e-6;
        m = std::max(m, d / denom);
    }
    return m;
}
static double max_abs_err(const std::vector<float>& x, const std::vector<float>& ref) {
    double m = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) m = std::max(m, std::fabs((double)x[i] - (double)ref[i]));
    return m;
}

// ------------------------------------------------------------------ HLSL
static const char* HLSL_NAIVE = R"(
cbuffer P:register(b0){uint M;uint K;uint N;uint pad;};
StructuredBuffer<float> A:register(t0);
StructuredBuffer<float> B:register(t1);
RWStructuredBuffer<float> C:register(u0);
[numthreads(16,16,1)]
void CSMain(uint3 t:SV_DispatchThreadID){
  uint r=t.y,c=t.x; if(r>=M||c>=N)return;
  float acc=0; for(uint k=0;k<K;k++) acc+=A[r*K+k]*B[k*N+c];
  C[r*N+c]=acc;
})";
static const char* HLSL_TILED = R"(
cbuffer P:register(b0){uint M;uint K;uint N;uint pad;};
StructuredBuffer<float> A:register(t0);
StructuredBuffer<float> B:register(t1);
RWStructuredBuffer<float> C:register(u0);
#define TILE 16
groupshared float As[TILE][TILE];
groupshared float Bs[TILE][TILE];
[numthreads(TILE,TILE,1)]
void CSMain(uint3 gid:SV_GroupID,uint3 lid:SV_GroupThreadID){
  uint row=gid.y*TILE+lid.y, col=gid.x*TILE+lid.x;
  float acc=0; uint nt=(K+TILE-1)/TILE;
  for(uint t=0;t<nt;t++){
    uint ac=t*TILE+lid.x; As[lid.y][lid.x]=(row<M&&ac<K)?A[row*K+ac]:0.f;
    uint br=t*TILE+lid.y; Bs[lid.y][lid.x]=(br<K&&col<N)?B[br*N+col]:0.f;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for(uint k=0;k<TILE;k++) acc+=As[lid.y][k]*Bs[k][lid.x];
    GroupMemoryBarrierWithGroupSync();
  }
  if(row<M&&col<N) C[row*N+col]=acc;
})";

// ------------------------------------------------------------------ D3D11
struct D3D {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    D3D_FEATURE_LEVEL fl{};
    bool init() {
        D3D_FEATURE_LEVEL want[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
        HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
            want, 2, D3D11_SDK_VERSION, &dev, &fl, &ctx);
        return SUCCEEDED(hr);
    }
};
static ID3D11ComputeShader* compileCS(ID3D11Device* dev, const char* src, double* compile_ms) {
    ID3DBlob* blob = nullptr; ID3DBlob* err = nullptr;
    double t0 = qpc_now();
    HRESULT hr = D3DCompile(src, strlen(src), "cs", nullptr, nullptr, "CSMain", "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &err);
    *compile_ms = (qpc_now() - t0) * 1e3;
    if (FAILED(hr)) { if (err) printf("HLSL err: %s\n", (char*)err->GetBufferPointer()); return nullptr; }
    ID3D11ComputeShader* cs = nullptr;
    dev->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &cs);
    blob->Release(); if (err) err->Release();
    return cs;
}
static ID3D11Buffer* structBuf(ID3D11Device* dev, const void* data, UINT bytes, bool uav) {
    D3D11_BUFFER_DESC bd{}; bd.ByteWidth = bytes; bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = uav ? D3D11_BIND_UNORDERED_ACCESS : D3D11_BIND_SHADER_RESOURCE;
    bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; bd.StructureByteStride = 4;
    D3D11_SUBRESOURCE_DATA sd{}; sd.pSysMem = data;
    ID3D11Buffer* b = nullptr; dev->CreateBuffer(&bd, data ? &sd : nullptr, &b); return b;
}
// returns GFLOPS; fills out C
static double d3d11_gemm(D3D& d, ID3D11ComputeShader* cs, const std::vector<float>& A,
    const std::vector<float>& B, std::vector<float>& C, int M, int N, int K, int iters) {
    ID3D11Buffer* bA = structBuf(d.dev, A.data(), (UINT)(A.size() * 4), false);
    ID3D11Buffer* bB = structBuf(d.dev, B.data(), (UINT)(B.size() * 4), false);
    ID3D11Buffer* bC = structBuf(d.dev, nullptr, (UINT)(C.size() * 4), true);
    ID3D11ShaderResourceView* sA = nullptr; ID3D11ShaderResourceView* sB = nullptr;
    D3D11_SHADER_RESOURCE_VIEW_DESC sv{}; sv.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX; sv.Format = DXGI_FORMAT_UNKNOWN;
    sv.BufferEx.NumElements = (UINT)A.size(); d.dev->CreateShaderResourceView(bA, &sv, &sA);
    sv.BufferEx.NumElements = (UINT)B.size(); d.dev->CreateShaderResourceView(bB, &sv, &sB);
    ID3D11UnorderedAccessView* uC = nullptr;
    D3D11_UNORDERED_ACCESS_VIEW_DESC ud{}; ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER; ud.Format = DXGI_FORMAT_UNKNOWN;
    ud.Buffer.NumElements = (UINT)C.size(); d.dev->CreateUnorderedAccessView(bC, &ud, &uC);
    struct P { UINT M, K, N, pad; } p{ (UINT)M,(UINT)K,(UINT)N,0 };
    D3D11_BUFFER_DESC cbd{}; cbd.ByteWidth = sizeof(P); cbd.Usage = D3D11_USAGE_DEFAULT; cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA cs0{}; cs0.pSysMem = &p; ID3D11Buffer* cb = nullptr; d.dev->CreateBuffer(&cbd, &cs0, &cb);
    D3D11_BUFFER_DESC stg{}; stg.ByteWidth = (UINT)(C.size() * 4); stg.Usage = D3D11_USAGE_STAGING; stg.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Buffer* bStg = nullptr; d.dev->CreateBuffer(&stg, nullptr, &bStg);

    d.ctx->CSSetShader(cs, nullptr, 0);
    ID3D11ShaderResourceView* srvs[2] = { sA, sB }; d.ctx->CSSetShaderResources(0, 2, srvs);
    d.ctx->CSSetUnorderedAccessViews(0, 1, &uC, nullptr);
    d.ctx->CSSetConstantBuffers(0, 1, &cb);
    UINT gx = (N + 15) / 16, gy = (M + 15) / 16;

    d.ctx->Dispatch(gx, gy, 1);                 // warmup
    d.ctx->CopyResource(bStg, bC); D3D11_MAPPED_SUBRESOURCE ms; d.ctx->Map(bStg, 0, D3D11_MAP_READ, 0, &ms); d.ctx->Unmap(bStg, 0);

    double t0 = qpc_now();
    for (int i = 0; i < iters; ++i) d.ctx->Dispatch(gx, gy, 1);
    d.ctx->CopyResource(bStg, bC);
    d.ctx->Map(bStg, 0, D3D11_MAP_READ, 0, &ms);   // blocks until GPU done
    memcpy(C.data(), ms.pData, C.size() * 4);
    d.ctx->Unmap(bStg, 0);
    double sec = (qpc_now() - t0) / iters;

    bA->Release(); bB->Release(); bC->Release(); sA->Release(); sB->Release(); uC->Release(); cb->Release(); bStg->Release();
    return 2.0 * M * N * K / sec / 1e9;
}

// ------------------------------------------------------------------ OpenCL (dynamic)
typedef int cl_int; typedef unsigned cl_uint; typedef unsigned long long cl_ulong;
typedef void *clh;
typedef cl_int(__stdcall* t_GetPlatformIDs)(cl_uint, clh*, cl_uint*);
typedef cl_int(__stdcall* t_GetDeviceIDs)(clh, cl_ulong, cl_uint, clh*, cl_uint*);
typedef clh(__stdcall* t_CreateContext)(void*, cl_uint, clh*, void*, void*, cl_int*);
typedef clh(__stdcall* t_CreateCommandQueue)(clh, clh, cl_ulong, cl_int*);
typedef clh(__stdcall* t_CreateProgramWithSource)(clh, cl_uint, const char**, size_t*, cl_int*);
typedef cl_int(__stdcall* t_BuildProgram)(clh, cl_uint, clh*, const char*, void*, void*);
typedef cl_int(__stdcall* t_GetProgramBuildInfo)(clh, clh, cl_uint, size_t, void*, size_t*);
typedef clh(__stdcall* t_CreateKernel)(clh, const char*, cl_int*);
typedef clh(__stdcall* t_CreateBuffer)(clh, cl_ulong, size_t, void*, cl_int*);
typedef cl_int(__stdcall* t_EnqueueWriteBuffer)(clh, clh, cl_uint, size_t, size_t, const void*, cl_uint, void*, void*);
typedef cl_int(__stdcall* t_EnqueueReadBuffer)(clh, clh, cl_uint, size_t, size_t, void*, cl_uint, void*, void*);
typedef cl_int(__stdcall* t_SetKernelArg)(clh, cl_uint, size_t, const void*);
typedef cl_int(__stdcall* t_EnqueueNDRangeKernel)(clh, clh, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, void*, void*);
typedef cl_int(__stdcall* t_Finish)(clh);

struct CL {
    t_GetPlatformIDs GetPlatformIDs; t_GetDeviceIDs GetDeviceIDs; t_CreateContext CreateContext;
    t_CreateCommandQueue CreateCommandQueue; t_CreateProgramWithSource CreateProgramWithSource;
    t_BuildProgram BuildProgram; t_GetProgramBuildInfo GetProgramBuildInfo; t_CreateKernel CreateKernel;
    t_CreateBuffer CreateBuffer; t_EnqueueWriteBuffer EnqueueWriteBuffer; t_EnqueueReadBuffer EnqueueReadBuffer;
    t_SetKernelArg SetKernelArg; t_EnqueueNDRangeKernel EnqueueNDRangeKernel; t_Finish Finish;
    clh ctx = nullptr, q = nullptr, dev = nullptr;
    bool load() {
        HMODULE h = LoadLibraryA("OpenCL.dll"); if (!h) return false;
        #define G(n) n = (t_##n)GetProcAddress(h,"cl" #n); if(!n) return false;
        G(GetPlatformIDs) G(GetDeviceIDs) G(CreateContext) G(CreateCommandQueue) G(CreateProgramWithSource)
        G(BuildProgram) G(GetProgramBuildInfo) G(CreateKernel) G(CreateBuffer) G(EnqueueWriteBuffer)
        G(EnqueueReadBuffer) G(SetKernelArg) G(EnqueueNDRangeKernel) G(Finish)
        #undef G
        clh pl[4]; cl_uint np; GetPlatformIDs(4, pl, &np);
        clh dv[4]; cl_uint nd; GetDeviceIDs(pl[0], 4 /*GPU*/, 4, dv, &nd); dev = dv[0];
        cl_int e; ctx = CreateContext(nullptr, 1, &dev, nullptr, nullptr, &e);
        q = CreateCommandQueue(ctx, dev, 0, &e); return ctx && q;
    }
    // returns GFLOPS; fills C. tiled => local {16,16} and global rounded up.
    double gemm(const char* src, const std::vector<float>& A, const std::vector<float>& B,
        std::vector<float>& C, int M, int N, int K, int iters, bool tiled) {
        cl_int e; const char* s = src; size_t len = strlen(src);
        clh prog = CreateProgramWithSource(ctx, 1, &s, &len, &e);
        int be = BuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);
        if (be != 0) { char log[4096]{}; GetProgramBuildInfo(prog, dev, 0x1183, sizeof(log), log, nullptr); printf("CL build err %d: %s\n", be, log); return 0; }
        clh k = CreateKernel(prog, "gemm", &e);
        clh bA = CreateBuffer(ctx, 1, A.size() * 4, nullptr, &e);
        clh bB = CreateBuffer(ctx, 1, B.size() * 4, nullptr, &e);
        clh bC = CreateBuffer(ctx, 1, C.size() * 4, nullptr, &e);
        EnqueueWriteBuffer(q, bA, 1, 0, A.size() * 4, A.data(), 0, nullptr, nullptr);
        EnqueueWriteBuffer(q, bB, 1, 0, B.size() * 4, B.data(), 0, nullptr, nullptr);
        SetKernelArg(k, 0, sizeof(clh), &bA); SetKernelArg(k, 1, sizeof(clh), &bB); SetKernelArg(k, 2, sizeof(clh), &bC);
        cl_int m = M, n = N, kk = K;
        SetKernelArg(k, 3, sizeof(cl_int), &m); SetKernelArg(k, 4, sizeof(cl_int), &n); SetKernelArg(k, 5, sizeof(cl_int), &kk);
        size_t gN = tiled ? ((N + 15) / 16) * 16 : (size_t)N;
        size_t gM = tiled ? ((M + 15) / 16) * 16 : (size_t)M;
        size_t g[2] = { gN, gM }; size_t l[2] = { 16, 16 };
        for (int w = 0; w < 3; ++w) EnqueueNDRangeKernel(q, k, 2, nullptr, g, tiled ? l : nullptr, 0, nullptr, nullptr);
        Finish(q);
        double t0 = qpc_now();
        for (int i = 0; i < iters; ++i) EnqueueNDRangeKernel(q, k, 2, nullptr, g, tiled ? l : nullptr, 0, nullptr, nullptr);
        Finish(q);
        double sec = (qpc_now() - t0) / iters;
        EnqueueReadBuffer(q, bC, 1, 0, C.size() * 4, C.data(), 0, nullptr, nullptr); Finish(q);
        return 2.0 * M * N * K / sec / 1e9;
    }
};
static const char* CL_NAIVE = R"(
__kernel void gemm(__global const float*A,__global const float*B,__global float*C,const int M,const int N,const int K){
  int r=get_global_id(1),c=get_global_id(0); if(r>=M||c>=N)return;
  float acc=0; for(int k=0;k<K;k++) acc+=A[r*K+k]*B[k*N+c]; C[r*N+c]=acc;
})";
static const char* CL_TILED = R"(
#define TILE 16
__kernel void gemm(__global const float*A,__global const float*B,__global float*C,const int M,const int N,const int K){
  __local float As[TILE][TILE]; __local float Bs[TILE][TILE];
  int lr=get_local_id(1),lc=get_local_id(0);
  int row=get_group_id(1)*TILE+lr, col=get_group_id(0)*TILE+lc;
  float acc=0; int nt=(K+TILE-1)/TILE;
  for(int t=0;t<nt;t++){
    int ac=t*TILE+lc; As[lr][lc]=(row<M&&ac<K)?A[row*K+ac]:0.f;
    int br=t*TILE+lr; Bs[lr][lc]=(br<K&&col<N)?B[br*N+col]:0.f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int k=0;k<TILE;k++) acc+=As[lr][k]*Bs[k][lc];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(row<M&&col<N) C[row*N+col]=acc;
})";

// ------------------------------------------------------------------ main
static std::vector<float> randmat(int n, unsigned seed) {
    std::mt19937 rng(seed); std::uniform_real_distribution<float> d(-1, 1);
    std::vector<float> v(n); for (auto& x : v) x = d(rng); return v;
}
int main() {
    D3D d; if (!d.init()) { printf("D3D11 init failed\n"); return 1; }
    printf("D3D11 device up (feature level 0x%x)\n", (int)d.fl);
    CL cl; if (!cl.load()) { printf("OpenCL load failed\n"); return 1; }
    printf("OpenCL ICD loaded\n\n");

    double cms;
    ID3D11ComputeShader* csNaive = compileCS(d.dev, HLSL_NAIVE, &cms);
    ID3D11ComputeShader* csTiled = compileCS(d.dev, HLSL_TILED, &cms);
    if (!csNaive || !csTiled) { printf("HLSL compile failed\n"); return 1; }

    // ---- CORRECTNESS FIRST (rectangular shape) ----
    { int M = 320, N = 192, K = 256;
      auto A = randmat(M * K, 1), B = randmat(K * N, 2);
      std::vector<float> ref(M * N); cpu_gemm(A.data(), B.data(), ref.data(), M, N, K);
      std::vector<float> C(M * N);
      printf("=== correctness vs CPU ref (%dx%dx%d)  [max abs err | max rel err] ===\n", M, N, K);
      d3d11_gemm(d, csNaive, A, B, C, M, N, K, 1); printf("  D3D11  naive : %.2e | %.2e\n", max_abs_err(C, ref), max_rel_err(C, ref));
      d3d11_gemm(d, csTiled, A, B, C, M, N, K, 1); printf("  D3D11  tiled : %.2e | %.2e\n", max_abs_err(C, ref), max_rel_err(C, ref));
      cl.gemm(CL_NAIVE, A, B, C, M, N, K, 1, false);  printf("  OpenCL naive : %.2e | %.2e\n", max_abs_err(C, ref), max_rel_err(C, ref));
      cl.gemm(CL_TILED, A, B, C, M, N, K, 1, true);   printf("  OpenCL tiled : %.2e | %.2e\n", max_abs_err(C, ref), max_rel_err(C, ref));
      printf("  (abs err ~1e-3 on values up to ~16 = correct f32; rel err near zero-crossings is a metric artifact)\n\n");
    }

    // ---- BENCHMARK (square) ----
    int shapes[][2] = { {1024, 10}, {512, 30} };
    for (auto& sh : shapes) {
        int S = sh[0], iters = sh[1];
        auto A = randmat(S * S, 1), B = randmat(S * S, 2); std::vector<float> C(S * S);
        printf("=== GEMM %dx%dx%d  (%d iters)  GFLOPS ===\n", S, S, S, iters);
        double a = d3d11_gemm(d, csNaive, A, B, C, S, S, S, iters);
        double b = d3d11_gemm(d, csTiled, A, B, C, S, S, S, iters);
        double c = cl.gemm(CL_NAIVE, A, B, C, S, S, S, iters, false);
        double e = cl.gemm(CL_TILED, A, B, C, S, S, S, iters, true);
        printf("            naive     tiled\n");
        printf("  D3D11  %8.1f  %8.1f\n", a, b);
        printf("  OpenCL %8.1f  %8.1f\n\n", c, e);
    }
    return 0;
}
