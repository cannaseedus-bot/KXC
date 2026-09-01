// cssc_matmul_run.cpp — the sm5 leg of the unified `.gemm-field` CSS-C rule.
//
// Runs the compiler-generated cs_5_0 GEMM (lowered from shaders/klsl/cssc_gemm.kuhul)
// with M,N,K,seed taken from the parsed .cssc manifest, and verifies vs CPU.
// The webgl2 leg (cssc-unified-verify.html) runs the SAME manifest's matmul in-browser.
//
// Build:  vcvars64 && cl /std:c++17 /EHsc /O2 cssc_matmul_run.cpp
// Run:    cssc_matmul_run.exe [hlsl] [M] [N] [K] [seed]

#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <sstream>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")

static std::string readFile(const char* p) { std::ifstream f(p, std::ios::binary); if (!f) return {}; std::stringstream ss; ss << f.rdbuf(); return ss.str(); }
static ID3D11Buffer* structBuf(ID3D11Device* dev, const void* data, UINT bytes, bool uav) {
    D3D11_BUFFER_DESC bd{}; bd.ByteWidth = bytes; bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = uav ? D3D11_BIND_UNORDERED_ACCESS : D3D11_BIND_SHADER_RESOURCE;
    bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; bd.StructureByteStride = 4;
    D3D11_SUBRESOURCE_DATA sd{}; sd.pSysMem = data;
    ID3D11Buffer* b = nullptr; dev->CreateBuffer(&bd, data ? &sd : nullptr, &b); return b;
}

int main(int argc, char** argv) {
    const char* hlslPath = (argc > 1) ? argv[1]
        : "C:\\Users\\canna\\.ASX.cpp\\www\\native\\cssc_gate_run\\lowered\\cssc_gemm-matmul.hlsl";
    UINT M = (argc > 2) ? atoi(argv[2]) : 64;
    UINT N = (argc > 3) ? atoi(argv[3]) : 48;
    UINT K = (argc > 4) ? atoi(argv[4]) : 56;
    uint32_t seed = (argc > 5) ? (uint32_t)strtoul(argv[5], nullptr, 10) : 12345u;

    printf("=== CSS-C unified: .gemm-field -> sm5 leg (D3D11 cs_5_0) ===\n");
    printf("[manifest] op=matmul M=%u N=%u K=%u seed=%u  (from demo.cssc)\n", M, N, K, seed);
    printf("[lower]    %s\n", hlslPath);

    std::string hlsl = readFile(hlslPath);
    if (hlsl.empty()) { printf("could not read lowered GEMM HLSL\n"); return 1; }

    D3D_FEATURE_LEVEL fl{}; ID3D11Device* dev = nullptr; ID3D11DeviceContext* ctx = nullptr;
    D3D_FEATURE_LEVEL want[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, want, 2, D3D11_SDK_VERSION, &dev, &fl, &ctx))) {
        printf("[gate] D3D11 unavailable -> sm5 not runnable\n"); return 1;
    }
    printf("[gate]     RUN @ sm5 (d3d11_cs_5_0, feature level 0x%x)\n\n", (int)fl);

    ID3DBlob* blob = nullptr; ID3DBlob* err = nullptr;
    if (FAILED(D3DCompile(hlsl.data(), hlsl.size(), "gemm", nullptr, nullptr, "CSMain", "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &err))) { if (err) printf("HLSL err: %s\n", (char*)err->GetBufferPointer()); return 1; }
    ID3D11ComputeShader* cs = nullptr; dev->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &cs);
    blob->Release(); if (err) err->Release();

    // deterministic inputs from seed (32-bit LCG matching the browser's Math.imul LCG)
    uint32_t s = seed; auto rng = [&]() { s = (uint32_t)(s * 1664525u + 1013904223u); return ((float)s / 4294967296.0f) * 2.0f - 1.0f; };
    std::vector<float> A(M * K), B(K * N), C(M * N), Ref(M * N);
    for (auto& x : A) x = rng();
    for (auto& x : B) x = rng();
    for (UINT r = 0; r < M; ++r) for (UINT c = 0; c < N; ++c) { float acc = 0; for (UINT k = 0; k < K; ++k) acc += A[r * K + k] * B[k * N + c]; Ref[r * N + c] = acc; }

    ID3D11Buffer* bA = structBuf(dev, A.data(), M * K * 4, false);
    ID3D11Buffer* bB = structBuf(dev, B.data(), K * N * 4, false);
    ID3D11Buffer* bC = structBuf(dev, nullptr, M * N * 4, true);
    ID3D11ShaderResourceView* sA = nullptr; ID3D11ShaderResourceView* sB = nullptr;
    D3D11_SHADER_RESOURCE_VIEW_DESC sv{}; sv.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX; sv.Format = DXGI_FORMAT_UNKNOWN;
    sv.BufferEx.NumElements = M * K; dev->CreateShaderResourceView(bA, &sv, &sA);
    sv.BufferEx.NumElements = K * N; dev->CreateShaderResourceView(bB, &sv, &sB);
    ID3D11UnorderedAccessView* uC = nullptr;
    D3D11_UNORDERED_ACCESS_VIEW_DESC ud{}; ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER; ud.Format = DXGI_FORMAT_UNKNOWN;
    ud.Buffer.NumElements = M * N; dev->CreateUnorderedAccessView(bC, &ud, &uC);
    struct P { UINT M, N, K, pad; } p{ M, N, K, 0 };
    D3D11_BUFFER_DESC cbd{}; cbd.ByteWidth = sizeof(P); cbd.Usage = D3D11_USAGE_DEFAULT; cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA cs0{}; cs0.pSysMem = &p; ID3D11Buffer* cb = nullptr; dev->CreateBuffer(&cbd, &cs0, &cb);
    D3D11_BUFFER_DESC stg{}; stg.ByteWidth = M * N * 4; stg.Usage = D3D11_USAGE_STAGING; stg.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Buffer* bStg = nullptr; dev->CreateBuffer(&stg, nullptr, &bStg);

    ctx->CSSetShader(cs, nullptr, 0);
    ID3D11ShaderResourceView* srvs[2] = { sA, sB }; ctx->CSSetShaderResources(0, 2, srvs);
    ctx->CSSetUnorderedAccessViews(0, 1, &uC, nullptr);
    ctx->CSSetConstantBuffers(0, 1, &cb);
    ctx->Dispatch((N + 15) / 16, (M + 15) / 16, 1);
    ctx->CopyResource(bStg, bC);
    D3D11_MAPPED_SUBRESOURCE ms; ctx->Map(bStg, 0, D3D11_MAP_READ, 0, &ms); memcpy(C.data(), ms.pData, M * N * 4); ctx->Unmap(bStg, 0);

    double maxAbs = 0.0; for (UINT i = 0; i < M * N; ++i) maxAbs = std::max(maxAbs, std::fabs((double)C[i] - (double)Ref[i]));
    printf("[run]      GEMM %ux%ux%u on d3d11_cs_5_0\n", M, N, K);
    printf("[verify]   C[0..2] = %.4f %.4f %.4f  | Ref = %.4f %.4f %.4f\n", C[0], C[1], C[2], Ref[0], Ref[1], Ref[2]);
    printf("[verify]   max abs err = %.2e\n\n", maxAbs);

    bool pass = maxAbs < 1e-3;
    printf("=== %s: sm5 leg (%.2e vs CPU) ===\n", pass ? "PASS" : "FAIL", maxAbs);
    return pass ? 0 : 1;
}
