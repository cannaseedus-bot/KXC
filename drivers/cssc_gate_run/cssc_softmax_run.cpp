// cssc_softmax_run.cpp — softmax as the second real CSS-C tensor op (sm5 leg).
// Runs the compiler-lowered cs_5_0 row-softmax (from shaders/klsl/cssc_softmax.kuhul)
// and verifies vs a CPU row-softmax reference.
//
// Build:  vcvars64 && cl /std:c++17 /EHsc /O2 cssc_softmax_run.cpp
// Run:    cssc_softmax_run.exe [hlsl] [rows] [cols] [seed]

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
        : "C:\\Users\\canna\\.ASX.cpp\\www\\native\\cssc_gate_run\\lowered\\cssc_softmax-softmax.hlsl";
    UINT rows = (argc > 2) ? atoi(argv[2]) : 64;
    UINT cols = (argc > 3) ? atoi(argv[3]) : 48;
    uint32_t seed = (argc > 4) ? (uint32_t)strtoul(argv[4], nullptr, 10) : 12345u;

    printf("=== CSS-C tensor op #2: softmax -> sm5 (D3D11 cs_5_0) ===\n");
    printf("[manifest] op=softmax rows=%u cols=%u seed=%u\n", rows, cols, seed);
    printf("[lower]    %s\n", hlslPath);
    std::string hlsl = readFile(hlslPath);
    if (hlsl.empty()) { printf("could not read lowered softmax HLSL\n"); return 1; }

    D3D_FEATURE_LEVEL fl{}; ID3D11Device* dev = nullptr; ID3D11DeviceContext* ctx = nullptr;
    D3D_FEATURE_LEVEL want[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, want, 2, D3D11_SDK_VERSION, &dev, &fl, &ctx))) {
        printf("[gate] D3D11 unavailable\n"); return 1;
    }
    printf("[gate]     RUN @ sm5 (d3d11_cs_5_0, feature level 0x%x)\n\n", (int)fl);

    ID3DBlob* blob = nullptr; ID3DBlob* err = nullptr;
    if (FAILED(D3DCompile(hlsl.data(), hlsl.size(), "softmax", nullptr, nullptr, "CSMain", "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &err))) { if (err) printf("HLSL err: %s\n", (char*)err->GetBufferPointer()); return 1; }
    ID3D11ComputeShader* cs = nullptr; dev->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &cs);
    blob->Release(); if (err) err->Release();

    uint32_t s = seed; auto rng = [&]() { s = (uint32_t)(s * 1664525u + 1013904223u); return ((float)s / 4294967296.0f) * 6.0f - 3.0f; };
    const UINT n = rows * cols;
    std::vector<float> In(n), Out(n), Ref(n);
    for (auto& x : In) x = rng();
    // CPU row-softmax reference (numerically stable)
    for (UINT r = 0; r < rows; ++r) {
        float mx = -1e30f; for (UINT j = 0; j < cols; ++j) mx = std::max(mx, In[r * cols + j]);
        float sm = 0.f; for (UINT j = 0; j < cols; ++j) sm += std::exp(In[r * cols + j] - mx);
        for (UINT j = 0; j < cols; ++j) Ref[r * cols + j] = std::exp(In[r * cols + j] - mx) / sm;
    }

    ID3D11Buffer* bIn = structBuf(dev, In.data(), n * 4, false);
    ID3D11Buffer* bOut = structBuf(dev, nullptr, n * 4, true);
    ID3D11ShaderResourceView* sIn = nullptr;
    D3D11_SHADER_RESOURCE_VIEW_DESC sv{}; sv.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX; sv.Format = DXGI_FORMAT_UNKNOWN;
    sv.BufferEx.NumElements = n; dev->CreateShaderResourceView(bIn, &sv, &sIn);
    ID3D11UnorderedAccessView* uOut = nullptr;
    D3D11_UNORDERED_ACCESS_VIEW_DESC ud{}; ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER; ud.Format = DXGI_FORMAT_UNKNOWN;
    ud.Buffer.NumElements = n; dev->CreateUnorderedAccessView(bOut, &ud, &uOut);
    struct P { UINT rows, cols, pad0, pad1; } p{ rows, cols, 0, 0 };
    D3D11_BUFFER_DESC cbd{}; cbd.ByteWidth = sizeof(P); cbd.Usage = D3D11_USAGE_DEFAULT; cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA cs0{}; cs0.pSysMem = &p; ID3D11Buffer* cb = nullptr; dev->CreateBuffer(&cbd, &cs0, &cb);
    D3D11_BUFFER_DESC stg{}; stg.ByteWidth = n * 4; stg.Usage = D3D11_USAGE_STAGING; stg.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Buffer* bStg = nullptr; dev->CreateBuffer(&stg, nullptr, &bStg);

    ctx->CSSetShader(cs, nullptr, 0);
    ctx->CSSetShaderResources(0, 1, &sIn);
    ctx->CSSetUnorderedAccessViews(0, 1, &uOut, nullptr);
    ctx->CSSetConstantBuffers(0, 1, &cb);
    ctx->Dispatch((rows + 63) / 64, 1, 1);
    ctx->CopyResource(bStg, bOut);
    D3D11_MAPPED_SUBRESOURCE ms; ctx->Map(bStg, 0, D3D11_MAP_READ, 0, &ms); memcpy(Out.data(), ms.pData, n * 4); ctx->Unmap(bStg, 0);

    // verify + check each row sums to 1
    double maxAbs = 0.0, maxRowSumErr = 0.0;
    for (UINT i = 0; i < n; ++i) maxAbs = std::max(maxAbs, std::fabs((double)Out[i] - (double)Ref[i]));
    for (UINT r = 0; r < rows; ++r) { double s2 = 0; for (UINT j = 0; j < cols; ++j) s2 += Out[r * cols + j]; maxRowSumErr = std::max(maxRowSumErr, std::fabs(s2 - 1.0)); }
    printf("[run]      softmax %ux%u on d3d11_cs_5_0\n", rows, cols);
    printf("[verify]   Out row0[0..2] = %.5f %.5f %.5f  | Ref = %.5f %.5f %.5f\n", Out[0], Out[1], Out[2], Ref[0], Ref[1], Ref[2]);
    printf("[verify]   max abs err = %.2e   max |rowsum-1| = %.2e\n\n", maxAbs, maxRowSumErr);

    bool pass = maxAbs < 1e-5 && maxRowSumErr < 1e-4;
    printf("=== %s: softmax sm5 (%.2e vs CPU, rows sum to 1) ===\n", pass ? "PASS" : "FAIL", maxAbs);
    return pass ? 0 : 1;
}
