// cssc_klsl_run.cpp — closes the KLSL -> HLSL -> sm5 link.
//
// Loads HLSL that was LOWERED FROM A REAL .klsl BY THE REAL COMPILER
// (shaders/klsl/cssc_scale_bias.kuhul --compile-klsl-shaders.mjs--> lowered/*.hlsl),
// resolves the CSS-C gate against the measured device (-> sm5), dispatches the
// compiler-generated cs_5_0 kernel on D3D11, and verifies vs a CPU reference of the
// SAME KLSL semantics (out[i] = in[i]*scale + bias).
//
// Nothing hand-written in the kernel path: the compiler emitted CSMain; we only run it.
//
// Build:  vcvars64 && cl /std:c++17 /EHsc /O2 cssc_klsl_run.cpp
// Run:    cssc_klsl_run.exe [lowered/cssc_scale_bias-scale_bias.hlsl]

#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")

static std::string readFile(const char* p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

struct D3D {
    ID3D11Device* dev = nullptr; ID3D11DeviceContext* ctx = nullptr; D3D_FEATURE_LEVEL fl{};
    bool init() {
        D3D_FEATURE_LEVEL want[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
        return SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
            want, 2, D3D11_SDK_VERSION, &dev, &fl, &ctx));
    }
};
static ID3D11Buffer* structBuf(ID3D11Device* dev, const void* data, UINT bytes, bool uav) {
    D3D11_BUFFER_DESC bd{}; bd.ByteWidth = bytes; bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = uav ? D3D11_BIND_UNORDERED_ACCESS : D3D11_BIND_SHADER_RESOURCE;
    bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; bd.StructureByteStride = 4;
    D3D11_SUBRESOURCE_DATA sd{}; sd.pSysMem = data;
    ID3D11Buffer* b = nullptr; dev->CreateBuffer(&bd, data ? &sd : nullptr, &b); return b;
}

int main(int argc, char** argv) {
    const char* hlslPath = (argc > 1) ? argv[1]
        : "C:\\Users\\canna\\.ASX.cpp\\www\\native\\cssc_gate_run\\lowered\\cssc_scale_bias-scale_bias.hlsl";
    printf("=== CSS-C: KLSL -> HLSL -> sm5 (compiler-generated kernel) ===\n\n");
    printf("[klsl]     source : shaders/klsl/cssc_scale_bias.kuhul\n");
    printf("[lower]    hlsl   : %s\n", hlslPath);

    std::string hlsl = readFile(hlslPath);
    if (hlsl.empty()) { printf("could not read lowered HLSL (run compile-klsl-shaders.mjs first)\n"); return 1; }
    // sanity: it must be the compiler's output, not hand-written
    if (hlsl.find("Generated HLSL from KLSL") == std::string::npos)
        printf("[warn]     HLSL header missing 'Generated HLSL from KLSL' marker\n");

    // 1. measure device -> sm5 (d3d11_cs_5_0) available?
    D3D d; bool up = d.init();
    printf("[measure]  D3D11 device: %s", up ? "UP" : "FAILED");
    if (up) printf(" (feature level 0x%x)", (int)d.fl);
    printf("\n[gate]     requires=[webgpu, sm5, cpu] -> %s\n\n",
           up ? "RUN @ sm5 (d3d11_cs_5_0)" : "would fall to cpu");
    if (!up) return 1;

    // 2. compile the COMPILER-GENERATED HLSL as cs_5_0
    ID3DBlob* blob = nullptr; ID3DBlob* err = nullptr;
    HRESULT hr = D3DCompile(hlsl.data(), hlsl.size(), "cssc_klsl", nullptr, nullptr,
        "CSMain", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &err);
    if (FAILED(hr)) { if (err) printf("HLSL err: %s\n", (char*)err->GetBufferPointer()); return 1; }
    ID3D11ComputeShader* cs = nullptr;
    d.dev->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &cs);
    blob->Release(); if (err) err->Release();
    printf("[compile]  cs_5_0 OK (entry CSMain)\n");

    // 3. buffers matching the generated bindings: in_buf(t0) out_buf(u0) Params(b0){count,scale,bias}
    const UINT N = 256; const float scale = 2.5f, bias = 1.0f;
    std::vector<float> In(N), Out(N), Ref(N);
    for (UINT i = 0; i < N; ++i) { In[i] = i * 0.1f; Ref[i] = In[i] * scale + bias; }

    ID3D11Buffer* bIn = structBuf(d.dev, In.data(), N * 4, false);
    ID3D11Buffer* bOut = structBuf(d.dev, nullptr, N * 4, true);
    ID3D11ShaderResourceView* sIn = nullptr;
    D3D11_SHADER_RESOURCE_VIEW_DESC sv{}; sv.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
    sv.Format = DXGI_FORMAT_UNKNOWN; sv.BufferEx.NumElements = N; d.dev->CreateShaderResourceView(bIn, &sv, &sIn);
    ID3D11UnorderedAccessView* uOut = nullptr;
    D3D11_UNORDERED_ACCESS_VIEW_DESC ud{}; ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    ud.Format = DXGI_FORMAT_UNKNOWN; ud.Buffer.NumElements = N; d.dev->CreateUnorderedAccessView(bOut, &ud, &uOut);
    struct P { UINT count; float scale; float bias; float pad; } p{ N, scale, bias, 0 };  // cbuffer Params b0
    D3D11_BUFFER_DESC cbd{}; cbd.ByteWidth = sizeof(P); cbd.Usage = D3D11_USAGE_DEFAULT; cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA cs0{}; cs0.pSysMem = &p; ID3D11Buffer* cb = nullptr; d.dev->CreateBuffer(&cbd, &cs0, &cb);
    D3D11_BUFFER_DESC stg{}; stg.ByteWidth = N * 4; stg.Usage = D3D11_USAGE_STAGING; stg.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Buffer* bStg = nullptr; d.dev->CreateBuffer(&stg, nullptr, &bStg);

    // 4. dispatch
    d.ctx->CSSetShader(cs, nullptr, 0);
    d.ctx->CSSetShaderResources(0, 1, &sIn);
    d.ctx->CSSetUnorderedAccessViews(0, 1, &uOut, nullptr);
    d.ctx->CSSetConstantBuffers(0, 1, &cb);
    d.ctx->Dispatch((N + 63) / 64, 1, 1);
    d.ctx->CopyResource(bStg, bOut);
    D3D11_MAPPED_SUBRESOURCE ms; d.ctx->Map(bStg, 0, D3D11_MAP_READ, 0, &ms);
    memcpy(Out.data(), ms.pData, N * 4); d.ctx->Unmap(bStg, 0);

    // 5. verify vs CPU reference of the KLSL semantics
    double maxAbs = 0.0;
    for (UINT i = 0; i < N; ++i) maxAbs = std::max(maxAbs, std::fabs((double)Out[i] - (double)Ref[i]));
    printf("[run]      dispatched %u elements on d3d11_cs_5_0\n", N);
    printf("[verify]   Out[0..3] = %.4f %.4f %.4f %.4f\n", Out[0], Out[1], Out[2], Out[3]);
    printf("[verify]   Ref[0..3] = %.4f %.4f %.4f %.4f\n", Ref[0], Ref[1], Ref[2], Ref[3]);
    printf("[verify]   Out[255]=%.4f Ref[255]=%.4f  max abs err = %.2e\n\n", Out[255], Ref[255], maxAbs);

    cs->Release(); bIn->Release(); bOut->Release(); sIn->Release(); uOut->Release(); cb->Release(); bStg->Release();

    bool pass = maxAbs < 1e-5;
    printf("=== %s: .klsl -> (real compiler) -> HLSL -> cs_5_0 dispatch -> %s vs CPU ===\n",
           pass ? "PASS" : "FAIL", pass ? "correct" : "MISMATCH");
    return pass ? 0 : 1;
}
