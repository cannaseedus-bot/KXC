// cssc_gate_run.cpp — proves the CSS-C capability gate doesn't just RESOLVE, it RUNS.
//
// A trivial CSS-C manifest (requires: sm5 | cpu) is resolved against the MEASURED device,
// then the winning tier's runtime actually executes the kernel and the result is verified
// against a CPU reference. On this rig the gate must pick `sm5` (D3D11 cs_5_0) and dispatch.
//
// Mirrors the JS resolver (www/js/gpu/cssc_gate.js) + contract (cssc_capability_gate.json):
//   probe -> available_runtimes ; first requires-tier whose runtime is available -> RUN ;
//   else cpu floor if requested ; else DO_NOT_RUN. Nothing operates without a runtime.
//
// Build (from an x64 VS BuildTools shell):
//   vcvars64 && cl /std:c++17 /EHsc /O2 cssc_gate_run.cpp
// Self-contained: links d3d11 / d3dcompiler / dxguid via #pragma comment. No external headers.

#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")

// ---- a trivial CSS-C manifest: out[i] = in[i]*scale + bias, dispatched (N/64,1,1) ----
// This IS the "computation style sheet": a kernel + dispatch + inputs, gated by `requires`.
struct CsscManifest {
    const char* name;
    const char* compute_shader;         // KLSL would lower to this cs_5_0 HLSL (SHADER_PIPELINE.md)
    std::vector<std::string> requires_; // ordered ceiling->floor tier preference
    UINT dispatchN;                     // dispatch: (ceil(N/64),1,1)
    float scale, bias;                  // inputs: --scale, --bias
};

static const char* PI_KERNEL_HLSL = R"(
cbuffer P : register(b0) { float scale; float bias; uint n; uint pad; };
StructuredBuffer<float>   In  : register(t0);
RWStructuredBuffer<float> Out : register(u0);
[numthreads(64,1,1)]
void CSMain(uint3 t : SV_DispatchThreadID) {
    uint i = t.x; if (i >= n) return;
    Out[i] = In[i] * scale + bias;   // per-element compute style sheet
})";

// ---- the gate: resolve manifest.requires against MEASURED available runtimes ----
struct GateDecision { std::string decision, tier, runtime; };
static GateDecision resolveGate(const CsscManifest& m, const std::vector<std::string>& availRuntimes) {
    // tier -> runtime map (subset of cssc_capability_gate.json relevant to native)
    auto runtimeOf = [](const std::string& tier) -> std::string {
        if (tier == "webgpu")    return "dawn_webgpu";
        if (tier == "sm610")     return "d3d12_coopmat";
        if (tier == "d3d12_11x") return "d3d12_fl11x";
        if (tier == "sm5")       return "d3d11_cs_5_0";
        if (tier == "warp")      return "warp_d3d12";
        if (tier == "opencl")    return "opencl_1_2";
        if (tier == "webgl2")    return "angle_d3d11";
        if (tier == "cpu")       return "cpu_wasm";
        return "";
    };
    auto has = [&](const std::string& rt) {
        for (auto& a : availRuntimes) if (a == rt) return true; return false;
    };
    for (auto& tier : m.requires_) {
        std::string rt = runtimeOf(tier);
        if (!rt.empty() && has(rt)) return { "RUN", tier, rt };
    }
    for (auto& tier : m.requires_) if (tier == "cpu" && has("cpu_wasm")) return { "RUN", "cpu", "cpu_wasm" };
    return { "DO_NOT_RUN", "", "" };
}

// ---- D3D11 sm5 runtime ----
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

// dispatch the manifest's kernel on the D3D11 cs_5_0 runtime; fills Out. returns true on success.
static bool runSm5(D3D& d, const CsscManifest& m, const std::vector<float>& In, std::vector<float>& Out) {
    ID3DBlob* blob = nullptr; ID3DBlob* err = nullptr;
    HRESULT hr = D3DCompile(m.compute_shader, strlen(m.compute_shader), "cssc", nullptr, nullptr,
        "CSMain", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &err);
    if (FAILED(hr)) { if (err) printf("HLSL err: %s\n", (char*)err->GetBufferPointer()); return false; }
    ID3D11ComputeShader* cs = nullptr;
    d.dev->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &cs);
    blob->Release(); if (err) err->Release();

    UINT n = m.dispatchN;
    ID3D11Buffer* bIn = structBuf(d.dev, In.data(), n * 4, false);
    ID3D11Buffer* bOut = structBuf(d.dev, nullptr, n * 4, true);
    ID3D11ShaderResourceView* sIn = nullptr;
    D3D11_SHADER_RESOURCE_VIEW_DESC sv{}; sv.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
    sv.Format = DXGI_FORMAT_UNKNOWN; sv.BufferEx.NumElements = n; d.dev->CreateShaderResourceView(bIn, &sv, &sIn);
    ID3D11UnorderedAccessView* uOut = nullptr;
    D3D11_UNORDERED_ACCESS_VIEW_DESC ud{}; ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    ud.Format = DXGI_FORMAT_UNKNOWN; ud.Buffer.NumElements = n; d.dev->CreateUnorderedAccessView(bOut, &ud, &uOut);
    struct P { float scale, bias; UINT n, pad; } p{ m.scale, m.bias, n, 0 };
    D3D11_BUFFER_DESC cbd{}; cbd.ByteWidth = sizeof(P); cbd.Usage = D3D11_USAGE_DEFAULT; cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA cs0{}; cs0.pSysMem = &p; ID3D11Buffer* cb = nullptr; d.dev->CreateBuffer(&cbd, &cs0, &cb);
    D3D11_BUFFER_DESC stg{}; stg.ByteWidth = n * 4; stg.Usage = D3D11_USAGE_STAGING; stg.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Buffer* bStg = nullptr; d.dev->CreateBuffer(&stg, nullptr, &bStg);

    d.ctx->CSSetShader(cs, nullptr, 0);
    d.ctx->CSSetShaderResources(0, 1, &sIn);
    d.ctx->CSSetUnorderedAccessViews(0, 1, &uOut, nullptr);
    d.ctx->CSSetConstantBuffers(0, 1, &cb);
    d.ctx->Dispatch((n + 63) / 64, 1, 1);

    d.ctx->CopyResource(bStg, bOut);
    D3D11_MAPPED_SUBRESOURCE ms; d.ctx->Map(bStg, 0, D3D11_MAP_READ, 0, &ms);
    memcpy(Out.data(), ms.pData, n * 4); d.ctx->Unmap(bStg, 0);

    cs->Release(); bIn->Release(); bOut->Release(); sIn->Release(); uOut->Release(); cb->Release(); bStg->Release();
    return true;
}

int main() {
    printf("=== CSS-C gate: prove it RUNS (not just resolves) ===\n\n");

    // 1. MEASURE the device (never assume). D3D11 device creation = sm5 runtime available.
    D3D d; bool d3d = d.init();
    std::vector<std::string> avail = { "cpu_wasm" };            // cpu floor always present
    if (d3d) avail.push_back("d3d11_cs_5_0");                   // measured: sm5 present
    printf("[measure] D3D11 device: %s", d3d ? "UP" : "FAILED");
    if (d3d) printf(" (feature level 0x%x)", (int)d.fl);
    printf("\n[measure] available_runtimes = ");
    for (auto& a : avail) printf("%s ", a.c_str());
    printf("\n\n");

    // 2. the CSS-C manifest
    CsscManifest m{ "pi_scale", PI_KERNEL_HLSL, { "webgpu", "sm5", "cpu" }, 256, 2.5f, 1.0f };
    printf("[manifest] %s: requires=[webgpu, sm5, cpu] dispatch=(%u/64,1,1) scale=%.2f bias=%.2f\n",
           m.name, m.dispatchN, m.scale, m.bias);

    // 3. resolve through the gate
    GateDecision g = resolveGate(m, avail);
    printf("[gate]     %s @ %s (%s)\n\n", g.decision.c_str(), g.tier.c_str(), g.runtime.c_str());
    if (g.decision != "RUN") { printf("gate did not resolve to a runnable tier\n"); return 1; }

    // 4. RUN on the selected runtime + verify vs CPU reference
    std::vector<float> In(m.dispatchN), Out(m.dispatchN), Ref(m.dispatchN);
    for (UINT i = 0; i < m.dispatchN; ++i) In[i] = i * 0.1f;
    for (UINT i = 0; i < m.dispatchN; ++i) Ref[i] = In[i] * m.scale + m.bias;   // CPU reference

    bool ok = false;
    if (g.tier == "sm5") ok = runSm5(d, m, In, Out);
    if (!ok) { printf("runtime dispatch failed\n"); return 1; }

    double maxAbs = 0.0;
    for (UINT i = 0; i < m.dispatchN; ++i) maxAbs = std::max(maxAbs, std::fabs((double)Out[i] - (double)Ref[i]));
    printf("[run]      dispatched %u elements on %s\n", m.dispatchN, g.runtime.c_str());
    printf("[verify]   Out[0..3]  = %.4f %.4f %.4f %.4f\n", Out[0], Out[1], Out[2], Out[3]);
    printf("[verify]   Ref[0..3]  = %.4f %.4f %.4f %.4f\n", Ref[0], Ref[1], Ref[2], Ref[3]);
    printf("[verify]   Out[255]   = %.4f   Ref[255] = %.4f\n", Out[255], Ref[255]);
    printf("[verify]   max abs err = %.2e\n\n", maxAbs);

    bool pass = maxAbs < 1e-5;
    printf("=== %s: CSS-C manifest resolved -> %s -> dispatched cs_5_0 -> %s vs CPU ===\n",
           pass ? "PASS" : "FAIL", g.runtime.c_str(), pass ? "correct" : "MISMATCH");
    return pass ? 0 : 1;
}
