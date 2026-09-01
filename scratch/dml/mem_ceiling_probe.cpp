// mem_ceiling_probe.cpp — measure the GPU-resident memory ceiling on the HD 4600.
//
// KGRC proof-ladder companion to #001 (which put ~500 MB of gpt2 weights resident). Answers the
// scaling question empirically: how many bytes can actually stay RESIDENT on this UMA iGPU before
// (a) the WDDM manager reports us over-budget (soft residency ceiling), and (b) allocation/residency
// hard-fails (hard ceiling). Both numbers matter for "how big a model fits":
//   soft budget  = stay resident with NO eviction/paging  (the honest "model fits on GPU" number)
//   hard ceiling = last byte D3D12 will commit + MakeResident before failure
//
// Mechanism = exactly #001's residency path: D3D12 DEFAULT-heap COMMITTED buffers + explicit
// ID3D12Device::MakeResident (DirectML residency is D3D12 residency). Chunked at 256 MB, with a
// GlobalMemoryStatusEx free-RAM guard so it cannot exhaust the 16 GB box. Everything is freed at end.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <dxgi1_4.h>
#include <d3d12.h>
#include <wrl/client.h>
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d12.lib")
using Microsoft::WRL::ComPtr;

static const double MB = 1024.0 * 1024.0;
static double mb(uint64_t b) { return (double)b / MB; }

int main() {
    // ---- 1. adapter (the real Intel iGPU, not WARP) + its reported memory ----
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) { printf("[err] CreateDXGIFactory1\n"); return 1; }
    ComPtr<IDXGIAdapter1> a1; ComPtr<IDXGIAdapter3> adapter;
    for (UINT i = 0; factory->EnumAdapters1(i, &a1) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d; a1->GetDesc1(&d);
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { a1.Reset(); continue; }  // skip WARP
        a1.As(&adapter); break;
    }
    if (!adapter) { printf("[err] no hardware adapter\n"); return 1; }
    DXGI_ADAPTER_DESC1 desc; adapter->GetDesc1(&desc);
    wprintf(L"[dev] %s\n", desc.Description);
    printf("[desc] DedicatedVideoMemory %.0f MB   DedicatedSystemMemory %.0f MB   SharedSystemMemory %.0f MB\n",
           mb(desc.DedicatedVideoMemory), mb(desc.DedicatedSystemMemory), mb(desc.SharedSystemMemory));

    DXGI_QUERY_VIDEO_MEMORY_INFO vmiL{}, vmiN{};
    adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &vmiL);
    adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &vmiN);
    printf("[budget] LOCAL     budget %.0f MB  current-usage %.0f MB  avail-for-reservation %.0f MB\n",
           mb(vmiL.Budget), mb(vmiL.CurrentUsage), mb(vmiL.AvailableForReservation));
    printf("[budget] NON-LOCAL budget %.0f MB  current-usage %.0f MB\n", mb(vmiN.Budget), mb(vmiN.CurrentUsage));
    const uint64_t startUsage = vmiL.CurrentUsage;

    // ---- 2. D3D12 device (FL 11_0 — the HD 4600 ceiling; 12_0+ is unsupported here) ----
    ComPtr<ID3D12Device> dev;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)))) {
        printf("[err] D3D12CreateDevice FL11_0 failed\n"); return 1;
    }

    // ---- 3. grow resident DEFAULT-heap buffers until over-budget, then until hard failure ----
    const uint64_t CHUNK = (uint64_t)256 * 1024 * 1024;   // 256 MB per committed buffer
    const uint64_t FREE_RAM_FLOOR = (uint64_t)2 * 1024 * 1024 * 1024;  // never drive free RAM below 2 GB
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = CHUNK; rd.Height = 1;
    rd.DepthOrArraySize = 1; rd.MipLevels = 1; rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags = D3D12_RESOURCE_FLAG_NONE;

    std::vector<ComPtr<ID3D12Resource>> live;
    uint64_t committed = 0, soft_ceiling = 0; bool crossed = false;
    const char* stop = "hard failure";
    printf("\n  step  cumulative(MB)   LOCAL.usage(MB)   LOCAL.budget(MB)   freePhys(MB)   status\n");
    for (int step = 1; ; ++step) {
        MEMORYSTATUSEX ms{ sizeof(ms) }; GlobalMemoryStatusEx(&ms);
        if (ms.ullAvailPhys < FREE_RAM_FLOOR + CHUNK) { stop = "safety guard (free RAM floor)"; break; }

        ComPtr<ID3D12Resource> res;
        HRESULT hr = dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                  D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&res));
        if (FAILED(hr)) { printf("  %4d  CreateCommittedResource failed hr=0x%08lx\n", step, (unsigned long)hr); break; }
        ID3D12Pageable* p = res.Get();
        hr = dev->MakeResident(1, &p);
        if (FAILED(hr)) { printf("  %4d  MakeResident failed hr=0x%08lx\n", step, (unsigned long)hr); stop = "MakeResident refused"; break; }

        live.push_back(res);
        committed += CHUNK;
        adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &vmiL);
        GlobalMemoryStatusEx(&ms);
        bool over = vmiL.CurrentUsage > vmiL.Budget;
        if (over && !crossed) { crossed = true; soft_ceiling = committed - CHUNK; }
        printf("  %4d   %11.0f     %12.0f      %12.0f    %11.0f   %s\n",
               step, mb(committed), mb(vmiL.CurrentUsage), mb(vmiL.Budget), mb(ms.ullAvailPhys),
               over ? "OVER-BUDGET (would page)" : "resident");
        if (committed >= (uint64_t)12 * 1024 * 1024 * 1024) { stop = "reached 12 GB probe cap"; break; }
    }

    // ---- 4. verdict ----
    printf("\n=== MEMORY CEILING (HD 4600, D3D12 DEFAULT-heap resident) ===\n");
    printf("reported LOCAL budget (soft, stay-resident) : %.0f MB\n", mb(vmiL.Budget));
    if (crossed) printf("empirical over-budget crossing              : between %.0f and %.0f MB committed\n",
                        mb(soft_ceiling), mb(soft_ceiling + CHUNK));
    else         printf("empirical over-budget crossing              : NOT reached (%s)\n", stop);
    printf("hard resident ceiling (committed+MakeResident): %.0f MB  [stop: %s]\n", mb(committed), stop);
    printf("app-attributable resident growth             : %.0f MB (usage delta over start)\n",
           mb(vmiL.CurrentUsage > startUsage ? vmiL.CurrentUsage - startUsage : 0));
    printf("\ninterpretation for resident models (FP16 = 2 B/param):\n");
    printf("  soft budget %.2f GB  -> fits: gpt2(0.25GB FP16), a ~%.1fB-param FP16 model, or a ~%.1fB INT8 model\n",
           mb(vmiL.Budget)/1024.0, (mb(vmiL.Budget)/1024.0)/2.0, (mb(vmiL.Budget)/1024.0));
    printf("  Qwen-1.8B FP16 ~3.6 GB : %s the soft budget -> %s\n",
           (3.6*1024 <= mb(vmiL.Budget)) ? "<=" : ">",
           (3.6*1024 <= mb(vmiL.Budget)) ? "can stay fully resident" : "must quantize (INT8~1.8GB / Q4~0.9GB) or stream");

    live.clear();  // free everything
    return 0;
}
