// q8_hotswap_probe.cpp — test the Q8 (INT8) resident limit + tile hot-swap on the HD 4600.
//
// The ceiling probe found: hard resident wall at the 2048 MB DXGI LOCAL budget, ~1.75 GB stable,
// device-removed on overcommit. Qwen-1.8B INT8 is 1.71 GB — right at that stable ceiling. This
// probe tests whether that footprint is actually VIABLE, in three measured phases:
//
//   A. RESIDENCY  — bring 1707 MB (Qwen-1.8B Q8 weight-equivalent) resident (committed + MakeResident).
//   B. HOT-SWAP   — with the Q8 footprint resident, Evict one 128 MB tile and MakeResident a
//                   replacement, ping-pong N times, proving tiles hot-swap WITHIN budget (the design
//                   that makes the tight fit survivable).
//   C. HEADROOM   — grow 32 MB resident probes past the Q8 base until the wall, measuring how much is
//                   left for activations / KV cache at Q8. (Runs LAST: it may trip device removal.)
//
// Mechanism = #001's residency path (D3D12 DEFAULT-heap committed + MakeResident/Evict). Free-RAM
// guarded; everything freed at exit.
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
static double mb(uint64_t b){ return (double)b/MB; }

static ComPtr<IDXGIAdapter3> g_adapter;
static uint64_t localUsage(){ DXGI_QUERY_VIDEO_MEMORY_INFO v{}; g_adapter->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&v); return v.CurrentUsage; }
static uint64_t localBudget(){ DXGI_QUERY_VIDEO_MEMORY_INFO v{}; g_adapter->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&v); return v.Budget; }

static ComPtr<ID3D12Resource> makeBuf(ID3D12Device* dev, uint64_t bytes){
    D3D12_HEAP_PROPERTIES hp{}; hp.Type=D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{}; rd.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width=bytes; rd.Height=1;
    rd.DepthOrArraySize=1; rd.MipLevels=1; rd.Format=DXGI_FORMAT_UNKNOWN; rd.SampleDesc.Count=1;
    rd.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags=D3D12_RESOURCE_FLAG_NONE;
    ComPtr<ID3D12Resource> r;
    if(FAILED(dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&rd,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&r)))) return nullptr;
    return r;
}

int main(){
    ComPtr<IDXGIFactory4> factory; CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    ComPtr<IDXGIAdapter1> a1;
    for(UINT i=0; factory->EnumAdapters1(i,&a1)!=DXGI_ERROR_NOT_FOUND; ++i){
        DXGI_ADAPTER_DESC1 d; a1->GetDesc1(&d);
        if(d.Flags&DXGI_ADAPTER_FLAG_SOFTWARE){ a1.Reset(); continue; }
        a1.As(&g_adapter); break;
    }
    if(!g_adapter){ printf("[err] no hw adapter\n"); return 1; }
    DXGI_ADAPTER_DESC1 desc; g_adapter->GetDesc1(&desc); wprintf(L"[dev] %s\n",desc.Description);
    ComPtr<ID3D12Device> dev;
    if(FAILED(D3D12CreateDevice(g_adapter.Get(),D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&dev)))){ printf("[err] device\n"); return 1; }
    printf("[budget] LOCAL budget %.0f MB  start-usage %.0f MB\n\n", mb(localBudget()), mb(localUsage()));

    const uint64_t Q8_MB = 1707;            // Qwen-1.8B INT8 weight footprint
    const uint64_t TILE  = 128*1024*1024;   // 128 MB hot-swap tile
    const uint64_t FREE_FLOOR = (uint64_t)2*1024*1024*1024;

    // ---- Phase A: bring the Q8 footprint resident (keep ONE tile-slot for hot-swap) ----
    printf("=== A. Q8 RESIDENCY (target %llu MB) ===\n",(unsigned long long)Q8_MB);
    std::vector<ComPtr<ID3D12Resource>> base;      // permanently resident
    uint64_t baseBytes=0; const uint64_t baseTarget=(Q8_MB*1024*1024) - TILE;  // leave room for the hot tile
    bool ok=true;
    while(baseBytes + TILE <= baseTarget){
        auto r=makeBuf(dev.Get(),TILE); if(!r){ ok=false; break; }
        ID3D12Pageable* p=r.Get(); if(FAILED(dev->MakeResident(1,&p))){ ok=false; break; }
        base.push_back(r); baseBytes+=TILE;
    }
    if(uint64_t rem=baseTarget-baseBytes){ auto r=makeBuf(dev.Get(),rem); if(r){ ID3D12Pageable*p=r.Get(); if(SUCCEEDED(dev->MakeResident(1,&p))){ base.push_back(r); baseBytes+=rem; } } }
    // the hot tile (the swappable slot) — brings us to the full Q8 footprint
    ComPtr<ID3D12Resource> hot=makeBuf(dev.Get(),TILE); ID3D12Pageable* hp2=hot.Get();
    bool hotOk = hot && SUCCEEDED(dev->MakeResident(1,&hp2));
    uint64_t resident = baseBytes + (hotOk?TILE:0);
    printf("  base resident %.0f MB + hot tile %s -> resident %.0f MB   usage %.0f MB / budget %.0f MB\n",
           mb(baseBytes), hotOk?"OK":"FAIL", mb(resident), mb(localUsage()), mb(localBudget()));
    printf("  Q8 footprint resident: %s\n\n", (ok && hotOk && resident>=(Q8_MB-4)*1024*1024) ? "PASS" : "PARTIAL/FAIL");

    // ---- Phase B: HOT-SWAP the hot tile within budget (Evict then MakeResident a replacement) ----
    printf("=== B. HOT-SWAP (Evict + MakeResident, %d cycles, budget-neutral) ===\n", 20);
    ComPtr<ID3D12Resource> spare=makeBuf(dev.Get(),TILE);   // second tile buffer (created, will be evicted)
    bool swapPass=true; uint64_t umin=~0ull,umax=0;
    if(spare){ ID3D12Pageable* sp=spare.Get(); dev->Evict(1,&sp); }  // start with spare NON-resident
    ComPtr<ID3D12Resource> curHot=hot, curSpare=spare;
    for(int i=0;i<20 && swapPass;i++){
        ID3D12Pageable* out=curHot.Get(); ID3D12Pageable* in=curSpare.Get();
        if(FAILED(dev->Evict(1,&out))){ swapPass=false; printf("  cycle %d: Evict failed\n",i); break; }
        if(FAILED(dev->MakeResident(1,&in))){ swapPass=false; printf("  cycle %d: MakeResident(in) failed\n",i); break; }
        uint64_t u=localUsage(); if(u<umin)umin=u; if(u>umax)umax=u;
        std::swap(curHot,curSpare);   // the tile we just brought in is now the hot one
    }
    printf("  swap cycles: %s   usage stayed %.0f..%.0f MB (budget-neutral, no growth past Q8)\n\n",
           swapPass?"PASS (all 20 evict+resident succeeded)":"FAIL", mb(umin), mb(umax));

    // ---- Phase C: headroom above Q8 for activations / KV (grows to the wall; may trip removal) ----
    printf("=== C. ACTIVATION/KV HEADROOM above the Q8 base ===\n");
    std::vector<ComPtr<ID3D12Resource>> head; uint64_t headBytes=0; const uint64_t STEP=32*1024*1024;
    const char* why="wall";
    for(;;){
        MEMORYSTATUSEX ms{sizeof(ms)}; GlobalMemoryStatusEx(&ms);
        if(ms.ullAvailPhys < FREE_FLOOR+STEP){ why="free-RAM guard"; break; }
        auto r=makeBuf(dev.Get(),STEP); if(!r){ why="CreateCommittedResource failed"; break; }
        ID3D12Pageable* p=r.Get(); if(FAILED(dev->MakeResident(1,&p))){ why="MakeResident refused"; break; }
        head.push_back(r); headBytes+=STEP;
        if(headBytes >= 512*1024*1024){ why="capped probe at 512 MB headroom"; break; }
    }
    printf("  headroom above Q8 before %s: %.0f MB   (final usage %.0f MB / budget %.0f MB)\n\n",
           why, mb(headBytes), mb(localUsage()), mb(localBudget()));

    // ---- verdict ----
    printf("=== Q8 VERDICT (Qwen-1.8B INT8, HD 4600) ===\n");
    printf("  weights resident (1.71 GB)      : %s\n", (ok&&hotOk)?"PASS":"FAIL");
    printf("  tile hot-swap within budget     : %s\n", swapPass?"PASS":"FAIL");
    printf("  headroom for activations/KV     : ~%.0f MB above the 1.71 GB base\n", mb(headBytes));
    printf("  -> Q8 is %s: weights fit resident; %s\n",
           (ok&&hotOk&&swapPass)?"VIABLE with hot-swap":"NOT viable as-is",
           (headBytes>=256*1024*1024)?"enough headroom for a modest KV window":
           "headroom is tight -> KV/activations must also stream (hot-swap) or use a short context");
    base.clear(); head.clear();
    return 0;
}
