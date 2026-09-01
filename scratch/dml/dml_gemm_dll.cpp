// dml_gemm_dll.cpp — DirectML GEMM as a C-ABI DLL, shared by the KHANARY inference driver
// (ctypes) and the ggml-xcfe backend (LoadLibrary). Amortized: persistent D3D12+DML device,
// per-shape resource cache (persistently-mapped upload/readback, bind-once binding table),
// one GPU sync per call.
//
// Exports:
//   int dml_gemm_f32   (const float* A, const float* B, float* C, uint M,uint N,uint K)
//       C[M,N] = A[M,K] @ B[K,N]                     (B row-major [K,N])
//   int dml_gemm_bt_f32(const float* A, const float* B, float* C, uint M,uint N,uint K)
//       C[M,N] = A[M,K] @ B^T                        (B row-major [N,K]) — ggml MUL_MAT shape
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <DirectML.h>
#include <map>
#include <tuple>
#include <cstdint>
#include <cstring>
#include <algorithm>
#pragma comment(lib,"d3d12.lib")
#pragma comment(lib,"dxgi.lib")
#pragma comment(lib,"dxguid.lib")
#pragma comment(lib,"DirectML.lib")

#define OK(x) do{ if(FAILED(x)) return -1; }while(0)

static D3D12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE t){ D3D12_HEAP_PROPERTIES h{}; h.Type=t; h.CPUPageProperty=D3D12_CPU_PAGE_PROPERTY_UNKNOWN; h.MemoryPoolPreference=D3D12_MEMORY_POOL_UNKNOWN; h.CreationNodeMask=1; h.VisibleNodeMask=1; return h; }
static D3D12_RESOURCE_DESC bufDesc(UINT64 b,D3D12_RESOURCE_FLAGS f=D3D12_RESOURCE_FLAG_NONE){ D3D12_RESOURCE_DESC d{}; d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; d.Width=b; d.Height=1; d.DepthOrArraySize=1; d.MipLevels=1; d.Format=DXGI_FORMAT_UNKNOWN; d.SampleDesc.Count=1; d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags=f; return d; }
static D3D12_RESOURCE_BARRIER bar(ID3D12Resource*r,D3D12_RESOURCE_STATES a,D3D12_RESOURCE_STATES b){ D3D12_RESOURCE_BARRIER br{}; br.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; br.Transition.pResource=r; br.Transition.StateBefore=a; br.Transition.StateAfter=b; br.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; return br; }

struct Ctx {
    ID3D12Device* dev=nullptr; ID3D12CommandQueue* q=nullptr; ID3D12CommandAllocator* alloc=nullptr;
    ID3D12GraphicsCommandList* cl=nullptr; ID3D12Fence* fence=nullptr; UINT64 fv=0; HANDLE fe=nullptr;
    IDMLDevice* dml=nullptr; IDMLCommandRecorder* rec=nullptr; bool init=false;

    struct Shape {
        IDMLCompiledOperator* op=nullptr; ID3D12DescriptorHeap* heap=nullptr; IDMLBindingTable* bt=nullptr; UINT descN=0;
        ID3D12Resource *aBuf=nullptr,*oBuf=nullptr,*upA=nullptr,*rb=nullptr,*tmp=nullptr;
        void *upAp=nullptr;
        UINT64 aB=0,bB=0,oB=0,tmpB=0;
    };
    std::map<std::tuple<UINT,UINT,UINT,int>,Shape> cache;
    // GPU-resident weights, keyed by host pointer: each weight uploaded once, reused every call.
    struct Weight { ID3D12Resource* buf=nullptr; UINT64 bytes=0; };
    std::map<const float*,Weight> weights;

    void flush(){ cl->Close(); ID3D12CommandList*ls[]={cl}; q->ExecuteCommandLists(1,ls); q->Signal(fence,++fv); fence->SetEventOnCompletion(fv,fe); WaitForSingleObject(fe,INFINITE); alloc->Reset(); cl->Reset(alloc,nullptr); }
};
static Ctx g;

static int ensureInit(){
    if(g.init) return 0;
    IDXGIFactory4* fac=nullptr; OK(CreateDXGIFactory1(IID_PPV_ARGS(&fac)));
    IDXGIAdapter1* ad=nullptr; DXGI_ADAPTER_DESC1 dd{};
    for(UINT i=0; fac->EnumAdapters1(i,&ad)!=DXGI_ERROR_NOT_FOUND; ++i){
        ad->GetDesc1(&dd); if(dd.Flags & DXGI_ADAPTER_FLAG_SOFTWARE){ ad->Release(); ad=nullptr; continue; }
        if(SUCCEEDED(D3D12CreateDevice(ad,D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&g.dev)))) break;
        ad->Release(); ad=nullptr;
    }
    if(!g.dev) return -1;
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
    OK(g.dev->CreateCommandQueue(&qd,IID_PPV_ARGS(&g.q)));
    OK(g.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&g.alloc)));
    OK(g.dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,g.alloc,nullptr,IID_PPV_ARGS(&g.cl)));
    OK(g.dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&g.fence)));
    g.fe=CreateEvent(nullptr,FALSE,FALSE,nullptr);
    OK(DMLCreateDevice(g.dev,DML_CREATE_DEVICE_FLAG_NONE,IID_PPV_ARGS(&g.dml)));
    OK(g.dml->CreateCommandRecorder(IID_PPV_ARGS(&g.rec)));
    g.init=true; return 0;
}

static ID3D12Resource* mkDefault(UINT64 b){ ID3D12Resource*r=nullptr; auto hp=heapProps(D3D12_HEAP_TYPE_DEFAULT); auto rd=bufDesc(b,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS); g.dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&rd,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&r)); return r; }
static ID3D12Resource* mkHeap(UINT64 b,D3D12_HEAP_TYPE t,D3D12_RESOURCE_STATES s){ ID3D12Resource*r=nullptr; auto hp=heapProps(t); auto rd=bufDesc(b); g.dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&rd,s,nullptr,IID_PPV_ARGS(&r)); return r; }

// Build (or fetch) the cached compiled operator + resources + bound binding table for a shape.
static int getShape(UINT M,UINT N,UINT K,bool transB,Ctx::Shape*& out){
    auto key=std::make_tuple(M,N,K,transB?1:0);
    auto it=g.cache.find(key);
    if(it!=g.cache.end()){ out=&it->second; return 0; }
    Ctx::Shape s{};
    // tensors: A[M,K]; B is [K,N] (plain) or [N,K] + TransB (transpose); Out[M,N]
    UINT aSz[4]={1,1,M,K}, bSz[4]={1,1, transB?N:K, transB?K:N}, oSz[4]={1,1,M,N};
    auto mk=[&](UINT*z,DML_BUFFER_TENSOR_DESC&bt,DML_TENSOR_DESC&td){ bt={}; bt.DataType=DML_TENSOR_DATA_TYPE_FLOAT32; bt.DimensionCount=4; bt.Sizes=z; bt.TotalTensorSizeInBytes=(UINT64)z[0]*z[1]*z[2]*z[3]*sizeof(float); td.Type=DML_TENSOR_TYPE_BUFFER; td.Desc=&bt; };
    DML_BUFFER_TENSOR_DESC aBT,bBT,oBT; DML_TENSOR_DESC aTD,bTD,oTD; mk(aSz,aBT,aTD); mk(bSz,bBT,bTD); mk(oSz,oBT,oTD);
    DML_GEMM_OPERATOR_DESC gemm{}; gemm.ATensor=&aTD; gemm.BTensor=&bTD; gemm.CTensor=nullptr; gemm.OutputTensor=&oTD;
    gemm.TransA=DML_MATRIX_TRANSFORM_NONE; gemm.TransB=transB?DML_MATRIX_TRANSFORM_TRANSPOSE:DML_MATRIX_TRANSFORM_NONE; gemm.Alpha=1.0f; gemm.Beta=0.0f;
    DML_OPERATOR_DESC opd{}; opd.Type=DML_OPERATOR_GEMM; opd.Desc=&gemm;
    IDMLOperator* op=nullptr; OK(g.dml->CreateOperator(&opd,IID_PPV_ARGS(&op)));
    OK(g.dml->CompileOperator(op,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&s.op))); op->Release();

    IDMLOperatorInitializer* ini=nullptr; IDMLCompiledOperator* cops[]={s.op};
    OK(g.dml->CreateOperatorInitializer(1,cops,IID_PPV_ARGS(&ini)));
    DML_BINDING_PROPERTIES ip=ini->GetBindingProperties(), ep=s.op->GetBindingProperties();
    s.descN=std::max(ip.RequiredDescriptorCount,ep.RequiredDescriptorCount);
    s.tmpB=std::max(ip.TemporaryResourceSize,ep.TemporaryResourceSize);
    D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors=s.descN; hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    OK(g.dev->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&s.heap)));

    s.aB=(UINT64)M*K*4; s.bB=(UINT64)N*K*4; s.oB=(UINT64)M*N*4;
    s.aBuf=mkDefault(s.aB); s.oBuf=mkDefault(s.oB);
    s.upA=mkHeap(s.aB,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
    s.rb =mkHeap(s.oB,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
    if(s.tmpB) s.tmp=mkDefault(s.tmpB);
    D3D12_RANGE nr{0,0}; s.upA->Map(0,&nr,&s.upAp);  // upload heap stays mapped (write-combined)

    // init (bind temp; GEMM has no persistent resource)
    ID3D12DescriptorHeap* heaps[]={s.heap}; g.cl->SetDescriptorHeaps(1,heaps);
    DML_BINDING_TABLE_DESC bd{}; bd.Dispatchable=ini; bd.CPUDescriptorHandle=s.heap->GetCPUDescriptorHandleForHeapStart(); bd.GPUDescriptorHandle=s.heap->GetGPUDescriptorHandleForHeapStart(); bd.SizeInDescriptors=s.descN;
    OK(g.dml->CreateBindingTable(&bd,IID_PPV_ARGS(&s.bt)));
    if(s.tmpB && ip.TemporaryResourceSize){ DML_BUFFER_BINDING tb{s.tmp,0,s.tmpB}; DML_BINDING_DESC td{DML_BINDING_TYPE_BUFFER,&tb}; s.bt->BindTemporaryResource(&td); }
    g.rec->RecordDispatch(g.cl,ini,s.bt); g.flush(); ini->Release();
    // Note: B binding varies per call (weights are separate resident resources), so we rebind in run().

    auto res=g.cache.emplace(key,s); out=&res.first->second; return 0;
}

// Fetch (or upload-once) the GPU-resident buffer for a weight, keyed by its host pointer.
// Assumes the caller does not mutate the data at that pointer (true for model weights).
static int getWeight(const float* B,UINT64 bytes,ID3D12Resource*& out){
    auto it=g.weights.find(B);
    if(it!=g.weights.end()){ out=it->second.buf; return 0; }
    ID3D12Resource* buf=mkDefault(bytes);
    ID3D12Resource* up=mkHeap(bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
    void* p=nullptr; D3D12_RANGE nr{0,0}; up->Map(0,&nr,&p); memcpy(p,B,bytes); up->Unmap(0,nullptr);
    auto b1=bar(buf,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST); g.cl->ResourceBarrier(1,&b1);
    g.cl->CopyBufferRegion(buf,0,up,0,bytes); g.flush(); up->Release();
    g.weights[B]=Ctx::Weight{buf,bytes}; out=buf; return 0;
}

static int run(const float* A,const float* B,float* C,UINT M,UINT N,UINT K,bool transB){
    if(ensureInit()) return -2;
    Ctx::Shape* s=nullptr; if(getShape(M,N,K,transB,s)) return -3;
    ID3D12Resource* bBuf=nullptr; if(getWeight(B,s->bB,bBuf)) return -4;   // resident weight (uploaded once)

    // rebind for this call: A scratch + this weight + output (weights are per-call resources).
    DML_BINDING_TABLE_DESC bd{}; bd.Dispatchable=s->op; bd.CPUDescriptorHandle=s->heap->GetCPUDescriptorHandleForHeapStart(); bd.GPUDescriptorHandle=s->heap->GetGPUDescriptorHandleForHeapStart(); bd.SizeInDescriptors=s->descN;
    s->bt->Reset(&bd);
    if(s->tmpB){ DML_BUFFER_BINDING tb{s->tmp,0,s->tmpB}; DML_BINDING_DESC td{DML_BINDING_TYPE_BUFFER,&tb}; s->bt->BindTemporaryResource(&td); }
    DML_BUFFER_BINDING inBB[2]={ {s->aBuf,0,s->aB},{bBuf,0,s->bB} };
    DML_BINDING_DESC inBD[3]={ {DML_BINDING_TYPE_BUFFER,&inBB[0]},{DML_BINDING_TYPE_BUFFER,&inBB[1]},{DML_BINDING_TYPE_NONE,nullptr} };
    s->bt->BindInputs(3,inBD);
    DML_BUFFER_BINDING oBB{s->oBuf,0,s->oB}; DML_BINDING_DESC oBD{DML_BINDING_TYPE_BUFFER,&oBB}; s->bt->BindOutputs(1,&oBD);

    memcpy(s->upAp,A,s->aB);   // only the activations upload each call; weight is resident
    ID3D12DescriptorHeap* heaps[]={s->heap}; g.cl->SetDescriptorHeaps(1,heaps);
    // one fused command list: upload A -> dispatch -> copy out; single flush.
    D3D12_RESOURCE_BARRIER pre[3]={ bar(s->aBuf,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST),
                                    bar(bBuf,   D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                                    bar(s->oBuf,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS) };
    g.cl->ResourceBarrier(3,pre);
    g.cl->CopyBufferRegion(s->aBuf,0,s->upA,0,s->aB);
    auto toUav=bar(s->aBuf,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    g.cl->ResourceBarrier(1,&toUav);
    g.rec->RecordDispatch(g.cl,s->op,s->bt);
    auto ob=bar(s->oBuf,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE); g.cl->ResourceBarrier(1,&ob);
    g.cl->CopyResource(s->rb,s->oBuf);
    g.flush();
    void* p=nullptr; D3D12_RANGE rr{0,(SIZE_T)s->oB}; s->rb->Map(0,&rr,&p); memcpy(C,p,s->oB);
    D3D12_RANGE nw{0,0}; s->rb->Unmap(0,&nw);   // readback: map with read range each call to invalidate CPU cache
    return 0;
}

extern "C" __declspec(dllexport) int dml_gemm_f32   (const float* A,const float* B,float* C,unsigned M,unsigned N,unsigned K){ return run(A,B,C,M,N,K,false); }
extern "C" __declspec(dllexport) int dml_gemm_bt_f32(const float* A,const float* B,float* C,unsigned M,unsigned N,unsigned K){ return run(A,B,C,M,N,K,true ); }
