// dml_gemm_bench.cpp — benchmark DirectML's DML_OPERATOR_GEMM on the HD 4600 against the exact
// shape our tiled cs_5_0 kernel runs (C[M,N]=A[M,K]@B[K,N], real gpt2 QKV weight), reusing the
// same gemm_A/B/Cref.bin and 100-iter timing model as ../matmul_run.cpp so the numbers compare.
//
// DirectML runs on a D3D12 device. This 2015 Intel HD 4600 creates D3D12 only at FL 11_0/11_1
// (tier-1) and has NO vendor metacommands, so DML falls back to its own generic compute shaders —
// whether those beat our hand-tiled cs_5_0 GEMM is exactly what this measures.
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <DirectML.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#pragma comment(lib,"d3d12.lib")
#pragma comment(lib,"dxgi.lib")
#pragma comment(lib,"dxguid.lib")
#pragma comment(lib,"DirectML.lib")

#define HR(x) do{ HRESULT _hr=(x); if(FAILED(_hr)){ printf("FAIL %s = 0x%08x\n",#x,(unsigned)_hr); return 1; } }while(0)

static std::vector<float> readBin(const char*p,size_t n){std::vector<float> v(n);std::ifstream f(p,std::ios::binary);f.read((char*)v.data(),n*4);return v;}
// FLOAT32 contiguous buffer size (DMLCalcBufferTensorSize lives in DirectMLX; inline it for FP32).
static UINT64 calcSize(const UINT*s,UINT n){ UINT64 e=1; for(UINT i=0;i<n;i++) e*=s[i]; return e*sizeof(float); }

// minimal CD3DX12 equivalents
static D3D12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE t){ D3D12_HEAP_PROPERTIES h{}; h.Type=t; h.CPUPageProperty=D3D12_CPU_PAGE_PROPERTY_UNKNOWN; h.MemoryPoolPreference=D3D12_MEMORY_POOL_UNKNOWN; h.CreationNodeMask=1; h.VisibleNodeMask=1; return h; }
static D3D12_RESOURCE_DESC bufDesc(UINT64 bytes,D3D12_RESOURCE_FLAGS f=D3D12_RESOURCE_FLAG_NONE){ D3D12_RESOURCE_DESC d{}; d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; d.Alignment=0; d.Width=bytes; d.Height=1; d.DepthOrArraySize=1; d.MipLevels=1; d.Format=DXGI_FORMAT_UNKNOWN; d.SampleDesc.Count=1; d.SampleDesc.Quality=0; d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags=f; return d; }
static D3D12_RESOURCE_BARRIER barrier(ID3D12Resource*r,D3D12_RESOURCE_STATES a,D3D12_RESOURCE_STATES b){ D3D12_RESOURCE_BARRIER br{}; br.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; br.Transition.pResource=r; br.Transition.StateBefore=a; br.Transition.StateAfter=b; br.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; return br; }

int main(){
    UINT M=0,N=0,K=0;
    { std::ifstream f("../gemm_dims.txt"); std::stringstream d; d<<f.rdbuf(); d>>M>>N>>K; }
    if(!M||!N||!K){printf("no dims (run from scratch/dml)\n");return 1;}
    std::vector<float> A=readBin("../gemm_A.bin",(size_t)M*K), B=readBin("../gemm_B.bin",(size_t)K*N), Cref=readBin("../gemm_Cref.bin",(size_t)M*N);

    // --- D3D12 device on the hardware adapter (skip WARP/software) ---
    IDXGIFactory4* fac=nullptr; HR(CreateDXGIFactory1(IID_PPV_ARGS(&fac)));
    ID3D12Device* dev=nullptr; IDXGIAdapter1* ad=nullptr; DXGI_ADAPTER_DESC1 adDesc{};
    for(UINT i=0; fac->EnumAdapters1(i,&ad)!=DXGI_ERROR_NOT_FOUND; ++i){
        ad->GetDesc1(&adDesc);
        if(adDesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE){ ad->Release(); ad=nullptr; continue; }
        if(SUCCEEDED(D3D12CreateDevice(ad,D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&dev)))) break;
        ad->Release(); ad=nullptr;
    }
    if(!dev){ printf("no hardware D3D12 device at FL 11_0\n"); return 1; }
    printf("[dev] D3D12 hardware adapter: %ls  GEMM C[%u,%u]=A[%u,%u]@B[%u,%u]  (%.0fM MACs)\n",
           adDesc.Description,M,N,M,K,K,N,(double)M*N*K/1e6);

    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandQueue* q=nullptr; HR(dev->CreateCommandQueue(&qd,IID_PPV_ARGS(&q)));
    ID3D12CommandAllocator* alloc=nullptr; HR(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&alloc)));
    ID3D12GraphicsCommandList* cl=nullptr; HR(dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,alloc,nullptr,IID_PPV_ARGS(&cl)));
    ID3D12Fence* fence=nullptr; HR(dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
    UINT64 fv=0; HANDLE fe=CreateEvent(nullptr,FALSE,FALSE,nullptr);
    auto flush=[&](){ cl->Close(); ID3D12CommandList*ls[]={cl}; q->ExecuteCommandLists(1,ls); q->Signal(fence,++fv); fence->SetEventOnCompletion(fv,fe); WaitForSingleObject(fe,INFINITE); alloc->Reset(); cl->Reset(alloc,nullptr); };

    // --- DirectML device + GEMM operator (FP32, C=A@B, no bias) ---
    IDMLDevice* dml=nullptr;
    HRESULT dmlhr=DMLCreateDevice(dev,DML_CREATE_DEVICE_FLAG_NONE,IID_PPV_ARGS(&dml));
    if(FAILED(dmlhr)){ printf("DMLCreateDevice failed 0x%08x (this DirectML.dll may not support FL 11_1 hardware)\n",(unsigned)dmlhr); return 2; }

    UINT aSz[4]={1,1,M,K}, bSz[4]={1,1,K,N}, oSz[4]={1,1,M,N};
    auto tdesc=[&](UINT*s,DML_BUFFER_TENSOR_DESC&bt,DML_TENSOR_DESC&td){ bt={}; bt.DataType=DML_TENSOR_DATA_TYPE_FLOAT32; bt.Flags=DML_TENSOR_FLAG_NONE; bt.DimensionCount=4; bt.Sizes=s; bt.Strides=nullptr; bt.TotalTensorSizeInBytes=calcSize(s,4); td.Type=DML_TENSOR_TYPE_BUFFER; td.Desc=&bt; };
    DML_BUFFER_TENSOR_DESC aBT,bBT,oBT; DML_TENSOR_DESC aTD,bTD,oTD;
    tdesc(aSz,aBT,aTD); tdesc(bSz,bBT,bTD); tdesc(oSz,oBT,oTD);

    DML_GEMM_OPERATOR_DESC gemm{}; gemm.ATensor=&aTD; gemm.BTensor=&bTD; gemm.CTensor=nullptr; gemm.OutputTensor=&oTD;
    gemm.TransA=DML_MATRIX_TRANSFORM_NONE; gemm.TransB=DML_MATRIX_TRANSFORM_NONE; gemm.Alpha=1.0f; gemm.Beta=0.0f; gemm.FusedActivation=nullptr;
    DML_OPERATOR_DESC opd{}; opd.Type=DML_OPERATOR_GEMM; opd.Desc=&gemm;
    IDMLOperator* op=nullptr; HR(dml->CreateOperator(&opd,IID_PPV_ARGS(&op)));
    IDMLCompiledOperator* cop=nullptr;
    HRESULT chr=dml->CompileOperator(op,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&cop));
    if(FAILED(chr)){ printf("CompileOperator(GEMM) failed 0x%08x\n",(unsigned)chr); return 2; }

    IDMLOperatorInitializer* init=nullptr; IDMLCompiledOperator* cops[]={cop};
    HR(dml->CreateOperatorInitializer(1,cops,IID_PPV_ARGS(&init)));
    DML_BINDING_PROPERTIES ip=init->GetBindingProperties(), ep=cop->GetBindingProperties();
    UINT descN=std::max(ip.RequiredDescriptorCount,ep.RequiredDescriptorCount);
    printf("[diag] exec: descriptors=%u tempSize=%llu persistSize=%llu\n",ep.RequiredDescriptorCount,(unsigned long long)ep.TemporaryResourceSize,(unsigned long long)ep.PersistentResourceSize);

    D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors=descN; hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ID3D12DescriptorHeap* heap=nullptr; HR(dev->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap)));
    ID3D12DescriptorHeap* heaps[]={heap}; cl->SetDescriptorHeaps(1,heaps);

    DML_BINDING_TABLE_DESC btd{}; btd.Dispatchable=init; btd.CPUDescriptorHandle=heap->GetCPUDescriptorHandleForHeapStart(); btd.GPUDescriptorHandle=heap->GetGPUDescriptorHandleForHeapStart(); btd.SizeInDescriptors=descN;
    IDMLBindingTable* bt=nullptr; HR(dml->CreateBindingTable(&btd,IID_PPV_ARGS(&bt)));

    UINT64 tmpSz=std::max(ip.TemporaryResourceSize,ep.TemporaryResourceSize), perSz=ep.PersistentResourceSize;
    ID3D12Resource *tmp=nullptr,*per=nullptr;
    auto mkDefault=[&](UINT64 b)->ID3D12Resource*{ ID3D12Resource*r=nullptr; auto hp=heapProps(D3D12_HEAP_TYPE_DEFAULT); auto rd=bufDesc(b,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS); dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&rd,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&r)); return r; };
    if(tmpSz){ tmp=mkDefault(tmpSz); DML_BUFFER_BINDING bb{tmp,0,tmpSz}; DML_BINDING_DESC bd{DML_BINDING_TYPE_BUFFER,&bb}; if(ip.TemporaryResourceSize) bt->BindTemporaryResource(&bd); }
    if(perSz){ per=mkDefault(perSz); DML_BUFFER_BINDING bb{per,0,perSz}; DML_BINDING_DESC bd{DML_BINDING_TYPE_BUFFER,&bb}; bt->BindOutputs(1,&bd); }

    IDMLCommandRecorder* rec=nullptr; HR(dml->CreateCommandRecorder(IID_PPV_ARGS(&rec)));
    rec->RecordDispatch(cl,init,bt); flush();   // initialize once

    // --- input/output buffers, upload A and B ---
    UINT64 aBytes=(UINT64)M*K*4, bBytes=(UINT64)K*N*4, oBytes=(UINT64)M*N*4;
    auto upload=[&](std::vector<float>&data,UINT64 bytes)->ID3D12Resource*{
        ID3D12Resource* gpu=mkDefault(bytes);
        ID3D12Resource* up=nullptr; auto hp=heapProps(D3D12_HEAP_TYPE_UPLOAD); auto rd=bufDesc(bytes);
        dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&rd,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&up));
        void* p=nullptr; D3D12_RANGE nr{0,0}; up->Map(0,&nr,&p); memcpy(p,data.data(),bytes); up->Unmap(0,nullptr);
        auto b1=barrier(gpu,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST); cl->ResourceBarrier(1,&b1);
        cl->CopyBufferRegion(gpu,0,up,0,bytes);
        auto b2=barrier(gpu,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS); cl->ResourceBarrier(1,&b2);
        flush(); return gpu; };
    ID3D12Resource* aBuf=upload(A,aBytes);
    ID3D12Resource* bBuf=upload(B,bBytes);
    ID3D12Resource* oBuf=mkDefault(oBytes);
    if(!aBuf||!bBuf||!oBuf){ printf("[diag] null buffer aBuf=%p bBuf=%p oBuf=%p\n",(void*)aBuf,(void*)bBuf,(void*)oBuf); return 3; }
    // diag: read aBuf back to confirm the upload landed on the GPU
    auto readback=[&](ID3D12Resource* src,UINT64 bytes,D3D12_RESOURCE_STATES before)->std::vector<float>{
        std::vector<float> out(bytes/4); ID3D12Resource* rb=nullptr; auto hp=heapProps(D3D12_HEAP_TYPE_READBACK); auto rd=bufDesc(bytes);
        dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&rd,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&rb));
        auto b1=barrier(src,before,D3D12_RESOURCE_STATE_COPY_SOURCE); cl->ResourceBarrier(1,&b1);
        cl->CopyResource(rb,src);
        auto b2=barrier(src,D3D12_RESOURCE_STATE_COPY_SOURCE,before); cl->ResourceBarrier(1,&b2);
        flush(); void* p=nullptr; D3D12_RANGE r{0,(SIZE_T)bytes}; rb->Map(0,&r,&p); memcpy(out.data(),p,bytes); rb->Unmap(0,nullptr); rb->Release(); return out; };
    { auto a=readback(aBuf,aBytes,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      printf("[diag] aBuf GPU [0..2]=(%.5f,%.5f,%.5f) vs file (%.5f,%.5f,%.5f)\n",a[0],a[1],a[2],A[0],A[1],A[2]); }

    // rebind table for execution
    btd.Dispatchable=cop; HR(bt->Reset(&btd));
    if(tmpSz){ DML_BUFFER_BINDING bb{tmp,0,tmpSz}; DML_BINDING_DESC bd{DML_BINDING_TYPE_BUFFER,&bb}; bt->BindTemporaryResource(&bd); }
    if(perSz){ DML_BUFFER_BINDING bb{per,0,perSz}; DML_BINDING_DESC bd{DML_BINDING_TYPE_BUFFER,&bb}; bt->BindPersistentResource(&bd); }
    // GEMM has 3 input slots [A, B, C]; C is null in the op desc, so bind it as TYPE_NONE.
    DML_BUFFER_BINDING inBB[2]={ {aBuf,0,aBytes},{bBuf,0,bBytes} };
    DML_BINDING_DESC inBD[3]={ {DML_BINDING_TYPE_BUFFER,&inBB[0]},{DML_BINDING_TYPE_BUFFER,&inBB[1]},{DML_BINDING_TYPE_NONE,nullptr} };
    bt->BindInputs(3,inBD);
    DML_BUFFER_BINDING oBB{oBuf,0,oBytes}; DML_BINDING_DESC oBD{DML_BINDING_TYPE_BUFFER,&oBB}; bt->BindOutputs(1,&oBD);

    cl->SetDescriptorHeaps(1,heaps);
    // correctness pass: one dispatch, read back
    rec->RecordDispatch(cl,cop,bt); flush();
    HRESULT drr=dev->GetDeviceRemovedReason();
    if(drr!=S_OK) printf("[diag] DEVICE REMOVED after GEMM dispatch: 0x%08x\n",(unsigned)drr);
    std::vector<float> C(M*N);
    { ID3D12Resource* rb=nullptr; auto hp=heapProps(D3D12_HEAP_TYPE_READBACK); auto rd=bufDesc(oBytes);
      dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&rd,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&rb));
      auto b1=barrier(oBuf,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE); cl->ResourceBarrier(1,&b1);
      cl->CopyResource(rb,oBuf);
      auto b2=barrier(oBuf,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS); cl->ResourceBarrier(1,&b2);
      flush();
      void* p=nullptr; D3D12_RANGE r{0,(SIZE_T)oBytes}; rb->Map(0,&r,&p); memcpy(C.data(),p,oBytes); rb->Unmap(0,nullptr); rb->Release(); }

    // --- timing: N dispatches recorded into one list, executed once, fence-wait (mirrors matmul_run) ---
    cl->SetDescriptorHeaps(1,heaps);
    int IT=100;
    for(int i=0;i<IT;i++) rec->RecordDispatch(cl,cop,bt);   // warm/compile PSO
    flush();
    LARGE_INTEGER f,t0,t1; QueryPerformanceFrequency(&f);
    cl->SetDescriptorHeaps(1,heaps);
    for(int i=0;i<IT;i++) rec->RecordDispatch(cl,cop,bt);
    QueryPerformanceCounter(&t0);
    flush();
    QueryPerformanceCounter(&t1);
    double msPer=(double)(t1.QuadPart-t0.QuadPart)*1000.0/f.QuadPart/IT;
    printf("[perf] DirectML GEMM %ux%ux%u  %.3f ms/iter (%d iters, one submit)\n",M,K,N,msPer,IT);

    // correctness vs f64 reference (scale-normalized, same metric as matmul_run)
    double maxAbs=0,absmax=0; size_t worst=0;
    for(size_t i=0;i<(size_t)M*N;i++){ double a=std::fabs((double)C[i]-Cref[i]); if(a>maxAbs){maxAbs=a;worst=i;} if(std::fabs((double)Cref[i])>absmax)absmax=std::fabs((double)Cref[i]); }
    double normErr=maxAbs/absmax;
    printf("[C[0..2]] dml=(%.5f,%.5f,%.5f) ref=(%.5f,%.5f,%.5f)\n",C[0],C[1],C[2],Cref[0],Cref[1],Cref[2]);
    printf("[verify] max abs err=%.3e  scale=%.3f  scale-normalized err=%.2e\n",maxAbs,absmax,normErr);
    bool pass=normErr<1e-4;   // DML may reorder accumulation; looser than our fp32 kernel's 1e-5
    printf("=== %s: DirectML GEMM on %ls -> %s vs numpy f64 ===\n",pass?"PASS":"FAIL",adDesc.Description,pass?"matches":"MISMATCH");
    return pass?0:1;
}
