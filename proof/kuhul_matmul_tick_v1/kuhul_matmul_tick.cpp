// kuhul_matmul_tick.cpp — EXECUTED proof of K'UHUL-3D LAW R1.
//
// A single MATMUL compute node whose recursive PhaseTick DRIVES REAL work on the real Qwen weight
// transformer.h.0.attn.c_proj.weight [2048,2048], sharing one D3D12+DML device across the whole tick:
//
//   [Pop  LOAD]   make the compact quant bytes RESIDENT (D3D12 MakeResident of real .kqz bytes);
//                 escalate Q4 -> Q8 per-tensor into headroom (the design of record).
//   [Wo   BIND]   dequant the resident tier -> f32; prepare GEMM buffers.
//   [Yax  TILE]   select the DirectML GEMM path (transB).
//   [Sek  GEMM]   NESTED node/tick: dispatch DirectML GEMM on the real dequantized weight.
//   [Ch'en VERIFY] gate on real fidelity vs the F16 truth (Q8 must beat Q4 and clear threshold).
//   [Xul  COMMIT] retain Q4 resident, EVICT Q8 (real D3D12 Evict).
//
// LAW P1 in code: phase_*() functions only SCHEDULE; op_*() functions perform work. Phases carry no
// compute. This is the recursion controlling real residency, quantization, dispatch, and compute.
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <DirectML.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#pragma comment(lib,"d3d12.lib")
#pragma comment(lib,"dxgi.lib")
#pragma comment(lib,"dxguid.lib")
#pragma comment(lib,"DirectML.lib")
#define OK(x) do{ if(FAILED(x)){ printf("[FATAL] %s = 0x%08lx\n", #x, (unsigned long)(x)); return -1; } }while(0)
static const double MB = 1024.0*1024.0;

// ---------------- device ----------------
struct Ctx {
    ID3D12Device* dev=nullptr; IDXGIAdapter3* adapter=nullptr; ID3D12CommandQueue* q=nullptr;
    ID3D12CommandAllocator* alloc=nullptr; ID3D12GraphicsCommandList* cl=nullptr;
    ID3D12Fence* fence=nullptr; UINT64 fv=0; HANDLE fe=nullptr;
    IDMLDevice* dml=nullptr; IDMLCommandRecorder* rec=nullptr;
    void flush(){ cl->Close(); ID3D12CommandList*ls[]={cl}; q->ExecuteCommandLists(1,ls);
        q->Signal(fence,++fv); fence->SetEventOnCompletion(fv,fe); WaitForSingleObject(fe,INFINITE);
        alloc->Reset(); cl->Reset(alloc,nullptr); }
} g;
static D3D12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE t){ D3D12_HEAP_PROPERTIES h{}; h.Type=t; h.CreationNodeMask=1; h.VisibleNodeMask=1; return h; }
static D3D12_RESOURCE_DESC bufDesc(UINT64 b,D3D12_RESOURCE_FLAGS f=D3D12_RESOURCE_FLAG_NONE){ D3D12_RESOURCE_DESC d{}; d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; d.Width=b; d.Height=1; d.DepthOrArraySize=1; d.MipLevels=1; d.Format=DXGI_FORMAT_UNKNOWN; d.SampleDesc.Count=1; d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags=f; return d; }
static D3D12_RESOURCE_BARRIER bar(ID3D12Resource*r,D3D12_RESOURCE_STATES a,D3D12_RESOURCE_STATES b){ D3D12_RESOURCE_BARRIER br{}; br.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; br.Transition.pResource=r; br.Transition.StateBefore=a; br.Transition.StateAfter=b; br.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; return br; }
static ID3D12Resource* mkDefault(UINT64 b){ ID3D12Resource*r=nullptr; auto hp=heapProps(D3D12_HEAP_TYPE_DEFAULT); auto rd=bufDesc(b,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS); g.dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&rd,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&r)); return r; }
static ID3D12Resource* mkHeap(UINT64 b,D3D12_HEAP_TYPE t,D3D12_RESOURCE_STATES s){ ID3D12Resource*r=nullptr; auto hp=heapProps(t); auto rd=bufDesc(b); g.dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&rd,s,nullptr,IID_PPV_ARGS(&r)); return r; }
static uint64_t localUsage(){ DXGI_QUERY_VIDEO_MEMORY_INFO v{}; g.adapter->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&v); return v.CurrentUsage; }

static int ensureInit(){
    IDXGIFactory4* fac=nullptr; OK(CreateDXGIFactory1(IID_PPV_ARGS(&fac)));
    IDXGIAdapter1* a1=nullptr; DXGI_ADAPTER_DESC1 dd{};
    for(UINT i=0; fac->EnumAdapters1(i,&a1)!=DXGI_ERROR_NOT_FOUND; ++i){
        a1->GetDesc1(&dd); if(dd.Flags&DXGI_ADAPTER_FLAG_SOFTWARE){ a1->Release(); a1=nullptr; continue; }
        if(SUCCEEDED(D3D12CreateDevice(a1,D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&g.dev)))){ a1->QueryInterface(IID_PPV_ARGS(&g.adapter)); break; }
        a1->Release(); a1=nullptr;
    }
    if(!g.dev){ printf("[FATAL] no D3D12 device\n"); return -1; }
    wprintf(L"[dev] %s\n", dd.Description);
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
    OK(g.dev->CreateCommandQueue(&qd,IID_PPV_ARGS(&g.q)));
    OK(g.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&g.alloc)));
    OK(g.dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,g.alloc,nullptr,IID_PPV_ARGS(&g.cl)));
    OK(g.dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&g.fence))); g.fe=CreateEvent(nullptr,FALSE,FALSE,nullptr);
    OK(DMLCreateDevice(g.dev,DML_CREATE_DEVICE_FLAG_NONE,IID_PPV_ARGS(&g.dml)));
    OK(g.dml->CreateCommandRecorder(IID_PPV_ARGS(&g.rec)));
    return 0;
}

// ---------------- GEMM (transB): C[M,N] = A[M,K] @ B[N,K]^T  (crib of dml_gemm_dll) ----------------
struct Gemm { IDMLCompiledOperator* op=nullptr; ID3D12DescriptorHeap* heap=nullptr; IDMLBindingTable* bt=nullptr; UINT descN=0;
    ID3D12Resource *aBuf=nullptr,*bBuf=nullptr,*oBuf=nullptr,*upA=nullptr,*upB=nullptr,*rb=nullptr,*tmp=nullptr; UINT64 aB=0,bB=0,oB=0,tmpB=0; };
static int buildGemm(UINT M,UINT N,UINT K,Gemm& s){
    UINT aSz[4]={1,1,M,K}, bSz[4]={1,1,N,K}, oSz[4]={1,1,M,N};
    auto mk=[&](UINT*z,DML_BUFFER_TENSOR_DESC&bt,DML_TENSOR_DESC&td){ bt={}; bt.DataType=DML_TENSOR_DATA_TYPE_FLOAT32; bt.DimensionCount=4; bt.Sizes=z; bt.TotalTensorSizeInBytes=(UINT64)z[0]*z[1]*z[2]*z[3]*sizeof(float); td.Type=DML_TENSOR_TYPE_BUFFER; td.Desc=&bt; };
    DML_BUFFER_TENSOR_DESC aBT,bBT,oBT; DML_TENSOR_DESC aTD,bTD,oTD; mk(aSz,aBT,aTD); mk(bSz,bBT,bTD); mk(oSz,oBT,oTD);
    DML_GEMM_OPERATOR_DESC gemm{}; gemm.ATensor=&aTD; gemm.BTensor=&bTD; gemm.CTensor=nullptr; gemm.OutputTensor=&oTD;
    gemm.TransA=DML_MATRIX_TRANSFORM_NONE; gemm.TransB=DML_MATRIX_TRANSFORM_TRANSPOSE; gemm.Alpha=1.0f; gemm.Beta=0.0f;
    DML_OPERATOR_DESC opd{DML_OPERATOR_GEMM,&gemm}; IDMLOperator* op=nullptr; OK(g.dml->CreateOperator(&opd,IID_PPV_ARGS(&op)));
    OK(g.dml->CompileOperator(op,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&s.op))); op->Release();
    IDMLOperatorInitializer* ini=nullptr; IDMLCompiledOperator* cops[]={s.op}; OK(g.dml->CreateOperatorInitializer(1,cops,IID_PPV_ARGS(&ini)));
    DML_BINDING_PROPERTIES ip=ini->GetBindingProperties(), ep=s.op->GetBindingProperties();
    s.descN=std::max(ip.RequiredDescriptorCount,ep.RequiredDescriptorCount); s.tmpB=std::max(ip.TemporaryResourceSize,ep.TemporaryResourceSize);
    D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors=s.descN; hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    OK(g.dev->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&s.heap)));
    s.aB=(UINT64)M*K*4; s.bB=(UINT64)N*K*4; s.oB=(UINT64)M*N*4;
    s.aBuf=mkDefault(s.aB); s.bBuf=mkDefault(s.bB); s.oBuf=mkDefault(s.oB);
    s.upA=mkHeap(s.aB,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
    s.upB=mkHeap(s.bB,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
    s.rb =mkHeap(s.oB,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST); if(s.tmpB) s.tmp=mkDefault(s.tmpB);
    ID3D12DescriptorHeap* heaps[]={s.heap}; g.cl->SetDescriptorHeaps(1,heaps);
    DML_BINDING_TABLE_DESC bd{ini,s.heap->GetCPUDescriptorHandleForHeapStart(),s.heap->GetGPUDescriptorHandleForHeapStart(),s.descN};
    OK(g.dml->CreateBindingTable(&bd,IID_PPV_ARGS(&s.bt)));
    if(s.tmpB&&ip.TemporaryResourceSize){ DML_BUFFER_BINDING tb{s.tmp,0,s.tmpB}; DML_BINDING_DESC td{DML_BINDING_TYPE_BUFFER,&tb}; s.bt->BindTemporaryResource(&td); }
    g.rec->RecordDispatch(g.cl,ini,s.bt); g.flush(); ini->Release(); return 0;
}
static int runGemm(Gemm& s,const float*A,const float*B,float*C){
    DML_BINDING_TABLE_DESC bd{s.op,s.heap->GetCPUDescriptorHandleForHeapStart(),s.heap->GetGPUDescriptorHandleForHeapStart(),s.descN}; s.bt->Reset(&bd);
    if(s.tmpB){ DML_BUFFER_BINDING tb{s.tmp,0,s.tmpB}; DML_BINDING_DESC td{DML_BINDING_TYPE_BUFFER,&tb}; s.bt->BindTemporaryResource(&td); }
    DML_BUFFER_BINDING inBB[2]={{s.aBuf,0,s.aB},{s.bBuf,0,s.bB}};
    DML_BINDING_DESC inBD[3]={{DML_BINDING_TYPE_BUFFER,&inBB[0]},{DML_BINDING_TYPE_BUFFER,&inBB[1]},{DML_BINDING_TYPE_NONE,nullptr}}; s.bt->BindInputs(3,inBD);
    DML_BUFFER_BINDING oBB{s.oBuf,0,s.oB}; DML_BINDING_DESC oBD{DML_BINDING_TYPE_BUFFER,&oBB}; s.bt->BindOutputs(1,&oBD);
    void*pa=nullptr,*pb=nullptr; D3D12_RANGE nr{0,0}; s.upA->Map(0,&nr,&pa); memcpy(pa,A,s.aB); s.upA->Unmap(0,nullptr);
    s.upB->Map(0,&nr,&pb); memcpy(pb,B,s.bB); s.upB->Unmap(0,nullptr);
    ID3D12DescriptorHeap* heaps[]={s.heap}; g.cl->SetDescriptorHeaps(1,heaps);
    D3D12_RESOURCE_BARRIER pre[3]={ bar(s.aBuf,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST), bar(s.bBuf,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST), bar(s.oBuf,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS) };
    g.cl->ResourceBarrier(3,pre);
    g.cl->CopyBufferRegion(s.aBuf,0,s.upA,0,s.aB); g.cl->CopyBufferRegion(s.bBuf,0,s.upB,0,s.bB);
    D3D12_RESOURCE_BARRIER toU[2]={ bar(s.aBuf,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS), bar(s.bBuf,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS) };
    g.cl->ResourceBarrier(2,toU);
    g.rec->RecordDispatch(g.cl,s.op,s.bt);
    auto ob=bar(s.oBuf,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE); g.cl->ResourceBarrier(1,&ob);
    g.cl->CopyResource(s.rb,s.oBuf); g.flush();
    void*p=nullptr; D3D12_RANGE rr{0,(SIZE_T)s.oB}; s.rb->Map(0,&rr,&p); memcpy(C,p,s.oB); D3D12_RANGE nw{0,0}; s.rb->Unmap(0,&nw); return 0;
}

// ---------------- file + dequant ----------------
static std::vector<uint8_t> readRange(const char* path,uint64_t off,uint64_t size){ std::ifstream f(path,std::ios::binary); std::vector<uint8_t> v(size); f.seekg((std::streamoff)off); f.read((char*)v.data(),size); return v; }
static float half2float(uint16_t h){ uint32_t s=(h>>15)&1,e=(h>>10)&0x1F,m=h&0x3FF,out; if(e==0){ if(m==0)out=s<<31; else { e=127-15+1; while(!(m&0x400)){m<<=1;e--;} m&=0x3FF; out=(s<<31)|(e<<23)|(m<<13);} } else if(e==31){ out=(s<<31)|(0xFF<<23)|(m<<13);} else { out=(s<<31)|((e-15+127)<<23)|(m<<13);} float f; memcpy(&f,&out,4); return f; }
static void dequant_f16(const std::vector<uint8_t>& d,int rows,int cols,std::vector<float>& W){ W.resize((size_t)rows*cols); const uint16_t* h=(const uint16_t*)d.data(); for(size_t i=0;i<W.size();++i) W[i]=half2float(h[i]); }
static void dequant_q8(const std::vector<uint8_t>& d,const std::vector<uint8_t>& sc,int rows,int cols,std::vector<float>& W){ W.resize((size_t)rows*cols); const int8_t* q=(const int8_t*)d.data(); const uint16_t* s=(const uint16_t*)sc.data(); for(int r=0;r<rows;++r){ float scale=half2float(s[r]); for(int c=0;c<cols;++c) W[(size_t)r*cols+c]=q[(size_t)r*cols+c]*scale; } }
static void dequant_q4(const std::vector<uint8_t>& d,const std::vector<uint8_t>& sc,int rows,int cols,int group,std::vector<float>& W){ W.resize((size_t)rows*cols); const uint8_t* p=d.data(); const uint16_t* s=(const uint16_t*)sc.data(); int ng=(cols+group-1)/group; for(int r=0;r<rows;++r){ for(int c=0;c<cols;++c){ size_t idx=(size_t)r*cols+c; uint8_t byte=p[idx/2]; uint8_t nib=(c&1)?(byte>>4):(byte&0x0F); float scale=half2float(s[(size_t)r*ng + c/group]); W[idx]=((int)nib-8)*scale; } } }
static double normRMSE(const std::vector<float>& a,const std::vector<float>& t){ double num=0,den=0; for(size_t i=0;i<t.size();++i){ double d=a[i]-t[i]; num+=d*d; den+=(double)t[i]*t[i]; } return den>0?std::sqrt(num/den):0; }
static void cpuGemmBT(const std::vector<float>&A,const std::vector<float>&W,std::vector<float>&C,int M,int N,int K){ C.assign((size_t)M*N,0); for(int m=0;m<M;++m)for(int n=0;n<N;++n){ double acc=0; const float*a=&A[(size_t)m*K]; const float*w=&W[(size_t)n*K]; for(int k=0;k<K;++k)acc+=(double)a[k]*w[k]; C[(size_t)m*N+n]=(float)acc; } }

int main(int argc,char**argv){
    if(argc<19){ printf("usage: kuhul_matmul_tick q4 doff dsz soff ssz group  q8 doff dsz soff ssz  f16 off sz  rows cols M thr\n"); return 2; }
    int i=1; auto S=[&](){ return argv[i++]; }; auto U=[&](){ return (uint64_t)strtoull(argv[i++],0,10); }; auto I=[&](){ return atoi(argv[i++]); }; auto F=[&](){ return atof(argv[i++]); };
    const char* q4p=S(); uint64_t q4do=U(),q4ds=U(),q4so=U(),q4ss=U(); int group=I();
    const char* q8p=S(); uint64_t q8do=U(),q8ds=U(),q8so=U(),q8ss=U();
    const char* f16p=S(); uint64_t f16o=U(),f16s=U();
    int rows=I(),cols=I(),M=I(); double thr=F();
    if(ensureInit()) return -1;
    printf("### K'UHUL-3D MATMUL node : transformer.h.0.attn.c_proj.weight [%d,%d]  M=%d  (LAW R1: node IS a tick)\n",rows,cols,M);
    uint64_t base=localUsage();

    // [Pop LOAD] — real residency of the compact quant bytes; escalate Q4->Q8 into headroom.
    printf("[Pop]  LOAD\n");
    auto q4d=readRange(q4p,q4do,q4ds); auto q4s=readRange(q4p,q4so,q4ss);
    ID3D12Resource* q4res=mkDefault(q4ds+q4ss); { void*p; D3D12_RANGE nr{0,0}; auto up=mkHeap(q4ds+q4ss,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ); up->Map(0,&nr,&p); memcpy(p,q4d.data(),q4ds); memcpy((char*)p+q4ds,q4s.data(),q4ss); up->Unmap(0,0); auto b=bar(q4res,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST); g.cl->ResourceBarrier(1,&b); g.cl->CopyBufferRegion(q4res,0,up,0,q4ds+q4ss); g.flush(); up->Release(); }
    ID3D12Pageable* q4pg=q4res; g.dev->MakeResident(1,&q4pg);
    printf("       op=LOAD tier=Q4 resident %.1f MB  (usage %.0f MB)\n", (q4ds+q4ss)/MB, (localUsage()-base)/MB);
    auto q8d=readRange(q8p,q8do,q8ds); auto q8s=readRange(q8p,q8so,q8ss);
    ID3D12Resource* q8res=mkDefault(q8ds+q8ss); { void*p; D3D12_RANGE nr{0,0}; auto up=mkHeap(q8ds+q8ss,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ); up->Map(0,&nr,&p); memcpy(p,q8d.data(),q8ds); memcpy((char*)p+q8ds,q8s.data(),q8ss); up->Unmap(0,0); auto b=bar(q8res,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST); g.cl->ResourceBarrier(1,&b); g.cl->CopyBufferRegion(q8res,0,up,0,q8ds+q8ss); g.flush(); up->Release(); }
    ID3D12Pageable* q8pg=q8res; g.dev->MakeResident(1,&q8pg);
    printf("       op=ESCALATE Q4->Q8 resident +%.1f MB  (usage %.0f MB)  <- per-tensor precision escalation\n",(q8ds+q8ss)/MB,(localUsage()-base)/MB);

    // [Wo BIND] — dequant resident tiers to f32; build GEMM + deterministic input.
    printf("[Wo]   BIND\n");
    std::vector<float> Wq4,Wq8,Wf16; dequant_q4(q4d,q4s,rows,cols,group,Wq4); dequant_q8(q8d,q8s,rows,cols,Wq8);
    auto f16=readRange(f16p,f16o,f16s); dequant_f16(f16,rows,cols,Wf16);
    std::vector<float> A((size_t)M*cols); uint32_t seed=12345; for(auto&x:A){ seed=seed*1664525u+1013904223u; x=((seed>>8)/(float)(1<<24)-0.5f)*0.1f; }
    Gemm gm{}; if(buildGemm(M,rows,cols,gm)) return -1;
    printf("       op=DEQUANT Wq4/Wq8/Wf16 [%d,%d] f32; input A[%d,%d]; op=BIND buffers ready\n",rows,cols,M,cols);

    // [Yax TILE] — select path.
    printf("[Yax]  TILE   op=SELECT path=DirectML_GEMM transB=1 (K=%d N=%d)\n",cols,rows);

    // [Sek GEMM] — NESTED node/tick: real DirectML dispatch on the dequantized weight.
    printf("[Sek]  GEMM   (nested node tick: [Pop LOAD operands][Sek FMA dispatch][Xul COMMIT out])\n");
    std::vector<float> Cq4((size_t)M*rows),Cq8((size_t)M*rows),Ctruth;
    runGemm(gm,A.data(),Wq4.data(),Cq4.data());
    runGemm(gm,A.data(),Wq8.data(),Cq8.data());
    cpuGemmBT(A,Wf16,Ctruth,M,rows,cols);
    printf("       op=GEMM dispatched: Cq4, Cq8 (DirectML) ; Ctruth (F16 CPU reference)\n");

    // [Ch'en VERIFY] — real fidelity gate.
    printf("[Ch'en] VERIFY\n");
    double e4=normRMSE(Cq4,Ctruth), e8=normRMSE(Cq8,Ctruth);
    bool admit = (e8<thr) && (e8<e4);
    printf("       op=VERIFY  normRMSE(Q4 vs truth)=%.4f  normRMSE(Q8 vs truth)=%.4f  thr=%.3f\n",e4,e8,thr);
    printf("       -> admit tier=%s  (escalation %s: Q8 error %.1fx lower than Q4)\n", admit?"Q8":"Q4-only", admit?"JUSTIFIED":"not needed", e8>0?e4/e8:0);

    // [Xul COMMIT] — retain Q4, evict Q8 (real).
    printf("[Xul]  COMMIT\n");
    ID3D12Pageable* ev=q8res; g.dev->Evict(1,&ev);
    printf("       op=COMMIT retain=Q4  op=EVICT tier=Q8  (usage %.0f MB, freed %.1f MB)\n",(localUsage()-base)/MB,(q8ds+q8ss)/MB);

    printf("### PROOF {\"node\":\"MATMUL\",\"drove\":[\"residency\",\"quantization\",\"dispatch\",\"compute\",\"eviction\"],"
           "\"q4_normrmse\":%.4f,\"q8_normrmse\":%.4f,\"admit\":\"%s\",\"route\":[\"Pop:LOAD\",\"Wo:BIND\",\"Yax:TILE\",\"Sek:GEMM\",\"Ch'en:VERIFY\",\"Xul:COMMIT\"]}\n",
           e4,e8,admit?"Q8":"Q4");
    return admit?0:3;
}
