// dml_mha_test.cpp — DE-RISK: does DML_OPERATOR_MULTIHEAD_ATTENTION run causal self-attention
// correctly on the HD 4600 (FL 11_1)? Feeds a packed QKV (gpt2 c_attn layout) + a causal
// relative-position bias (-inf upper triangle), checks vs a C++ causal-MHA reference.
#define NOMINMAX
#define DML_TARGET_VERSION_USE_LATEST 1
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <DirectML.h>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>
#pragma comment(lib,"d3d12.lib")
#pragma comment(lib,"dxgi.lib")
#pragma comment(lib,"dxguid.lib")
#pragma comment(lib,"DirectML.lib")
#define HR(x) do{ if(FAILED(x)){ printf("FAIL %s\n",#x); return {}; } }while(0)

static ID3D12Device* dev; static ID3D12CommandQueue* q; static ID3D12CommandAllocator* alloc;
static ID3D12GraphicsCommandList* cl; static ID3D12Fence* fence; static UINT64 fv; static HANDLE fe;
static IDMLDevice* dml; static IDMLCommandRecorder* rec;

static D3D12_HEAP_PROPERTIES hp(D3D12_HEAP_TYPE t){ D3D12_HEAP_PROPERTIES h{}; h.Type=t; h.CreationNodeMask=1; h.VisibleNodeMask=1; return h; }
static D3D12_RESOURCE_DESC bd(UINT64 b,D3D12_RESOURCE_FLAGS f=D3D12_RESOURCE_FLAG_NONE){ D3D12_RESOURCE_DESC d{}; d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; d.Width=b; d.Height=1; d.DepthOrArraySize=1; d.MipLevels=1; d.SampleDesc.Count=1; d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags=f; return d; }
static D3D12_RESOURCE_BARRIER br(ID3D12Resource*r,D3D12_RESOURCE_STATES a,D3D12_RESOURCE_STATES b){ D3D12_RESOURCE_BARRIER x{}; x.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; x.Transition.pResource=r; x.Transition.StateBefore=a; x.Transition.StateAfter=b; x.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; return x; }
static void flush(){ cl->Close(); ID3D12CommandList*ls[]={cl}; q->ExecuteCommandLists(1,ls); q->Signal(fence,++fv); fence->SetEventOnCompletion(fv,fe); WaitForSingleObject(fe,INFINITE); alloc->Reset(); cl->Reset(alloc,nullptr); }
static ID3D12Resource* mkDef(UINT64 b){ ID3D12Resource*r=nullptr; auto h=hp(D3D12_HEAP_TYPE_DEFAULT); auto d=bd(b,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS); dev->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&r)); return r; }
static ID3D12Resource* upload(const std::vector<float>& v){ UINT64 b=v.size()*4; ID3D12Resource* g=mkDef(b); auto h=hp(D3D12_HEAP_TYPE_UPLOAD); auto d=bd(b); ID3D12Resource*u=nullptr; dev->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&u)); void*p; D3D12_RANGE nr{0,0}; u->Map(0,&nr,&p); memcpy(p,v.data(),b); u->Unmap(0,nullptr); auto b1=br(g,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST); cl->ResourceBarrier(1,&b1); cl->CopyBufferRegion(g,0,u,0,b); auto b2=br(g,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS); cl->ResourceBarrier(1,&b2); flush(); u->Release(); return g; }

static std::vector<float> runOp(IDMLCompiledOperator* op, std::vector<ID3D12Resource*> ins, UINT outN){
    IDMLOperatorInitializer* ini=nullptr; IDMLCompiledOperator* cops[]={op}; HR(dml->CreateOperatorInitializer(1,cops,IID_PPV_ARGS(&ini)));
    DML_BINDING_PROPERTIES ip=ini->GetBindingProperties(), ep=op->GetBindingProperties();
    UINT descN=std::max(ip.RequiredDescriptorCount,ep.RequiredDescriptorCount);
    UINT64 tmpB=std::max(ip.TemporaryResourceSize,ep.TemporaryResourceSize), perB=ep.PersistentResourceSize;
    ID3D12DescriptorHeap* heap=nullptr; D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors=descN; hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE; HR(dev->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap)));
    ID3D12DescriptorHeap* heaps[]={heap}; cl->SetDescriptorHeaps(1,heaps);
    ID3D12Resource* tmp=tmpB?mkDef(tmpB):nullptr; ID3D12Resource* per=perB?mkDef(perB):nullptr;
    IDMLBindingTable* bt=nullptr; DML_BINDING_TABLE_DESC td{}; td.Dispatchable=ini; td.CPUDescriptorHandle=heap->GetCPUDescriptorHandleForHeapStart(); td.GPUDescriptorHandle=heap->GetGPUDescriptorHandleForHeapStart(); td.SizeInDescriptors=descN; HR(dml->CreateBindingTable(&td,IID_PPV_ARGS(&bt)));
    if(tmpB && ip.TemporaryResourceSize){ DML_BUFFER_BINDING b{tmp,0,tmpB}; DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER,&b}; bt->BindTemporaryResource(&d); }
    if(perB){ DML_BUFFER_BINDING b{per,0,perB}; DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER,&b}; bt->BindOutputs(1,&d); }
    rec->RecordDispatch(cl,ini,bt); flush(); ini->Release();
    ID3D12Resource* out=mkDef((UINT64)outN*4);
    td.Dispatchable=op; bt->Reset(&td);
    if(tmpB){ DML_BUFFER_BINDING b{tmp,0,tmpB}; DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER,&b}; bt->BindTemporaryResource(&d); }
    if(perB){ DML_BUFFER_BINDING b{per,0,perB}; DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER,&b}; bt->BindPersistentResource(&d); }
    // null entries -> DML_BINDING_TYPE_NONE (optional input slots must be bound positionally).
    std::vector<DML_BUFFER_BINDING> ibb(ins.size());
    std::vector<DML_BINDING_DESC> ibd(ins.size());
    for(size_t i=0;i<ins.size();i++){ if(ins[i]){ auto rd=ins[i]->GetDesc(); ibb[i]={ins[i],0,rd.Width}; ibd[i]={DML_BINDING_TYPE_BUFFER,&ibb[i]}; } else { ibd[i]={DML_BINDING_TYPE_NONE,nullptr}; } }
    bt->BindInputs((UINT)ibd.size(), ibd.data());
    // MHA has 3 output slots (Output, OutputPresentKey, OutputPresentValue) — bind positionally.
    DML_BUFFER_BINDING ob{out,0,(UINT64)outN*4};
    DML_BINDING_DESC od[3]={ {DML_BINDING_TYPE_BUFFER,&ob}, {DML_BINDING_TYPE_NONE,nullptr}, {DML_BINDING_TYPE_NONE,nullptr} };
    bt->BindOutputs(3,od);
    cl->SetDescriptorHeaps(1,heaps); rec->RecordDispatch(cl,op,bt); flush();
    std::vector<float> res(outN); ID3D12Resource* rb=nullptr; auto h=hp(D3D12_HEAP_TYPE_READBACK); auto d=bd((UINT64)outN*4); dev->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&rb));
    auto b1=br(out,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE); cl->ResourceBarrier(1,&b1); cl->CopyResource(rb,out); flush();
    void*p; D3D12_RANGE rr{0,(SIZE_T)outN*4}; rb->Map(0,&rr,&p); memcpy(res.data(),p,outN*4); D3D12_RANGE nw{0,0}; rb->Unmap(0,&nw); rb->Release();
    return res;
}
static DML_TENSOR_DESC Td(DML_BUFFER_TENSOR_DESC& bt, std::vector<UINT> sz){ bt={}; bt.DataType=DML_TENSOR_DATA_TYPE_FLOAT32; bt.DimensionCount=(UINT)sz.size(); static std::vector<std::vector<UINT>> keep; keep.push_back(sz); bt.Sizes=keep.back().data(); UINT64 e=1; for(UINT s: sz) e*=s; bt.TotalTensorSizeInBytes=e*4; DML_TENSOR_DESC t{DML_TENSOR_TYPE_BUFFER,&bt}; return t; }

int main(){
    IDXGIFactory4* fac; CreateDXGIFactory1(IID_PPV_ARGS(&fac)); IDXGIAdapter1* ad=nullptr; DXGI_ADAPTER_DESC1 dd{};
    for(UINT i=0; fac->EnumAdapters1(i,&ad)!=DXGI_ERROR_NOT_FOUND; ++i){ ad->GetDesc1(&dd); if(dd.Flags&DXGI_ADAPTER_FLAG_SOFTWARE){ad->Release();ad=nullptr;continue;} if(SUCCEEDED(D3D12CreateDevice(ad,D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&dev)))) break; ad->Release(); ad=nullptr; }
    if(!dev){ printf("no d3d12\n"); return 1; }
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type=D3D12_COMMAND_LIST_TYPE_DIRECT; dev->CreateCommandQueue(&qd,IID_PPV_ARGS(&q));
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&alloc)); dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,alloc,nullptr,IID_PPV_ARGS(&cl));
    dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)); fe=CreateEvent(nullptr,FALSE,FALSE,nullptr);
    DMLCreateDevice(dev,DML_CREATE_DEVICE_FLAG_NONE,IID_PPV_ARGS(&dml)); dml->CreateCommandRecorder(IID_PPV_ARGS(&rec));
    printf("[dev] %ls\n", dd.Description);

    const UINT S=4, Hn=2, Hd=3, E=Hn*Hd;   // seq, heads, head_dim, embed
    const float scale = 1.0f/std::sqrt((float)Hd);

    // What DML feature level does this device actually support? (MHA needs 6.1 = 0x6100)
    {
        DML_FEATURE_LEVEL want[]={DML_FEATURE_LEVEL_1_0,DML_FEATURE_LEVEL_2_0,DML_FEATURE_LEVEL_3_0,DML_FEATURE_LEVEL_4_0,DML_FEATURE_LEVEL_5_0,DML_FEATURE_LEVEL_5_1,DML_FEATURE_LEVEL_6_0,DML_FEATURE_LEVEL_6_1,DML_FEATURE_LEVEL_6_2};
        DML_FEATURE_QUERY_FEATURE_LEVELS qy{ (UINT)(sizeof(want)/sizeof(want[0])), want };
        DML_FEATURE_DATA_FEATURE_LEVELS data{};
        if(SUCCEEDED(dml->CheckFeatureSupport(DML_FEATURE_FEATURE_LEVELS,sizeof(qy),&qy,sizeof(data),&data)))
            printf("[dml] max supported feature level = 0x%04x (MHA needs 0x6100)\n",(unsigned)data.MaxSupportedFeatureLevel);
    }

    // --- PROBE: which stacked-QKV / output dim convention does DML MHA accept here? ---
    auto probe=[&](const char* name, std::vector<UINT> qkvDims, std::vector<UINT> outDims, bool rpb){
        DML_BUFFER_TENSOR_DESC qb,rb,ob; DML_TENSOR_DESC qt=Td(qb,qkvDims), rt=Td(rb,{1,Hn,S,S}), ot=Td(ob,outDims);
        DML_MULTIHEAD_ATTENTION_OPERATOR_DESC m{}; m.StackedQueryKeyValueTensor=&qt; if(rpb) m.RelativePositionBiasTensor=&rt; m.OutputTensor=&ot;
        m.Scale=scale; m.MaskFilterValue=-1e9f; m.HeadCount=Hn; m.MaskType=DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
        DML_OPERATOR_DESC od{DML_OPERATOR_MULTIHEAD_ATTENTION,&m};
        IDMLOperator* o=nullptr; HRESULT ch=dml->CreateOperator(&od,IID_PPV_ARGS(&o)); HRESULT ph=S_OK;
        if(SUCCEEDED(ch)){ IDMLCompiledOperator* c=nullptr; ph=dml->CompileOperator(o,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&c)); if(c)c->Release(); o->Release(); }
        printf("  [probe %-22s] create=0x%08x compile=0x%08x %s\n", name,(unsigned)ch,(unsigned)ph, (SUCCEEDED(ch)&&SUCCEEDED(ph))?"<== OK":"");
    };
    printf("[probe] rpb=off\n");
    probe("qkv5D{1,S,3,Hn,Hd} out3D", {1,S,3,Hn,Hd}, {1,S,E}, false);
    probe("qkv5D out4D",              {1,S,3,Hn,Hd}, {1,1,S,E}, false);
    probe("qkv4D{1,S,3,E} out4D",     {1,S,3,E},     {1,1,S,E}, false);
    probe("qkv3D{1,S,3E} out3D",      {1,S,3*E},     {1,S,E}, false);
    probe("qkv4D{1,S,3,E} out3D",     {1,S,3,E},     {1,S,E}, false);
    // separate Q/K/V probe
    auto probeQKV=[&](const char* name, std::vector<UINT> qkv, std::vector<UINT> outDims){
        DML_BUFFER_TENSOR_DESC qb,kb,vb,ob; DML_TENSOR_DESC qt=Td(qb,qkv),kt=Td(kb,qkv),vt=Td(vb,qkv),ot=Td(ob,outDims);
        DML_MULTIHEAD_ATTENTION_OPERATOR_DESC m{}; m.QueryTensor=&qt; m.KeyTensor=&kt; m.ValueTensor=&vt; m.OutputTensor=&ot;
        m.Scale=scale; m.MaskFilterValue=-1e9f; m.HeadCount=Hn; m.MaskType=DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
        DML_OPERATOR_DESC od{DML_OPERATOR_MULTIHEAD_ATTENTION,&m};
        IDMLOperator* o=nullptr; HRESULT ch=dml->CreateOperator(&od,IID_PPV_ARGS(&o)); HRESULT ph=S_OK;
        if(SUCCEEDED(ch)){ IDMLCompiledOperator* c=nullptr; ph=dml->CompileOperator(o,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&c)); if(c)c->Release(); o->Release(); }
        printf("  [probeQKV %-20s] create=0x%08x compile=0x%08x %s\n", name,(unsigned)ch,(unsigned)ph, (SUCCEEDED(ch)&&SUCCEEDED(ph))?"<== OK":"");
    };
    printf("[probe] separate Q/K/V\n");
    probeQKV("qkv3D{1,S,E} out3D",   {1,S,E},    {1,S,E});
    probeQKV("qkv4D{1,1,S,E} out4D", {1,1,S,E},  {1,1,S,E});
    probeQKV("qkv4D{1,S,Hn,Hd}",     {1,S,Hn,Hd},{1,S,E});
    probeQKV("qkv4D{1,S,1,E} out",   {1,S,1,E},  {1,S,1,E});
    printf("[chosen] separate Q/K/V {1,S,E}, output {1,S,E}, HeadCount=%u, causal via RelativePositionBias\n", Hn);

    // gpt2 c_attn output [S,3E] = [q(E)|k(E)|v(E)] per token; split into separate Q/K/V [S,E].
    std::vector<float> qkv(S*3*Hn*Hd); for(size_t i=0;i<qkv.size();i++) qkv[i]=std::sin(0.3f*(float)i)*0.8f;
    std::vector<float> Qd(S*E), Kd(S*E), Vd(S*E);
    for(UINT s=0;s<S;s++) for(UINT h=0;h<Hn;h++) for(UINT d=0;d<Hd;d++){
        Qd[s*E+h*Hd+d]=qkv[((s*3+0)*Hn+h)*Hd+d]; Kd[s*E+h*Hd+d]=qkv[((s*3+1)*Hn+h)*Hd+d]; Vd[s*E+h*Hd+d]=qkv[((s*3+2)*Hn+h)*Hd+d];
    }
    // causal relative-position bias [1,Hn,S,S]: 0 where j<=i, -1e9 where j>i
    std::vector<float> rpb(Hn*S*S); for(UINT h=0;h<Hn;h++) for(UINT i=0;i<S;i++) for(UINT j=0;j<S;j++) rpb[(h*S+i)*S+j] = (j<=i)?0.0f:-1e9f;

    DML_BUFFER_TENSOR_DESC qB,kB,vB,rpbB,outB; DML_TENSOR_DESC qT=Td(qB,{1,S,E}), kT=Td(kB,{1,S,E}), vT=Td(vB,{1,S,E}), rpbT=Td(rpbB,{1,Hn,S,S}), outT=Td(outB,{1,S,E});
    DML_MULTIHEAD_ATTENTION_OPERATOR_DESC mha{}; mha.QueryTensor=&qT; mha.KeyTensor=&kT; mha.ValueTensor=&vT; mha.RelativePositionBiasTensor=&rpbT; mha.OutputTensor=&outT;
    mha.Scale=scale; mha.MaskFilterValue=-1e9f; mha.HeadCount=Hn; mha.MaskType=DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
    DML_OPERATOR_DESC od{DML_OPERATOR_MULTIHEAD_ATTENTION,&mha};
    IDMLOperator* o=nullptr; HRESULT chr=dml->CreateOperator(&od,IID_PPV_ARGS(&o));
    if(FAILED(chr)){ printf("[MHA] CreateOperator FAILED 0x%08x\n",(unsigned)chr); return 2; }
    IDMLCompiledOperator* c=nullptr; HRESULT phr=dml->CompileOperator(o,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&c));
    if(FAILED(phr)){ printf("[MHA] CompileOperator FAILED 0x%08x\n",(unsigned)phr); return 2; }
    // MHA input slots (struct order): Query,Key,Value, StackedQK,StackedKV,StackedQKV, Bias, Mask,
    // RelativePositionBias, PastKey,PastValue. Bind the 4 real ones at their slots, NONE elsewhere.
    ID3D12Resource *qR=upload(Qd),*kR=upload(Kd),*vR=upload(Vd),*rR=upload(rpb);
    auto out=runOp(c,{qR,kR,vR, nullptr,nullptr,nullptr, nullptr, nullptr, rR, nullptr,nullptr}, S*E);

    // C++ causal-MHA reference (Q/K/V laid out [S, h*Hd+d])
    auto Q=[&](UINT s,UINT h,UINT d){ return Qd[s*E+h*Hd+d]; };
    auto K=[&](UINT s,UINT h,UINT d){ return Kd[s*E+h*Hd+d]; };
    auto V=[&](UINT s,UINT h,UINT d){ return Vd[s*E+h*Hd+d]; };
    std::vector<float> ref(S*E);
    for(UINT h=0;h<Hn;h++) for(UINT i=0;i<S;i++){
        std::vector<float> sc(i+1); float mx=-1e30f;
        for(UINT j=0;j<=i;j++){ float s=0; for(UINT d=0;d<Hd;d++) s+=Q(i,h,d)*K(j,h,d); s*=scale; sc[j]=s; mx=std::max(mx,s); }
        float sum=0; for(UINT j=0;j<=i;j++){ sc[j]=std::exp(sc[j]-mx); sum+=sc[j]; }
        for(UINT d=0;d<Hd;d++){ float acc=0; for(UINT j=0;j<=i;j++) acc+=sc[j]/sum*V(j,h,d); ref[i*E + h*Hd + d]=acc; }
    }
    double maxAbs=0,absmax=0; for(size_t i=0;i<ref.size();i++){ maxAbs=std::max(maxAbs,(double)std::fabs(out[i]-ref[i])); absmax=std::max(absmax,(double)std::fabs(ref[i])); }
    double nrm=maxAbs/absmax;
    printf("[out[0..3]] gpu=(%.5f,%.5f,%.5f,%.5f)\n           ref=(%.5f,%.5f,%.5f,%.5f)\n",out[0],out[1],out[2],out[3],ref[0],ref[1],ref[2],ref[3]);
    printf("[verify] scale-norm err %.2e\n", nrm);
    bool pass=nrm<1e-3;
    printf("=== %s: DML_OPERATOR_MULTIHEAD_ATTENTION causal self-attn on HD4600 (FL 11_1) ===\n", pass?"PASS":"FAIL");
    return pass?0:1;
}
