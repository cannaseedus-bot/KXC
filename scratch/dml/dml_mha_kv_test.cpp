// dml_mha_kv_test.cpp — DE-RISK: does DML MHA's KV cache (PastKey/Value in, OutputPresentKey/
// Value out) run correctly on the HD 4600? Models one DECODE step: a single new-token query
// attends over past_len cached keys/values + the current one. Verifies output AND the appended
// present K/V vs a C++ reference. This is the primitive the on-device generation loop needs.
#define NOMINMAX
#define DML_TARGET_VERSION_USE_LATEST 1
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <DirectML.h>
#include <vector>
#include <cstdio>
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
static std::vector<float> readback(ID3D12Resource* src,UINT n){ std::vector<float> out(n); ID3D12Resource* rb=nullptr; auto h=hp(D3D12_HEAP_TYPE_READBACK); auto d=bd((UINT64)n*4); dev->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&rb)); auto b1=br(src,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE); cl->ResourceBarrier(1,&b1); cl->CopyResource(rb,src); auto b2=br(src,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS); cl->ResourceBarrier(1,&b2); flush(); void*p; D3D12_RANGE rr{0,(SIZE_T)n*4}; rb->Map(0,&rr,&p); memcpy(out.data(),p,n*4); D3D12_RANGE nw{0,0}; rb->Unmap(0,&nw); rb->Release(); return out; }
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

    const UINT Hn=2, Hd=3, E=Hn*Hd, P=3, Tot=P+1;   // heads, head_dim, embed, past_len, total after decode
    const float scale=1.0f/std::sqrt((float)Hd);
    // new-token Q/K/V [1,1,E]; past K/V [1,Hn,P,Hd]
    std::vector<float> Q(E),Kn(E),Vn(E),pastK(Hn*P*Hd),pastV(Hn*P*Hd);
    for(UINT i=0;i<E;i++){ Q[i]=std::sin(0.3f*i)*0.7f; Kn[i]=std::cos(0.2f*i)*0.6f; Vn[i]=std::sin(0.17f*i+1)*0.5f; }
    for(UINT i=0;i<pastK.size();i++){ pastK[i]=std::sin(0.11f*i)*0.5f; pastV[i]=std::cos(0.13f*i)*0.4f; }

    // PROBE past/present shape convention
    auto probe=[&](const char* name, std::vector<UINT> pastDims, std::vector<UINT> presDims){
        DML_BUFFER_TENSOR_DESC qb,kb,vb,pkb,pvb,ob,okb,ovb; DML_TENSOR_DESC qt=Td(qb,{1,1,E}),kt=Td(kb,{1,1,E}),vt=Td(vb,{1,1,E}),pkt=Td(pkb,pastDims),pvt=Td(pvb,pastDims),ot=Td(ob,{1,1,E}),okt=Td(okb,presDims),ovt=Td(ovb,presDims);
        DML_MULTIHEAD_ATTENTION_OPERATOR_DESC m{}; m.QueryTensor=&qt; m.KeyTensor=&kt; m.ValueTensor=&vt; m.PastKeyTensor=&pkt; m.PastValueTensor=&pvt; m.OutputTensor=&ot; m.OutputPresentKeyTensor=&okt; m.OutputPresentValueTensor=&ovt;
        m.Scale=scale; m.MaskFilterValue=-1e9f; m.HeadCount=Hn; m.MaskType=DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
        DML_OPERATOR_DESC od{DML_OPERATOR_MULTIHEAD_ATTENTION,&m};
        IDMLOperator* o=nullptr; HRESULT ch=dml->CreateOperator(&od,IID_PPV_ARGS(&o)); HRESULT ph=S_OK; IDMLCompiledOperator* c=nullptr;
        if(SUCCEEDED(ch)){ ph=dml->CompileOperator(o,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&c)); o->Release(); }
        printf("  [probe %-26s] create=0x%08x compile=0x%08x %s\n", name,(unsigned)ch,(unsigned)ph,(SUCCEEDED(ch)&&SUCCEEDED(ph))?"<== OK":"");
        return (SUCCEEDED(ch)&&SUCCEEDED(ph))? c : nullptr;
    };
    IDMLCompiledOperator* c=nullptr;
    if(!c) c=probe("past[1,Hn,P,Hd] pres[1,Hn,Tot,Hd]", {1,Hn,P,Hd}, {1,Hn,Tot,Hd});
    if(!c) c=probe("past[Hn,P,Hd] pres[Hn,Tot,Hd]",     {Hn,P,Hd},   {Hn,Tot,Hd});
    if(!c){ printf("=== FAIL: no MHA KV-cache shape accepted (create/compile) ===\n"); return 2; }

    // bind + run one decode step. Input slots: Q0 K1 V2 (3,4,5,6,7,8 null) PastKey9 PastValue10.
    // Output slots: Output0 PresentKey1 PresentValue2.
    IDMLOperatorInitializer* ini=nullptr; IDMLCompiledOperator* cops[]={c}; dml->CreateOperatorInitializer(1,cops,IID_PPV_ARGS(&ini));
    DML_BINDING_PROPERTIES ip=ini->GetBindingProperties(), ep=c->GetBindingProperties();
    UINT descN=std::max(ip.RequiredDescriptorCount,ep.RequiredDescriptorCount); UINT64 tmpB=std::max(ip.TemporaryResourceSize,ep.TemporaryResourceSize);
    ID3D12DescriptorHeap* heap=nullptr; D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors=descN; hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE; dev->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap));
    ID3D12DescriptorHeap* heaps[]={heap}; cl->SetDescriptorHeaps(1,heaps);
    ID3D12Resource* tmp=tmpB?mkDef(tmpB):nullptr;
    IDMLBindingTable* bt=nullptr; DML_BINDING_TABLE_DESC td{}; td.Dispatchable=ini; td.CPUDescriptorHandle=heap->GetCPUDescriptorHandleForHeapStart(); td.GPUDescriptorHandle=heap->GetGPUDescriptorHandleForHeapStart(); td.SizeInDescriptors=descN; dml->CreateBindingTable(&td,IID_PPV_ARGS(&bt));
    if(tmpB&&ip.TemporaryResourceSize){ DML_BUFFER_BINDING b{tmp,0,tmpB}; DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER,&b}; bt->BindTemporaryResource(&d); }
    rec->RecordDispatch(cl,ini,bt); flush(); ini->Release();

    ID3D12Resource *qR=upload(Q),*kR=upload(Kn),*vR=upload(Vn),*pkR=upload(pastK),*pvR=upload(pastV);
    ID3D12Resource *outR=mkDef((UINT64)E*4),*okR=mkDef((UINT64)Hn*Tot*Hd*4),*ovR=mkDef((UINT64)Hn*Tot*Hd*4);
    td.Dispatchable=c; bt->Reset(&td);
    if(tmpB){ DML_BUFFER_BINDING b{tmp,0,tmpB}; DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER,&b}; bt->BindTemporaryResource(&d); }
    ID3D12Resource* inSlots[11]={qR,kR,vR, nullptr,nullptr,nullptr, nullptr,nullptr,nullptr, pkR,pvR};
    DML_BUFFER_BINDING ibb[11]; DML_BINDING_DESC ibd[11];
    for(int i=0;i<11;i++){ if(inSlots[i]){ auto r=inSlots[i]->GetDesc(); ibb[i]={inSlots[i],0,r.Width}; ibd[i]={DML_BINDING_TYPE_BUFFER,&ibb[i]}; } else ibd[i]={DML_BINDING_TYPE_NONE,nullptr}; }
    bt->BindInputs(11,ibd);
    ID3D12Resource* outSlots[3]={outR,okR,ovR}; DML_BUFFER_BINDING obb[3]; DML_BINDING_DESC obd[3];
    for(int i=0;i<3;i++){ auto r=outSlots[i]->GetDesc(); obb[i]={outSlots[i],0,r.Width}; obd[i]={DML_BINDING_TYPE_BUFFER,&obb[i]}; }
    bt->BindOutputs(3,obd);
    cl->SetDescriptorHeaps(1,heaps); rec->RecordDispatch(cl,c,bt); flush();

    auto out=readback(outR,E); auto presK=readback(okR,Hn*Tot*Hd); auto presV=readback(ovR,Hn*Tot*Hd);

    // C++ reference: full K/V per head = [pastK(P) ; newK(1)], causal (query is newest -> all Tot allowed)
    auto qh=[&](UINT h,UINT d){ return Q[h*Hd+d]; };
    auto pk=[&](UINT h,UINT t,UINT d){ return pastK[(h*P+t)*Hd+d]; };
    auto pv=[&](UINT h,UINT t,UINT d){ return pastV[(h*P+t)*Hd+d]; };
    auto nk=[&](UINT h,UINT d){ return Kn[h*Hd+d]; };
    auto nv=[&](UINT h,UINT d){ return Vn[h*Hd+d]; };
    std::vector<float> ref(E);
    for(UINT h=0;h<Hn;h++){
        std::vector<float> sc(Tot); float mx=-1e30f;
        for(UINT t=0;t<Tot;t++){ float s=0; for(UINT d=0;d<Hd;d++){ float kv=(t<P)?pk(h,t,d):nk(h,d); s+=qh(h,d)*kv; } s*=scale; sc[t]=s; mx=std::max(mx,s); }
        float sum=0; for(UINT t=0;t<Tot;t++){ sc[t]=std::exp(sc[t]-mx); sum+=sc[t]; }
        for(UINT d=0;d<Hd;d++){ float acc=0; for(UINT t=0;t<Tot;t++){ float vv=(t<P)?pv(h,t,d):nv(h,d); acc+=sc[t]/sum*vv; } ref[h*Hd+d]=acc; }
    }
    // present K/V layout [Hn,Tot,Hd]
    auto PK=[&](UINT h,UINT t,UINT d){ return presK[(h*Tot+t)*Hd+d]; };
    auto PV=[&](UINT h,UINT t,UINT d){ return presV[(h*Tot+t)*Hd+d]; };

    // (1) shape invariant: present.seq == past.seq + 1  (structural, from the accepted probe)
    bool i1 = (Tot == P+1);
    // (2) prefix invariant: present[0:P] == past   (K and V)
    double pfx=0; for(UINT h=0;h<Hn;h++) for(UINT t=0;t<P;t++) for(UINT d=0;d<Hd;d++){ pfx=std::max(pfx,(double)std::fabs(PK(h,t,d)-pk(h,t,d))); pfx=std::max(pfx,(double)std::fabs(PV(h,t,d)-pv(h,t,d))); }
    // (3) append invariant: present[P] == new K/V for the incoming token
    double app=0; for(UINT h=0;h<Hn;h++) for(UINT d=0;d<Hd;d++){ app=std::max(app,(double)std::fabs(PK(h,P,d)-nk(h,d))); app=std::max(app,(double)std::fabs(PV(h,P,d)-nv(h,d))); }
    // (4) attention-output correctness vs CPU reference
    double oe=0,om=0; for(UINT i=0;i<E;i++){ oe=std::max(oe,(double)std::fabs(out[i]-ref[i])); om=std::max(om,(double)std::fabs(ref[i])); }
    double outn=oe/om;

    printf("[inv 1 shape ] present.seq=%u == past.seq+1=%u : %s\n", Tot, P+1, i1?"OK":"FAIL");
    printf("[inv 2 prefix] present[0:P] == past (K&V)   maxabs %.2e : %s\n", pfx, pfx<1e-6?"OK":"FAIL");
    printf("[inv 3 append] present[P]   == new  (K&V)   maxabs %.2e : %s\n", app, app<1e-6?"OK":"FAIL");
    printf("[inv 4 output] decode out vs CPU ref  scale-norm %.2e : %s\n", outn, outn<1e-3?"OK":"FAIL");
    bool pass = i1 && pfx<1e-6 && app<1e-6 && outn<1e-3;
    printf("=== %s: DML MHA KV-cache decode step on HD4600 (past %u -> present %u), all 4 invariants ===\n", pass?"PASS":"FAIL", P, Tot);
    return pass?0:1;
}
