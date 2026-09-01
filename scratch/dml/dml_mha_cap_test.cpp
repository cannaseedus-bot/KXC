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

    // #004-B1 capacity semantics: PHYSICAL past capacity C > logical extent P. Fill P real past
    // entries + (C-P) garbage; a validity mask (RelativePositionBias -inf on the unused slots)
    // must make the output identical to an exact-extent-P attention. Question: extent<capacity
    // => same output?  (native MHA1/PastSequenceLengths is unavailable on this device — probed.)
    const UINT Hn=2, Hd=3, E=Hn*Hd, C=6, P=3, Tot=C+1;   // capacity C, extent P, keys = C past + 1 current
    const float scale=1.0f/std::sqrt((float)Hd);
    std::vector<float> Q(E),Kn(E),Vn(E),pastK(Hn*C*Hd),pastV(Hn*C*Hd);
    for(UINT i=0;i<E;i++){ Q[i]=std::sin(0.3f*i)*0.7f; Kn[i]=std::cos(0.2f*i)*0.6f; Vn[i]=std::sin(0.17f*i+1)*0.5f; }
    // physical past: first P per head are real, slots [P..C) are GARBAGE (must be masked out)
    for(UINT h=0;h<Hn;h++) for(UINT t=0;t<C;t++) for(UINT d=0;d<Hd;d++){
        float real = t<P; pastK[(h*C+t)*Hd+d] = real? std::sin(0.11f*((h*P+t)*Hd+d))*0.5f : 999.0f;
        pastV[(h*C+t)*Hd+d] = real? std::cos(0.13f*((h*P+t)*Hd+d))*0.4f : -999.0f; }
    // validity mask [1,Hn,1,C+1]: 0 for real past (0..P-1) and current (C); -1e9 for garbage (P..C-1)
    std::vector<float> rpb(Hn*(C+1)); for(UINT h=0;h<Hn;h++) for(UINT k=0;k<=C;k++) rpb[h*(C+1)+k] = (k<P || k==C)? 0.0f : -1e9f;

    // base MHA (available), capacity past [1,Hn,C,Hd] + mask [1,Hn,1,C+1], present [1,Hn,C+1,Hd]
    DML_BUFFER_TENSOR_DESC qb,kb,vb,pkb,pvb,rbb,ob,okb,ovb;
    DML_TENSOR_DESC qt=Td(qb,{1,1,E}),kt=Td(kb,{1,1,E}),vt=Td(vb,{1,1,E}),pkt=Td(pkb,{1,Hn,C,Hd}),pvt=Td(pvb,{1,Hn,C,Hd}),rbt=Td(rbb,{1,Hn,1,C+1}),ot=Td(ob,{1,1,E}),okt=Td(okb,{1,Hn,Tot,Hd}),ovt=Td(ovb,{1,Hn,Tot,Hd});
    DML_MULTIHEAD_ATTENTION_OPERATOR_DESC m{}; m.QueryTensor=&qt; m.KeyTensor=&kt; m.ValueTensor=&vt; m.PastKeyTensor=&pkt; m.PastValueTensor=&pvt; m.RelativePositionBiasTensor=&rbt; m.OutputTensor=&ot; m.OutputPresentKeyTensor=&okt; m.OutputPresentValueTensor=&ovt;
    m.Scale=scale; m.MaskFilterValue=-1e9f; m.HeadCount=Hn; m.MaskType=DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
    DML_OPERATOR_DESC od{DML_OPERATOR_MULTIHEAD_ATTENTION,&m};
    IDMLOperator* o=nullptr; if(FAILED(dml->CreateOperator(&od,IID_PPV_ARGS(&o)))){ printf("cap MHA create FAILED\n"); return 2; }
    IDMLCompiledOperator* c=nullptr; if(FAILED(dml->CompileOperator(o,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&c)))){ printf("cap MHA compile FAILED\n"); return 2; } o->Release();

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

    ID3D12Resource *qR=upload(Q),*kR=upload(Kn),*vR=upload(Vn),*pkR=upload(pastK),*pvR=upload(pastV),*rpbR=upload(rpb);
    ID3D12Resource *outR=mkDef((UINT64)E*4),*okR=mkDef((UINT64)Hn*Tot*Hd*4),*ovR=mkDef((UINT64)Hn*Tot*Hd*4);
    td.Dispatchable=c; bt->Reset(&td);
    if(tmpB){ DML_BUFFER_BINDING b{tmp,0,tmpB}; DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER,&b}; bt->BindTemporaryResource(&d); }
    ID3D12Resource* inSlots[11]={qR,kR,vR, nullptr,nullptr,nullptr, nullptr,nullptr,rpbR, pkR,pvR};   // rpb mask @ slot 8
    DML_BUFFER_BINDING ibb[11]; DML_BINDING_DESC ibd[11];
    for(int i=0;i<11;i++){ if(inSlots[i]){ auto r=inSlots[i]->GetDesc(); ibb[i]={inSlots[i],0,r.Width}; ibd[i]={DML_BINDING_TYPE_BUFFER,&ibb[i]}; } else ibd[i]={DML_BINDING_TYPE_NONE,nullptr}; }
    bt->BindInputs(11,ibd);
    ID3D12Resource* outSlots[3]={outR,okR,ovR}; DML_BUFFER_BINDING obb[3]; DML_BINDING_DESC obd[3];
    for(int i=0;i<3;i++){ auto r=outSlots[i]->GetDesc(); obb[i]={outSlots[i],0,r.Width}; obd[i]={DML_BINDING_TYPE_BUFFER,&obb[i]}; }
    bt->BindOutputs(3,obd);
    cl->SetDescriptorHeaps(1,heaps); rec->RecordDispatch(cl,c,bt); flush();

    auto out=readback(outR,E); auto presK=readback(okR,Hn*Tot*Hd); auto presV=readback(ovR,Hn*Tot*Hd);

    // EXACT-EXTENT reference: query attends to the P REAL past + current only (garbage ignored).
    auto qh=[&](UINT h,UINT d){ return Q[h*Hd+d]; };
    auto pkR_=[&](UINT h,UINT t,UINT d){ return pastK[(h*C+t)*Hd+d]; };   // physical stride C; t<P real
    auto pvR_=[&](UINT h,UINT t,UINT d){ return pastV[(h*C+t)*Hd+d]; };
    auto nk=[&](UINT h,UINT d){ return Kn[h*Hd+d]; };
    auto nv=[&](UINT h,UINT d){ return Vn[h*Hd+d]; };
    std::vector<float> ref(E);
    for(UINT h=0;h<Hn;h++){
        std::vector<float> sc(P+1); float mx=-1e30f;
        for(UINT j=0;j<P;j++){ float s=0; for(UINT d=0;d<Hd;d++) s+=qh(h,d)*pkR_(h,j,d); s*=scale; sc[j]=s; mx=std::max(mx,s); }
        { float s=0; for(UINT d=0;d<Hd;d++) s+=qh(h,d)*nk(h,d); s*=scale; sc[P]=s; mx=std::max(mx,s); }
        float sum=0; for(UINT j=0;j<=P;j++){ sc[j]=std::exp(sc[j]-mx); sum+=sc[j]; }
        for(UINT d=0;d<Hd;d++){ float acc=0; for(UINT j=0;j<P;j++) acc+=sc[j]/sum*pvR_(h,j,d); acc+=sc[P]/sum*nv(h,d); ref[h*Hd+d]=acc; }
    }
    // present layout [Hn,Tot=C+1,Hd]: present[0:C]=physical past, present[C]=current (appended at end C, not P)
    auto PK=[&](UINT h,UINT t,UINT d){ return presK[(h*Tot+t)*Hd+d]; };

    // (B1) capacity-output semantics: extent<capacity produces the SAME output as exact-extent P
    double oe=0,om=0; for(UINT i=0;i<E;i++){ oe=std::max(oe,(double)std::fabs(out[i]-ref[i])); om=std::max(om,(double)std::fabs(ref[i])); }
    double outn=oe/om;
    // prefix preserved (real past) + append at PHYSICAL end C (confirms base MHA does NOT compact to extent)
    double pfx=0; for(UINT h=0;h<Hn;h++) for(UINT t=0;t<P;t++) for(UINT d=0;d<Hd;d++) pfx=std::max(pfx,(double)std::fabs(PK(h,t,d)-pkR_(h,t,d)));
    double appEnd=0; for(UINT h=0;h<Hn;h++) for(UINT d=0;d<Hd;d++) appEnd=std::max(appEnd,(double)std::fabs(PK(h,C,d)-nk(h,d)));

    printf("[B1 output ] capacity C=%u extent P=%u : out vs exact-extent ref  scale-norm %.2e : %s\n", C,P,outn, outn<1e-3?"OK":"FAIL");
    printf("[   prefix ] present[0:P] == real past  maxabs %.2e : %s\n", pfx, pfx<1e-6?"OK":"FAIL");
    printf("[   append ] present appended at PHYSICAL end (slot C=%u), NOT extent P=%u  maxabs %.2e : %s (base MHA does not compact -> blocks fixed-capacity feedback)\n", C,P,appEnd, appEnd<1e-6?"confirmed":"?");
    bool pass = outn<1e-3 && pfx<1e-6;
    printf("=== %s: #004-B1 capacity-output semantics (extent<capacity -> same output via base MHA + validity mask) ===\n", pass?"PASS":"FAIL");
    return pass?0:1;
}
