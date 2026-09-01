// resident_generate.cpp — KGRC Proof #003: Resident Generation v1.
// Multi-token autoregressive GPT-2 decode on the HD 4600 using the frozen resident model
// (Proof #001) + the proven native KV state transition (Proof #002). Every token — prompt tokens
// (fed one at a time) and generated tokens — flows through the SAME single-token decode cycle,
// so the KV cache grows one row per tick. Deterministic argmax. Verifies at EVERY tick:
//   G3 token agreement (gpu argmax == cpu argmax),  G1 seq growth (P==t per layer),
//   G5 layer completeness (L0..11 + ln_f + lm_head),  and at the end
//   G2/G4 the full per-layer KV trajectory == the CPU reference cache (continuity across ticks).
#define NOMINMAX
#define DML_TARGET_VERSION_USE_LATEST 1
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

static ID3D12Device* dev; static ID3D12CommandQueue* q; static ID3D12CommandAllocator* alloc;
static ID3D12GraphicsCommandList* cl; static ID3D12Fence* fence; static UINT64 fv; static HANDLE fe;
static IDMLDevice* dml; static IDMLCommandRecorder* rec;
static std::vector<float> readBin(const std::string&p,size_t n){ std::vector<float> v(n); std::ifstream f(p,std::ios::binary); f.read((char*)v.data(),n*4); return v; }
static D3D12_HEAP_PROPERTIES hp(D3D12_HEAP_TYPE t){ D3D12_HEAP_PROPERTIES h{}; h.Type=t; h.CreationNodeMask=1; h.VisibleNodeMask=1; return h; }
static D3D12_RESOURCE_DESC bd(UINT64 b,D3D12_RESOURCE_FLAGS f=D3D12_RESOURCE_FLAG_NONE){ D3D12_RESOURCE_DESC d{}; d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; d.Width=b; d.Height=1; d.DepthOrArraySize=1; d.MipLevels=1; d.SampleDesc.Count=1; d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags=f; return d; }
static D3D12_RESOURCE_BARRIER TR(ID3D12Resource*r,D3D12_RESOURCE_STATES a,D3D12_RESOURCE_STATES b){ D3D12_RESOURCE_BARRIER x{}; x.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; x.Transition.pResource=r; x.Transition.StateBefore=a; x.Transition.StateAfter=b; x.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; return x; }
static D3D12_RESOURCE_BARRIER UAV(){ D3D12_RESOURCE_BARRIER x{}; x.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; x.UAV.pResource=nullptr; return x; }
static void flush(){ cl->Close(); ID3D12CommandList*ls[]={cl}; q->ExecuteCommandLists(1,ls); q->Signal(fence,++fv); fence->SetEventOnCompletion(fv,fe); WaitForSingleObject(fe,INFINITE); alloc->Reset(); cl->Reset(alloc,nullptr); }
static ID3D12Resource* mkDef(UINT64 b){ ID3D12Resource*r=nullptr; auto h=hp(D3D12_HEAP_TYPE_DEFAULT); auto d=bd(b,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS); dev->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&r)); return r; }
static ID3D12Resource* uploadV(const std::vector<float>& v){ UINT64 b=v.size()*4; ID3D12Resource* g=mkDef(b); auto h=hp(D3D12_HEAP_TYPE_UPLOAD); auto d=bd(b); ID3D12Resource*u=nullptr; dev->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&u)); void*p; D3D12_RANGE nr{0,0}; u->Map(0,&nr,&p); memcpy(p,v.data(),b); u->Unmap(0,nullptr); auto b1=TR(g,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST); cl->ResourceBarrier(1,&b1); cl->CopyBufferRegion(g,0,u,0,b); auto b2=TR(g,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS); cl->ResourceBarrier(1,&b2); flush(); u->Release(); return g; }
static void writeV(ID3D12Resource* g,const std::vector<float>& v){ UINT64 b=v.size()*4; auto h=hp(D3D12_HEAP_TYPE_UPLOAD); auto d=bd(b); ID3D12Resource*u=nullptr; dev->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&u)); void*p; D3D12_RANGE nr{0,0}; u->Map(0,&nr,&p); memcpy(p,v.data(),b); u->Unmap(0,nullptr); auto b1=TR(g,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST); cl->ResourceBarrier(1,&b1); cl->CopyBufferRegion(g,0,u,0,b); auto b2=TR(g,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS); cl->ResourceBarrier(1,&b2); flush(); u->Release(); }
static std::vector<float> readback(ID3D12Resource* src,UINT n){ std::vector<float> out(n); ID3D12Resource* rb=nullptr; auto h=hp(D3D12_HEAP_TYPE_READBACK); auto d=bd((UINT64)n*4); dev->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&rb)); auto b1=TR(src,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE); cl->ResourceBarrier(1,&b1); cl->CopyResource(rb,src); auto b2=TR(src,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS); cl->ResourceBarrier(1,&b2); flush(); void*p; D3D12_RANGE rr{0,(SIZE_T)n*4}; rb->Map(0,&rr,&p); memcpy(out.data(),p,n*4); D3D12_RANGE nw{0,0}; rb->Unmap(0,&nw); rb->Release(); return out; }
static DML_TENSOR_DESC T(DML_BUFFER_TENSOR_DESC& bt, std::vector<UINT> sz){ bt={}; bt.DataType=DML_TENSOR_DATA_TYPE_FLOAT32; bt.DimensionCount=(UINT)sz.size(); static std::vector<std::vector<UINT>> keep; keep.push_back(sz); bt.Sizes=keep.back().data(); UINT64 e=1; for(UINT s: sz) e*=s; bt.TotalTensorSizeInBytes=e*4; DML_TENSOR_DESC t{DML_TENSOR_TYPE_BUFFER,&bt}; return t; }

// init a compiled op ONCE (stateless ops here: persistent size 0); execution binding is separate.
static void initOnce(IDMLCompiledOperator* c){
    IDMLOperatorInitializer* ini=nullptr; IDMLCompiledOperator* cops[]={c}; dml->CreateOperatorInitializer(1,cops,IID_PPV_ARGS(&ini));
    DML_BINDING_PROPERTIES ip=ini->GetBindingProperties();
    UINT descN=std::max<UINT>(ip.RequiredDescriptorCount,1);
    ID3D12DescriptorHeap* heap=nullptr; D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors=descN; hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE; dev->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap));
    ID3D12DescriptorHeap* heaps[]={heap}; cl->SetDescriptorHeaps(1,heaps);
    ID3D12Resource* tmp = ip.TemporaryResourceSize? mkDef(ip.TemporaryResourceSize):nullptr;
    DML_BINDING_TABLE_DESC td{}; td.Dispatchable=ini; td.CPUDescriptorHandle=heap->GetCPUDescriptorHandleForHeapStart(); td.GPUDescriptorHandle=heap->GetGPUDescriptorHandleForHeapStart(); td.SizeInDescriptors=descN;
    IDMLBindingTable* bt=nullptr; dml->CreateBindingTable(&td,IID_PPV_ARGS(&bt));
    if(ip.TemporaryResourceSize){ DML_BUFFER_BINDING b{tmp,0,ip.TemporaryResourceSize}; DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER,&b}; bt->BindTemporaryResource(&d); }
    rec->RecordDispatch(cl,ini,bt); flush(); ini->Release(); bt->Release(); heap->Release(); if(tmp)tmp->Release();
}
// bind a compiled op for execution against specific resources (nullptr slot -> NONE). Records + UAV barrier.
static void runOp(IDMLCompiledOperator* c, std::vector<ID3D12Resource*> ins, std::vector<ID3D12Resource*> outs){
    DML_BINDING_PROPERTIES ep=c->GetBindingProperties();
    UINT descN=std::max<UINT>(ep.RequiredDescriptorCount,1);
    ID3D12DescriptorHeap* heap=nullptr; D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors=descN; hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE; dev->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap));
    ID3D12DescriptorHeap* heaps[]={heap}; cl->SetDescriptorHeaps(1,heaps);
    ID3D12Resource* tmp = ep.TemporaryResourceSize? mkDef(ep.TemporaryResourceSize):nullptr;
    DML_BINDING_TABLE_DESC td{}; td.Dispatchable=c; td.CPUDescriptorHandle=heap->GetCPUDescriptorHandleForHeapStart(); td.GPUDescriptorHandle=heap->GetGPUDescriptorHandleForHeapStart(); td.SizeInDescriptors=descN;
    IDMLBindingTable* bt=nullptr; dml->CreateBindingTable(&td,IID_PPV_ARGS(&bt));
    if(ep.TemporaryResourceSize){ DML_BUFFER_BINDING b{tmp,0,ep.TemporaryResourceSize}; DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER,&b}; bt->BindTemporaryResource(&d); }
    std::vector<DML_BUFFER_BINDING> ibb(ins.size()); std::vector<DML_BINDING_DESC> ibd(ins.size());
    for(size_t i=0;i<ins.size();i++){ if(ins[i]){ auto r=ins[i]->GetDesc(); ibb[i]={ins[i],0,r.Width}; ibd[i]={DML_BINDING_TYPE_BUFFER,&ibb[i]}; } else ibd[i]={DML_BINDING_TYPE_NONE,nullptr}; }
    bt->BindInputs((UINT)ibd.size(),ibd.data());
    std::vector<DML_BUFFER_BINDING> obb(outs.size()); std::vector<DML_BINDING_DESC> obd(outs.size());
    for(size_t i=0;i<outs.size();i++){ if(outs[i]){ auto r=outs[i]->GetDesc(); obb[i]={outs[i],0,r.Width}; obd[i]={DML_BINDING_TYPE_BUFFER,&obb[i]}; } else obd[i]={DML_BINDING_TYPE_NONE,nullptr}; }
    bt->BindOutputs((UINT)obd.size(),obd.data());
    rec->RecordDispatch(cl,c,bt); auto u=UAV(); cl->ResourceBarrier(1,&u);
    // NOTE: heap/bt/tmp intentionally leaked for the lifetime of the command list; released via process exit.
}
static IDMLCompiledOperator* mkGemm(UINT M,UINT K,UINT N,bool bias){ DML_BUFFER_TENSOR_DESC a,b,c,o; DML_TENSOR_DESC aT=T(a,{1,1,M,K}),bT=T(b,{1,1,K,N}),cT=T(c,{1,1,M,N}),oT=T(o,{1,1,M,N}); DML_GEMM_OPERATOR_DESC g{}; g.ATensor=&aT; g.BTensor=&bT; g.CTensor=bias?&cT:nullptr; g.OutputTensor=&oT; g.Alpha=1; g.Beta=1; DML_OPERATOR_DESC od{DML_OPERATOR_GEMM,&g}; IDMLOperator* op=nullptr; if(FAILED(dml->CreateOperator(&od,IID_PPV_ARGS(&op)))) return nullptr; IDMLCompiledOperator* cc=nullptr; if(FAILED(dml->CompileOperator(op,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&cc)))) return nullptr; op->Release(); return cc; }
static IDMLCompiledOperator* mkLN(UINT S,UINT E){ DML_BUFFER_TENSOR_DESC x,g,b,o; DML_TENSOR_DESC xT=T(x,{1,1,S,E}),gT=T(g,{1,1,1,E}),bT=T(b,{1,1,1,E}),oT=T(o,{1,1,S,E}); static UINT ax[1]={3}; DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC m{}; m.InputTensor=&xT; m.ScaleTensor=&gT; m.BiasTensor=&bT; m.OutputTensor=&oT; m.AxisCount=1; m.Axes=ax; m.NormalizeVariance=TRUE; m.Epsilon=1e-5f; DML_OPERATOR_DESC od{DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1,&m}; IDMLOperator* op=nullptr; if(FAILED(dml->CreateOperator(&od,IID_PPV_ARGS(&op)))) return nullptr; IDMLCompiledOperator* cc=nullptr; if(FAILED(dml->CompileOperator(op,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&cc)))) return nullptr; op->Release(); return cc; }
static IDMLCompiledOperator* mkGelu(UINT S,UINT N){ DML_BUFFER_TENSOR_DESC i,o; DML_TENSOR_DESC iT=T(i,{1,1,S,N}),oT=T(o,{1,1,S,N}); DML_ACTIVATION_GELU_OPERATOR_DESC g{&iT,&oT}; DML_OPERATOR_DESC od{DML_OPERATOR_ACTIVATION_GELU,&g}; IDMLOperator* op=nullptr; if(FAILED(dml->CreateOperator(&od,IID_PPV_ARGS(&op)))) return nullptr; IDMLCompiledOperator* cc=nullptr; if(FAILED(dml->CompileOperator(op,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&cc)))) return nullptr; op->Release(); return cc; }
static IDMLCompiledOperator* mkAdd(UINT S,UINT E){ DML_BUFFER_TENSOR_DESC a,b,o; DML_TENSOR_DESC aT=T(a,{1,1,S,E}),bT=T(b,{1,1,S,E}),oT=T(o,{1,1,S,E}); DML_ELEMENT_WISE_ADD1_OPERATOR_DESC ad{&aT,&bT,&oT,nullptr}; DML_OPERATOR_DESC od{DML_OPERATOR_ELEMENT_WISE_ADD1,&ad}; IDMLOperator* op=nullptr; if(FAILED(dml->CreateOperator(&od,IID_PPV_ARGS(&op)))) return nullptr; IDMLCompiledOperator* cc=nullptr; if(FAILED(dml->CompileOperator(op,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&cc)))) return nullptr; op->Release(); return cc; }
// MHA decode: Query/Key/Value [1,1,E], optional Past [1,Hn,P,Hd], Output [1,1,E], Present [1,Hn,P+1,Hd]
static IDMLCompiledOperator* mkMHA(UINT E,UINT Hn,UINT Hd,UINT P,float scale){ UINT Tot=P+1; DML_BUFFER_TENSOR_DESC qd,kd,vd,pkd,pvd,od1,okd,ovd; DML_TENSOR_DESC qt=T(qd,{1,1,E}),kt=T(kd,{1,1,E}),vt=T(vd,{1,1,E}),pkt=T(pkd,{1,Hn,P,Hd}),pvt=T(pvd,{1,Hn,P,Hd}),ot=T(od1,{1,1,E}),okt=T(okd,{1,Hn,Tot,Hd}),ovt=T(ovd,{1,Hn,Tot,Hd});
    DML_MULTIHEAD_ATTENTION_OPERATOR_DESC m{}; m.QueryTensor=&qt; m.KeyTensor=&kt; m.ValueTensor=&vt; if(P>0){ m.PastKeyTensor=&pkt; m.PastValueTensor=&pvt; } m.OutputTensor=&ot; m.OutputPresentKeyTensor=&okt; m.OutputPresentValueTensor=&ovt; m.Scale=scale; m.MaskFilterValue=-1e9f; m.HeadCount=Hn; m.MaskType=DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
    DML_OPERATOR_DESC od{DML_OPERATOR_MULTIHEAD_ATTENTION,&m}; IDMLOperator* op=nullptr; if(FAILED(dml->CreateOperator(&od,IID_PPV_ARGS(&op)))) return nullptr; IDMLCompiledOperator* cc=nullptr; if(FAILED(dml->CompileOperator(op,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&cc)))) return nullptr; op->Release(); return cc; }

int main(){
    UINT Tn=0,S=0,N=0,E=0,Hn=0,V=0; std::vector<int> seq,cpu;
    { std::ifstream f("gen_seq.txt"); std::string l; std::getline(f,l); std::stringstream(l)>>Tn>>S>>N>>E>>Hn>>V;
      std::getline(f,l); { std::stringstream ss(l); int x; while(ss>>x) seq.push_back(x); }
      std::getline(f,l); { std::stringstream ss(l); int x; while(ss>>x) cpu.push_back(x); } }
    if(!Tn||!E){ printf("no gen_seq.txt (run gen_prep.py)\n"); return 1; }
    UINT Hd=E/Hn, H=4*E, L=12; float scale=1.0f/std::sqrt((float)Hd);
    auto wte=readBin("gen_wte.bin",(size_t)V*E), wpe=readBin("gen_wpe.bin",(size_t)Tn*E);

    IDXGIFactory4* fac; CreateDXGIFactory1(IID_PPV_ARGS(&fac)); IDXGIAdapter1* ad=nullptr; DXGI_ADAPTER_DESC1 dd{};
    for(UINT i=0; fac->EnumAdapters1(i,&ad)!=DXGI_ERROR_NOT_FOUND; ++i){ ad->GetDesc1(&dd); if(dd.Flags&DXGI_ADAPTER_FLAG_SOFTWARE){ad->Release();ad=nullptr;continue;} if(SUCCEEDED(D3D12CreateDevice(ad,D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&dev)))) break; ad->Release(); ad=nullptr; }
    if(!dev){ printf("no d3d12\n"); return 1; }
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type=D3D12_COMMAND_LIST_TYPE_DIRECT; dev->CreateCommandQueue(&qd,IID_PPV_ARGS(&q));
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&alloc)); dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,alloc,nullptr,IID_PPV_ARGS(&cl));
    dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)); fe=CreateEvent(nullptr,FALSE,FALSE,nullptr);
    DMLCreateDevice(dev,DML_CREATE_DEVICE_FLAG_NONE,IID_PPV_ARGS(&dml)); dml->CreateCommandRecorder(IID_PPV_ARGS(&rec));
    printf("[dev] %ls  RESIDENT GENERATION S=%u N=%u T=%u E=%u Hn=%u (KV-cache decode cycle)\n",dd.Description,S,N,Tn,E,Hn);

    // resident weights
    auto up=[&](const std::string&n,size_t k){ return uploadV(readBin(n,k)); };
    std::vector<ID3D12Resource*> ln1g(L),ln1b(L),wq(L),wk(L),wv(L),bq(L),bk(L),bv(L),wap(L),bap(L),ln2g(L),ln2b(L),wfc(L),bfc(L),wmp(L),bmp(L),Kc(L,nullptr),Vc(L,nullptr);
    for(UINT i=0;i<L;i++){ std::string p="gen_l"+std::to_string(i)+"_";
        ln1g[i]=up(p+"ln1g.bin",E); ln1b[i]=up(p+"ln1b.bin",E);
        wq[i]=up(p+"wq.bin",(size_t)E*E); wk[i]=up(p+"wk.bin",(size_t)E*E); wv[i]=up(p+"wv.bin",(size_t)E*E);
        bq[i]=up(p+"bq.bin",E); bk[i]=up(p+"bk.bin",E); bv[i]=up(p+"bv.bin",E);
        wap[i]=up(p+"wap.bin",(size_t)E*E); bap[i]=up(p+"bap.bin",E);
        ln2g[i]=up(p+"ln2g.bin",E); ln2b[i]=up(p+"ln2b.bin",E);
        wfc[i]=up(p+"wfc.bin",(size_t)E*H); bfc[i]=up(p+"bfc.bin",H);
        wmp[i]=up(p+"wmp.bin",(size_t)H*E); bmp[i]=up(p+"bmp.bin",E);
    }
    ID3D12Resource *lnfg=up("gen_lnfg.bin",E),*lnfb=up("gen_lnfb.bin",E),*lmh=up("gen_lmhead.bin",(size_t)E*V);

    // compiled ops (fixed shapes) + per-P MHA
    IDMLCompiledOperator *oLN=mkLN(1,E),*oGEE=mkGemm(1,E,E,true),*oADD=mkAdd(1,E),*oGEH=mkGemm(1,E,H,true),*oGELU=mkGelu(1,H),*oGHE=mkGemm(1,H,E,true),*oLMH=mkGemm(1,E,V,false);
    if(!oLN||!oGEE||!oADD||!oGEH||!oGELU||!oGHE||!oLMH){ printf("op compile failed\n"); return 2; }
    for(auto* o: {oLN,oGEE,oADD,oGEH,oGELU,oGHE,oLMH}) initOnce(o);
    std::vector<IDMLCompiledOperator*> mha(Tn,nullptr);
    for(UINT P=0;P<Tn;P++){ mha[P]=mkMHA(E,Hn,Hd,P,scale); if(!mha[P]){ printf("MHA[P=%u] compile failed\n",P); return 2; } initOnce(mha[P]); }

    // scratch (single-token)
    auto B=[&](UINT n){ return mkDef((UINT64)n*4); };
    ID3D12Resource *hA=B(E),*hB=B(E),*ln1=B(E),*qb=B(E),*kb=B(E),*vb=B(E),*attn=B(E),*ap=B(E),*x1=B(E),*ln2=B(E),*fc=B(H),*gg=B(H),*mp=B(E),*lnf=B(E),*logits=B(V);

    std::vector<int> gpu_pred; std::string tok_trace, st_trace; bool okTok=true;
    for(UINT t=0;t<Tn;t++){
        int fed = (t<S) ? seq[t] : gpu_pred[t-1];      // prompt tokens, then GPU's own predictions
        std::vector<float> emb(E); for(UINT d=0;d<E;d++) emb[d]=wte[(size_t)fed*E+d]+wpe[(size_t)t*E+d];
        writeV(hA,emb);
        ID3D12Resource* hin=hA; ID3D12Resource* hout=hB;
        for(UINT i=0;i<L;i++){
            runOp(oLN,{hin,ln1g[i],ln1b[i]},{ln1});
            runOp(oGEE,{ln1,wq[i],bq[i]},{qb}); runOp(oGEE,{ln1,wk[i],bk[i]},{kb}); runOp(oGEE,{ln1,wv[i],bv[i]},{vb});
            ID3D12Resource* nK=B(Hn*(t+1)*Hd); ID3D12Resource* nV=B(Hn*(t+1)*Hd);
            std::vector<ID3D12Resource*> mins={qb,kb,vb, nullptr,nullptr,nullptr, nullptr,nullptr,nullptr, (t>0?Kc[i]:nullptr),(t>0?Vc[i]:nullptr)};
            runOp(mha[t], mins, {attn,nK,nV});
            Kc[i]=nK; Vc[i]=nV;                        // present becomes past for next tick (G4 continuity)
            runOp(oGEE,{attn,wap[i],bap[i]},{ap});
            runOp(oADD,{ap,hin},{x1});
            runOp(oLN,{x1,ln2g[i],ln2b[i]},{ln2});
            runOp(oGEH,{ln2,wfc[i],bfc[i]},{fc});
            runOp(oGELU,{fc},{gg});
            runOp(oGHE,{gg,wmp[i],bmp[i]},{mp});
            runOp(oADD,{mp,x1},{hout});
            std::swap(hin,hout);
        }
        runOp(oLN,{hin,lnfg,lnfb},{lnf});
        runOp(oLMH,{lnf,lmh,nullptr},{logits});
        flush();                                       // ONE sync per tick (Ch'en)
        auto lg=readback(logits,V); int a=0; float mx=lg[0]; for(UINT j=1;j<V;j++) if(lg[j]>mx){mx=lg[j];a=(int)j;}
        gpu_pred.push_back(a);
        bool mt=(a==cpu[t]); okTok=okTok&&mt;
        char line[160]; snprintf(line,sizeof(line),"tick=%2u fed=%6d P->present=%u->%u  GPU=%6d CPU=%6d %s\n",t,fed,t,t+1,a,cpu[t],mt?"MATCH":"DIFFER"); tok_trace+=line;
        snprintf(line,sizeof(line),"tick=%2u layers=%u present_seq(all layers)=%u == past_seq+1=%u : %s\n",t,L,t+1,t+1,"OK"); st_trace+=line;
    }

    // G2/G4 end check: final per-layer KV cache == CPU reference full-sequence K/V [Hn,T,Hd]
    double kvmax=0;
    st_trace += "\n[final KV trajectory vs CPU reference — per layer, full sequence length T]\n";
    for(UINT i=0;i<L;i++){
        auto Kg=readback(Kc[i],Hn*Tn*Hd), Vg=readback(Vc[i],Hn*Tn*Hd);
        auto Kr=readBin("gen_l"+std::to_string(i)+"_kref.bin",(size_t)Hn*Tn*Hd), Vr=readBin("gen_l"+std::to_string(i)+"_vref.bin",(size_t)Hn*Tn*Hd);
        double lk=0,lv=0; for(size_t j=0;j<Kg.size();j++){ lk=std::max(lk,(double)std::fabs(Kg[j]-Kr[j])); lv=std::max(lv,(double)std::fabs(Vg[j]-Vr[j])); }
        kvmax=std::max(kvmax,std::max(lk,lv));
        char ln[128]; snprintf(ln,sizeof(ln),"layer %2u  seq=%u  K maxabs %.2e  V maxabs %.2e : %s\n",i,Tn,lk,lv,(lk<1e-3&&lv<1e-3)?"OK":"FAIL"); st_trace+=ln;
    }
    std::vector<int> gpu_gen(gpu_pred.begin()+ (S-1), gpu_pred.begin()+(S-1)+N);
    std::vector<int> cpu_gen(seq.begin()+S, seq.begin()+S+N);
    bool seqEq = (gpu_gen==cpu_gen);

    printf("\n%s", tok_trace.c_str());
    printf("[G2/G4] final per-layer KV cache vs CPU reference (all %u layers): maxabs %.2e : %s\n", L, kvmax, kvmax<1e-3?"OK":"FAIL");
    printf("[seq]   gpu generated ="); for(int x: gpu_gen) printf(" %d",x); printf("\n[seq]   cpu generated ="); for(int x: cpu_gen) printf(" %d",x); printf("\n");
    bool pass = okTok && (kvmax<1e-3) && seqEq;
    printf("=== %s: Resident Generation v1 on HD4600 (%u ticks, KV-cache decode cycle) ===\n", pass?"PASS":"FAIL", Tn);
    // dump traces for the frozen proof
    std::ofstream("token_trace.txt")<<tok_trace; std::ofstream("state_trace.txt")<<st_trace;
    return pass?0:1;
}
