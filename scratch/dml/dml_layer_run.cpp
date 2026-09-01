// dml_layer_run.cpp — a FULL gpt2 transformer layer on the HD 4600, activations RESIDENT.
// Assembles the attention block + MLP block: 12 DML ops chained into ONE command list with
// intermediate GPU buffers and ONE flush (only the final output is read back).
//   ln1=LN(x); Q,K,V=ln1@Wq/k/v+b; attn=causalMHA(Q,K,V); ap=attn@Wap+bap; x1=ap+x
//   ln2=LN(x1); fc=ln2@Wfc+bfc; g=gelu(fc); mp=g@Wmp+bmp; out=mp+x1
// Verifies vs the numpy reference from layer_prep.py.
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
static D3D12_RESOURCE_BARRIER trans(ID3D12Resource*r,D3D12_RESOURCE_STATES a,D3D12_RESOURCE_STATES b){ D3D12_RESOURCE_BARRIER x{}; x.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; x.Transition.pResource=r; x.Transition.StateBefore=a; x.Transition.StateAfter=b; x.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; return x; }
static D3D12_RESOURCE_BARRIER uavbar(){ D3D12_RESOURCE_BARRIER x{}; x.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; x.UAV.pResource=nullptr; return x; }
static void flush(){ cl->Close(); ID3D12CommandList*ls[]={cl}; q->ExecuteCommandLists(1,ls); q->Signal(fence,++fv); fence->SetEventOnCompletion(fv,fe); WaitForSingleObject(fe,INFINITE); alloc->Reset(); cl->Reset(alloc,nullptr); }
static ID3D12Resource* mkDef(UINT64 b){ ID3D12Resource*r=nullptr; auto h=hp(D3D12_HEAP_TYPE_DEFAULT); auto d=bd(b,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS); dev->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&r)); return r; }
static ID3D12Resource* upload(const std::vector<float>& v){ UINT64 b=v.size()*4; ID3D12Resource* g=mkDef(b); auto h=hp(D3D12_HEAP_TYPE_UPLOAD); auto d=bd(b); ID3D12Resource*u=nullptr; dev->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&u)); void*p; D3D12_RANGE nr{0,0}; u->Map(0,&nr,&p); memcpy(p,v.data(),b); u->Unmap(0,nullptr); auto b1=trans(g,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST); cl->ResourceBarrier(1,&b1); cl->CopyBufferRegion(g,0,u,0,b); auto b2=trans(g,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS); cl->ResourceBarrier(1,&b2); flush(); u->Release(); return g; }

static DML_TENSOR_DESC T(DML_BUFFER_TENSOR_DESC& bt, std::vector<UINT> sz){ bt={}; bt.DataType=DML_TENSOR_DATA_TYPE_FLOAT32; bt.DimensionCount=(UINT)sz.size(); static std::vector<std::vector<UINT>> keep; keep.push_back(sz); bt.Sizes=keep.back().data(); UINT64 e=1; for(UINT s: sz) e*=s; bt.TotalTensorSizeInBytes=e*4; DML_TENSOR_DESC t{DML_TENSOR_TYPE_BUFFER,&bt}; return t; }

struct Op { IDMLCompiledOperator* c; ID3D12DescriptorHeap* heap; IDMLBindingTable* bt; UINT descN; ID3D12Resource* tmp; UINT64 tmpB; };
static Op setupOp(IDMLCompiledOperator* c, std::vector<ID3D12Resource*> ins, std::vector<ID3D12Resource*> outs){
    Op o{}; o.c=c;
    IDMLOperatorInitializer* ini=nullptr; IDMLCompiledOperator* cops[]={c}; dml->CreateOperatorInitializer(1,cops,IID_PPV_ARGS(&ini));
    DML_BINDING_PROPERTIES ip=ini->GetBindingProperties(), ep=c->GetBindingProperties();
    o.descN=std::max(ip.RequiredDescriptorCount,ep.RequiredDescriptorCount); o.tmpB=std::max(ip.TemporaryResourceSize,ep.TemporaryResourceSize);
    D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors=o.descN; hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE; dev->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&o.heap));
    o.tmp=o.tmpB?mkDef(o.tmpB):nullptr;
    ID3D12DescriptorHeap* heaps[]={o.heap}; cl->SetDescriptorHeaps(1,heaps);
    DML_BINDING_TABLE_DESC td{}; td.Dispatchable=ini; td.CPUDescriptorHandle=o.heap->GetCPUDescriptorHandleForHeapStart(); td.GPUDescriptorHandle=o.heap->GetGPUDescriptorHandleForHeapStart(); td.SizeInDescriptors=o.descN; dml->CreateBindingTable(&td,IID_PPV_ARGS(&o.bt));
    if(o.tmpB && ip.TemporaryResourceSize){ DML_BUFFER_BINDING b{o.tmp,0,o.tmpB}; DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER,&b}; o.bt->BindTemporaryResource(&d); }
    rec->RecordDispatch(cl,ini,o.bt); flush(); ini->Release();
    td.Dispatchable=c; o.bt->Reset(&td);
    if(o.tmpB){ DML_BUFFER_BINDING b{o.tmp,0,o.tmpB}; DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER,&b}; o.bt->BindTemporaryResource(&d); }
    std::vector<DML_BUFFER_BINDING> ibb(ins.size()); std::vector<DML_BINDING_DESC> ibd(ins.size());
    for(size_t i=0;i<ins.size();i++){ if(ins[i]){ auto rd=ins[i]->GetDesc(); ibb[i]={ins[i],0,rd.Width}; ibd[i]={DML_BINDING_TYPE_BUFFER,&ibb[i]}; } else ibd[i]={DML_BINDING_TYPE_NONE,nullptr}; }
    o.bt->BindInputs((UINT)ibd.size(), ibd.data());
    std::vector<DML_BUFFER_BINDING> obb(outs.size()); std::vector<DML_BINDING_DESC> obd(outs.size());
    for(size_t i=0;i<outs.size();i++){ if(outs[i]){ auto rd=outs[i]->GetDesc(); obb[i]={outs[i],0,rd.Width}; obd[i]={DML_BINDING_TYPE_BUFFER,&obb[i]}; } else obd[i]={DML_BINDING_TYPE_NONE,nullptr}; }
    o.bt->BindOutputs((UINT)obd.size(), obd.data());
    return o;
}
static void record(Op& o){ ID3D12DescriptorHeap* heaps[]={o.heap}; cl->SetDescriptorHeaps(1,heaps); rec->RecordDispatch(cl,o.c,o.bt); auto u=uavbar(); cl->ResourceBarrier(1,&u); }

// GEMM+C: out[M,N] = A[M,K] @ B[K,N] + C[M,N]  (DML desc is copied at CreateOperator, so locals are fine)
static IDMLCompiledOperator* compileGemm(UINT M,UINT K,UINT N){
    DML_BUFFER_TENSOR_DESC a,b,c,o; DML_TENSOR_DESC aT=T(a,{1,1,M,K}),bT=T(b,{1,1,K,N}),cT=T(c,{1,1,M,N}),oT=T(o,{1,1,M,N});
    DML_GEMM_OPERATOR_DESC g{}; g.ATensor=&aT; g.BTensor=&bT; g.CTensor=&cT; g.OutputTensor=&oT; g.TransA=DML_MATRIX_TRANSFORM_NONE; g.TransB=DML_MATRIX_TRANSFORM_NONE; g.Alpha=1; g.Beta=1;
    DML_OPERATOR_DESC od{DML_OPERATOR_GEMM,&g}; IDMLOperator* op=nullptr; if(FAILED(dml->CreateOperator(&od,IID_PPV_ARGS(&op)))) return nullptr;
    IDMLCompiledOperator* cc=nullptr; if(FAILED(dml->CompileOperator(op,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&cc)))) return nullptr; op->Release(); return cc;
}
static IDMLCompiledOperator* compileLN(UINT S,UINT E){
    DML_BUFFER_TENSOR_DESC x,g,b,o; DML_TENSOR_DESC xT=T(x,{1,1,S,E}),gT=T(g,{1,1,1,E}),bT=T(b,{1,1,1,E}),oT=T(o,{1,1,S,E});
    static UINT axes[1]={3}; DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC m{}; m.InputTensor=&xT; m.ScaleTensor=&gT; m.BiasTensor=&bT; m.OutputTensor=&oT; m.AxisCount=1; m.Axes=axes; m.NormalizeVariance=TRUE; m.Epsilon=1e-5f;
    DML_OPERATOR_DESC od{DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1,&m}; IDMLOperator* op=nullptr; if(FAILED(dml->CreateOperator(&od,IID_PPV_ARGS(&op)))) return nullptr;
    IDMLCompiledOperator* cc=nullptr; if(FAILED(dml->CompileOperator(op,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&cc)))) return nullptr; op->Release(); return cc;
}
static IDMLCompiledOperator* compileGelu(UINT S,UINT N){
    DML_BUFFER_TENSOR_DESC i,o; DML_TENSOR_DESC iT=T(i,{1,1,S,N}),oT=T(o,{1,1,S,N});
    DML_ACTIVATION_GELU_OPERATOR_DESC g{&iT,&oT}; DML_OPERATOR_DESC od{DML_OPERATOR_ACTIVATION_GELU,&g}; IDMLOperator* op=nullptr; if(FAILED(dml->CreateOperator(&od,IID_PPV_ARGS(&op)))) return nullptr;
    IDMLCompiledOperator* cc=nullptr; if(FAILED(dml->CompileOperator(op,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&cc)))) return nullptr; op->Release(); return cc;
}
static IDMLCompiledOperator* compileAdd(UINT S,UINT E){
    DML_BUFFER_TENSOR_DESC a,b,o; DML_TENSOR_DESC aT=T(a,{1,1,S,E}),bT=T(b,{1,1,S,E}),oT=T(o,{1,1,S,E});
    DML_ELEMENT_WISE_ADD1_OPERATOR_DESC ad{&aT,&bT,&oT,nullptr}; DML_OPERATOR_DESC od{DML_OPERATOR_ELEMENT_WISE_ADD1,&ad}; IDMLOperator* op=nullptr; if(FAILED(dml->CreateOperator(&od,IID_PPV_ARGS(&op)))) return nullptr;
    IDMLCompiledOperator* cc=nullptr; if(FAILED(dml->CompileOperator(op,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&cc)))) return nullptr; op->Release(); return cc;
}
static IDMLCompiledOperator* compileMHA(UINT S,UINT E,UINT Hn,float scale){
    DML_BUFFER_TENSOR_DESC qd,kd,vd,rd,od2; DML_TENSOR_DESC qt=T(qd,{1,S,E}),kt=T(kd,{1,S,E}),vt=T(vd,{1,S,E}),rt=T(rd,{1,Hn,S,S}),ot=T(od2,{1,S,E});
    DML_MULTIHEAD_ATTENTION_OPERATOR_DESC m{}; m.QueryTensor=&qt; m.KeyTensor=&kt; m.ValueTensor=&vt; m.RelativePositionBiasTensor=&rt; m.OutputTensor=&ot;
    m.Scale=scale; m.MaskFilterValue=-1e9f; m.HeadCount=Hn; m.MaskType=DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
    DML_OPERATOR_DESC od{DML_OPERATOR_MULTIHEAD_ATTENTION,&m}; IDMLOperator* op=nullptr; if(FAILED(dml->CreateOperator(&od,IID_PPV_ARGS(&op)))) return nullptr;
    IDMLCompiledOperator* cc=nullptr; if(FAILED(dml->CompileOperator(op,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&cc)))) return nullptr; op->Release(); return cc;
}

int main(){
    UINT S=0,E=0,Hn=0,H=0; { std::ifstream f("ly_dims.txt"); std::stringstream d; d<<f.rdbuf(); d>>S>>E>>Hn>>H; }
    if(!S||!E||!Hn||!H){ printf("no dims (run from scratch/dml after layer_prep.py)\n"); return 1; }
    UINT Hd=E/Hn; float scale=1.0f/std::sqrt((float)Hd);
    auto rd=[&](const char*n,size_t k){ return readBin(n,k); };
    auto x=rd("ly_x.bin",(size_t)S*E), l1g=rd("ly_ln1g.bin",E), l1b=rd("ly_ln1b.bin",E), l2g=rd("ly_ln2g.bin",E), l2b=rd("ly_ln2b.bin",E);
    auto wq=rd("ly_wq.bin",(size_t)E*E), wk=rd("ly_wk.bin",(size_t)E*E), wv=rd("ly_wv.bin",(size_t)E*E), wap=rd("ly_wap.bin",(size_t)E*E), wfc=rd("ly_wfc.bin",(size_t)E*H), wmp=rd("ly_wmp.bin",(size_t)H*E);
    auto bq=rd("ly_bq.bin",(size_t)S*E), bk=rd("ly_bk.bin",(size_t)S*E), bv=rd("ly_bv.bin",(size_t)S*E), bap=rd("ly_bap.bin",(size_t)S*E), bfc=rd("ly_bfc.bin",(size_t)S*H), bmp=rd("ly_bmp.bin",(size_t)S*E);
    auto ref=rd("ly_ref.bin",(size_t)S*E);
    std::vector<float> rpb((size_t)Hn*S*S); for(UINT h=0;h<Hn;h++) for(UINT i=0;i<S;i++) for(UINT j=0;j<S;j++) rpb[(h*S+i)*S+j]=(j<=i)?0.0f:-1e9f;

    IDXGIFactory4* fac; CreateDXGIFactory1(IID_PPV_ARGS(&fac)); IDXGIAdapter1* ad=nullptr; DXGI_ADAPTER_DESC1 dd{};
    for(UINT i=0; fac->EnumAdapters1(i,&ad)!=DXGI_ERROR_NOT_FOUND; ++i){ ad->GetDesc1(&dd); if(dd.Flags&DXGI_ADAPTER_FLAG_SOFTWARE){ad->Release();ad=nullptr;continue;} if(SUCCEEDED(D3D12CreateDevice(ad,D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&dev)))) break; ad->Release(); ad=nullptr; }
    if(!dev){ printf("no d3d12\n"); return 1; }
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type=D3D12_COMMAND_LIST_TYPE_DIRECT; dev->CreateCommandQueue(&qd,IID_PPV_ARGS(&q));
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&alloc)); dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,alloc,nullptr,IID_PPV_ARGS(&cl));
    dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)); fe=CreateEvent(nullptr,FALSE,FALSE,nullptr);
    DMLCreateDevice(dev,DML_CREATE_DEVICE_FLAG_NONE,IID_PPV_ARGS(&dml)); dml->CreateCommandRecorder(IID_PPV_ARGS(&rec));
    printf("[dev] %ls  FULL LAYER S=%u E=%u Hn=%u H=%u (12 ops, one command list, activations resident)\n",dd.Description,S,E,Hn,H);

    ID3D12Resource *bx=upload(x),*bl1g=upload(l1g),*bl1b=upload(l1b),*bl2g=upload(l2g),*bl2b=upload(l2b);
    ID3D12Resource *bwq=upload(wq),*bwk=upload(wk),*bwv=upload(wv),*bwap=upload(wap),*bwfc=upload(wfc),*bwmp=upload(wmp);
    ID3D12Resource *bbq=upload(bq),*bbk=upload(bk),*bbv=upload(bv),*bbap=upload(bap),*bbfc=upload(bfc),*bbmp=upload(bmp),*brpb=upload(rpb);
    auto B=[&](UINT n){ return mkDef((UINT64)n*4); };
    ID3D12Resource *ln1=B(S*E),*qb=B(S*E),*kb=B(S*E),*vb=B(S*E),*attn=B(S*E),*ap=B(S*E),*x1=B(S*E),*ln2=B(S*E),*fc=B(S*H),*gg=B(S*H),*mp=B(S*E),*outb=B(S*E);

    IDMLCompiledOperator *cL1=compileLN(S,E),*cQ=compileGemm(S,E,E),*cK=compileGemm(S,E,E),*cV=compileGemm(S,E,E),*cM=compileMHA(S,E,Hn,scale),*cAp=compileGemm(S,E,E),*cA1=compileAdd(S,E),
        *cL2=compileLN(S,E),*cFc=compileGemm(S,E,H),*cGe=compileGelu(S,H),*cMp=compileGemm(S,H,E),*cA2=compileAdd(S,E);
    if(!cL1||!cQ||!cK||!cV||!cM||!cAp||!cA1||!cL2||!cFc||!cGe||!cMp||!cA2){ printf("a compile failed\n"); return 2; }

    // attention block
    Op o1=setupOp(cL1,{bx,bl1g,bl1b},{ln1});
    Op o2=setupOp(cQ,{ln1,bwq,bbq},{qb});
    Op o3=setupOp(cK,{ln1,bwk,bbk},{kb});
    Op o4=setupOp(cV,{ln1,bwv,bbv},{vb});
    Op o5=setupOp(cM,{qb,kb,vb, nullptr,nullptr,nullptr, nullptr,nullptr, brpb, nullptr,nullptr},{attn,nullptr,nullptr});
    Op o6=setupOp(cAp,{attn,bwap,bbap},{ap});
    Op o7=setupOp(cA1,{ap,bx},{x1});
    // MLP block
    Op o8=setupOp(cL2,{x1,bl2g,bl2b},{ln2});
    Op o9=setupOp(cFc,{ln2,bwfc,bbfc},{fc});
    Op o10=setupOp(cGe,{fc},{gg});
    Op o11=setupOp(cMp,{gg,bwmp,bbmp},{mp});
    Op o12=setupOp(cA2,{mp,x1},{outb});

    Op* ops[]={&o1,&o2,&o3,&o4,&o5,&o6,&o7,&o8,&o9,&o10,&o11,&o12};
    for(Op* o: ops) record(*o);
    auto b1=trans(outb,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE); cl->ResourceBarrier(1,&b1);
    ID3D12Resource* rbk=nullptr; auto rh=hp(D3D12_HEAP_TYPE_READBACK); auto rdz=bd((UINT64)S*E*4); dev->CreateCommittedResource(&rh,D3D12_HEAP_FLAG_NONE,&rdz,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&rbk));
    cl->CopyResource(rbk,outb); flush();
    std::vector<float> out(S*E); void*p; D3D12_RANGE rr{0,(SIZE_T)S*E*4}; rbk->Map(0,&rr,&p); memcpy(out.data(),p,S*E*4); D3D12_RANGE nw{0,0}; rbk->Unmap(0,&nw);

    double maxAbs=0,absmax=0; for(size_t i=0;i<(size_t)S*E;i++){ maxAbs=std::max(maxAbs,(double)std::fabs(out[i]-ref[i])); absmax=std::max(absmax,(double)std::fabs(ref[i])); }
    double nrm=maxAbs/absmax;
    printf("[out[0,0..2]] gpu=(%.5f,%.5f,%.5f) ref=(%.5f,%.5f,%.5f)\n",out[0],out[1],out[2],ref[0],ref[1],ref[2]);
    printf("[verify] max abs %.3e  scale %.3f  scale-norm %.2e\n",maxAbs,absmax,nrm);
    bool pass=nrm<1e-3;
    printf("=== %s: FULL gpt2 layer (12 ops, one command list, activations resident) on HD4600 vs numpy ===\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
