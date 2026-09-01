// dml_attn_run.cpp — fused gpt2 ATTENTION block on the HD 4600, activations RESIDENT on-device.
// Chains 7 de-risked DML ops into ONE command list with intermediate GPU buffers and ONE flush:
//   ln = LayerNorm(x)                 [MVN1]
//   Q = ln@Wq+bq, K = ln@Wk+bk, V = ln@Wv+bv   [3x GEMM with C]
//   attn = causal_MHA(Q,K,V)          [MULTIHEAD_ATTENTION, separate Q/K/V + causal rel-pos bias]
//   proj = attn@Wp+bp                 [GEMM with C]
//   out = proj + x                    [ELEMENT_WISE_ADD1]
// Verifies vs the numpy reference from attn_prep.py.
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

// ins/outs: nullptr entries -> DML_BINDING_TYPE_NONE (positional optional slots).
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
static IDMLCompiledOperator* compile(const DML_OPERATOR_DESC& od,const char* name){ IDMLOperator* o=nullptr; if(FAILED(dml->CreateOperator(&od,IID_PPV_ARGS(&o)))){ printf("%s create FAILED\n",name); return nullptr; } IDMLCompiledOperator* c=nullptr; if(FAILED(dml->CompileOperator(o,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&c)))){ printf("%s compile FAILED\n",name); return nullptr; } o->Release(); return c; }

int main(){
    UINT S=0,E=0,Hn=0; { std::ifstream f("attn_dims.txt"); std::stringstream d; d<<f.rdbuf(); d>>S>>E>>Hn; }
    if(!S||!E||!Hn){ printf("no dims (run from scratch/dml after attn_prep.py)\n"); return 1; }
    UINT Hd=E/Hn; float scale=1.0f/std::sqrt((float)Hd);
    auto x=readBin("attn_x.bin",(size_t)S*E), lng=readBin("attn_lng.bin",E), lnb=readBin("attn_lnb.bin",E);
    auto wq=readBin("attn_wq.bin",(size_t)E*E), wk=readBin("attn_wk.bin",(size_t)E*E), wv=readBin("attn_wv.bin",(size_t)E*E), wp=readBin("attn_wp.bin",(size_t)E*E);
    auto bq=readBin("attn_bq.bin",(size_t)S*E), bk=readBin("attn_bk.bin",(size_t)S*E), bv=readBin("attn_bv.bin",(size_t)S*E), bp=readBin("attn_bp.bin",(size_t)S*E);
    auto ref=readBin("attn_ref.bin",(size_t)S*E);
    std::vector<float> rpb((size_t)Hn*S*S); for(UINT h=0;h<Hn;h++) for(UINT i=0;i<S;i++) for(UINT j=0;j<S;j++) rpb[(h*S+i)*S+j]=(j<=i)?0.0f:-1e9f;

    IDXGIFactory4* fac; CreateDXGIFactory1(IID_PPV_ARGS(&fac)); IDXGIAdapter1* ad=nullptr; DXGI_ADAPTER_DESC1 dd{};
    for(UINT i=0; fac->EnumAdapters1(i,&ad)!=DXGI_ERROR_NOT_FOUND; ++i){ ad->GetDesc1(&dd); if(dd.Flags&DXGI_ADAPTER_FLAG_SOFTWARE){ad->Release();ad=nullptr;continue;} if(SUCCEEDED(D3D12CreateDevice(ad,D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&dev)))) break; ad->Release(); ad=nullptr; }
    if(!dev){ printf("no d3d12\n"); return 1; }
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type=D3D12_COMMAND_LIST_TYPE_DIRECT; dev->CreateCommandQueue(&qd,IID_PPV_ARGS(&q));
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&alloc)); dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,alloc,nullptr,IID_PPV_ARGS(&cl));
    dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)); fe=CreateEvent(nullptr,FALSE,FALSE,nullptr);
    DMLCreateDevice(dev,DML_CREATE_DEVICE_FLAG_NONE,IID_PPV_ARGS(&dml)); dml->CreateCommandRecorder(IID_PPV_ARGS(&rec));
    printf("[dev] %ls  ATTENTION S=%u E=%u Hn=%u Hd=%u (fused on-device: ln->QKV->MHA->proj->+res)\n",dd.Description,S,E,Hn,Hd);

    ID3D12Resource *bx=upload(x),*blng=upload(lng),*blnb=upload(lnb),*bwq=upload(wq),*bwk=upload(wk),*bwv=upload(wv),*bwp=upload(wp),*bbq=upload(bq),*bbk=upload(bk),*bbv=upload(bv),*bbp=upload(bp),*brpb=upload(rpb);
    ID3D12Resource *lnbuf=mkDef((UINT64)S*E*4),*qbuf=mkDef((UINT64)S*E*4),*kbuf=mkDef((UINT64)S*E*4),*vbuf=mkDef((UINT64)S*E*4),*attnbuf=mkDef((UINT64)S*E*4),*projbuf=mkDef((UINT64)S*E*4),*outbuf=mkDef((UINT64)S*E*4);

    DML_BUFFER_TENSOR_DESC t1,t2,t3,t4; DML_TENSOR_DESC xT=T(t1,{1,1,S,E}),lgT=T(t2,{1,1,1,E}),lbT=T(t3,{1,1,1,E}),lnoT=T(t4,{1,1,S,E});
    UINT axes[1]={3}; DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC mvn{}; mvn.InputTensor=&xT; mvn.ScaleTensor=&lgT; mvn.BiasTensor=&lbT; mvn.OutputTensor=&lnoT; mvn.AxisCount=1; mvn.Axes=axes; mvn.NormalizeVariance=TRUE; mvn.Epsilon=1e-5f;
    DML_OPERATOR_DESC mvnD{DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1,&mvn};

    // a GEMM+C op producing [S,E] = ln[S,E] @ W[E,E] + Cbias[S,E]
    auto gemmDesc=[&](DML_BUFFER_TENSOR_DESC&a,DML_BUFFER_TENSOR_DESC&b,DML_BUFFER_TENSOR_DESC&cc,DML_BUFFER_TENSOR_DESC&o,DML_TENSOR_DESC&aT,DML_TENSOR_DESC&bT,DML_TENSOR_DESC&cT,DML_TENSOR_DESC&oT,DML_GEMM_OPERATOR_DESC&g){
        aT=T(a,{1,1,S,E}); bT=T(b,{1,1,E,E}); cT=T(cc,{1,1,S,E}); oT=T(o,{1,1,S,E});
        g={}; g.ATensor=&aT; g.BTensor=&bT; g.CTensor=&cT; g.OutputTensor=&oT; g.TransA=DML_MATRIX_TRANSFORM_NONE; g.TransB=DML_MATRIX_TRANSFORM_NONE; g.Alpha=1; g.Beta=1; };
    DML_BUFFER_TENSOR_DESC qa,qb,qc,qo,ka,kb,kc,ko,va,vb,vc,vo,pa,pb,pc,po; DML_TENSOR_DESC qaT,qbT,qcT,qoT,kaT,kbT,kcT,koT,vaT,vbT,vcT,voT,paT,pbT,pcT,poT; DML_GEMM_OPERATOR_DESC gQ,gK,gV,gP;
    gemmDesc(qa,qb,qc,qo,qaT,qbT,qcT,qoT,gQ); gemmDesc(ka,kb,kc,ko,kaT,kbT,kcT,koT,gK); gemmDesc(va,vb,vc,vo,vaT,vbT,vcT,voT,gV); gemmDesc(pa,pb,pc,po,paT,pbT,pcT,poT,gP);
    DML_OPERATOR_DESC gQD{DML_OPERATOR_GEMM,&gQ},gKD{DML_OPERATOR_GEMM,&gK},gVD{DML_OPERATOR_GEMM,&gV},gPD{DML_OPERATOR_GEMM,&gP};

    // MHA: separate Q/K/V {1,S,E}, causal RelativePositionBias {1,Hn,S,S}, output {1,S,E}
    DML_BUFFER_TENSOR_DESC mq,mk,mv,mr,mo; DML_TENSOR_DESC mqT=T(mq,{1,S,E}),mkT=T(mk,{1,S,E}),mvT=T(mv,{1,S,E}),mrT=T(mr,{1,Hn,S,S}),moT=T(mo,{1,S,E});
    DML_MULTIHEAD_ATTENTION_OPERATOR_DESC mha{}; mha.QueryTensor=&mqT; mha.KeyTensor=&mkT; mha.ValueTensor=&mvT; mha.RelativePositionBiasTensor=&mrT; mha.OutputTensor=&moT;
    mha.Scale=scale; mha.MaskFilterValue=-1e9f; mha.HeadCount=Hn; mha.MaskType=DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
    DML_OPERATOR_DESC mhaD{DML_OPERATOR_MULTIHEAD_ATTENTION,&mha};

    DML_BUFFER_TENSOR_DESC ra,rb,ro; DML_TENSOR_DESC raT=T(ra,{1,1,S,E}),rbT=T(rb,{1,1,S,E}),roT=T(ro,{1,1,S,E});
    DML_ELEMENT_WISE_ADD1_OPERATOR_DESC add{&raT,&rbT,&roT,nullptr}; DML_OPERATOR_DESC addD{DML_OPERATOR_ELEMENT_WISE_ADD1,&add};

    IDMLCompiledOperator *cMvn=compile(mvnD,"MVN1"),*cQ=compile(gQD,"GEMM_q"),*cK=compile(gKD,"GEMM_k"),*cV=compile(gVD,"GEMM_v"),*cMha=compile(mhaD,"MHA"),*cP=compile(gPD,"GEMM_proj"),*cAdd=compile(addD,"ADD1");
    if(!cMvn||!cQ||!cK||!cV||!cMha||!cP||!cAdd) return 2;

    Op oMvn=setupOp(cMvn,{bx,blng,blnb},{lnbuf});
    Op oQ=setupOp(cQ,{lnbuf,bwq,bbq},{qbuf});
    Op oK=setupOp(cK,{lnbuf,bwk,bbk},{kbuf});
    Op oV=setupOp(cV,{lnbuf,bwv,bbv},{vbuf});
    Op oMha=setupOp(cMha,{qbuf,kbuf,vbuf, nullptr,nullptr,nullptr, nullptr,nullptr, brpb, nullptr,nullptr},{attnbuf,nullptr,nullptr});
    Op oP=setupOp(cP,{attnbuf,bwp,bbp},{projbuf});
    Op oAdd=setupOp(cAdd,{projbuf,bx},{outbuf});

    record(oMvn); record(oQ); record(oK); record(oV); record(oMha); record(oP); record(oAdd);
    auto b1=trans(outbuf,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE); cl->ResourceBarrier(1,&b1);
    ID3D12Resource* rbk=nullptr; auto rh=hp(D3D12_HEAP_TYPE_READBACK); auto rd=bd((UINT64)S*E*4); dev->CreateCommittedResource(&rh,D3D12_HEAP_FLAG_NONE,&rd,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&rbk));
    cl->CopyResource(rbk,outbuf); flush();
    std::vector<float> out(S*E); void*p; D3D12_RANGE rr{0,(SIZE_T)S*E*4}; rbk->Map(0,&rr,&p); memcpy(out.data(),p,S*E*4); D3D12_RANGE nw{0,0}; rbk->Unmap(0,&nw);

    double maxAbs=0,absmax=0; for(size_t i=0;i<(size_t)S*E;i++){ maxAbs=std::max(maxAbs,(double)std::fabs(out[i]-ref[i])); absmax=std::max(absmax,(double)std::fabs(ref[i])); }
    double nrm=maxAbs/absmax;
    printf("[out[0,0..2]] gpu=(%.5f,%.5f,%.5f) ref=(%.5f,%.5f,%.5f)\n",out[0],out[1],out[2],ref[0],ref[1],ref[2]);
    printf("[verify] max abs %.3e  scale %.3f  scale-norm %.2e\n",maxAbs,absmax,nrm);
    bool pass=nrm<1e-3;
    printf("=== %s: fused ATTENTION block (7 ops, one command list, activations resident) on HD4600 vs numpy ===\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
