// dml_mlp_run.cpp — fused gpt2 MLP block on the HD 4600, activations RESIDENT on-device.
// Chains 5 de-risked DML ops into ONE command list with intermediate GPU buffers and ONE flush:
//   ln = LayerNorm(x, g, b)      [MVN1, axis E]
//   fc = ln @ Wfc + bfc          [GEMM with C]
//   h  = gelu(fc)                [ACTIVATION_GELU, erf]
//   proj = h @ Wproj + bproj     [GEMM with C]
//   out = proj + x  (residual)   [ELEMENT_WISE_ADD1]
// Activations never leave the GPU between ops; only the final out_buf is read back. Verifies vs
// the numpy erf-gelu reference from mlp_prep.py.
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

static DML_TENSOR_DESC T(DML_BUFFER_TENSOR_DESC& bt, UINT a,UINT b,UINT c,UINT d){ bt={}; bt.DataType=DML_TENSOR_DATA_TYPE_FLOAT32; bt.DimensionCount=4; static thread_local UINT s[64][4]; static thread_local int n=0; UINT* z=s[n++%64]; z[0]=a;z[1]=b;z[2]=c;z[3]=d; bt.Sizes=z; bt.TotalTensorSizeInBytes=(UINT64)a*b*c*d*4; DML_TENSOR_DESC t{DML_TENSOR_TYPE_BUFFER,&bt}; return t; }

// A set-up DML op ready to record: compiled + its own descriptor heap + bound binding table.
struct Op { IDMLCompiledOperator* c; ID3D12DescriptorHeap* heap; IDMLBindingTable* bt; UINT descN; ID3D12Resource* tmp; UINT64 tmpB; };

static Op setupOp(IDMLCompiledOperator* c, std::vector<ID3D12Resource*> ins, ID3D12Resource* out, UINT64 outB){
    Op o{}; o.c=c;
    IDMLOperatorInitializer* ini=nullptr; IDMLCompiledOperator* cops[]={c}; dml->CreateOperatorInitializer(1,cops,IID_PPV_ARGS(&ini));
    DML_BINDING_PROPERTIES ip=ini->GetBindingProperties(), ep=c->GetBindingProperties();
    o.descN=std::max(ip.RequiredDescriptorCount,ep.RequiredDescriptorCount); o.tmpB=std::max(ip.TemporaryResourceSize,ep.TemporaryResourceSize);
    D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors=o.descN; hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE; dev->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&o.heap));
    o.tmp = o.tmpB? mkDef(o.tmpB):nullptr;
    ID3D12DescriptorHeap* heaps[]={o.heap}; cl->SetDescriptorHeaps(1,heaps);
    DML_BINDING_TABLE_DESC td{}; td.Dispatchable=ini; td.CPUDescriptorHandle=o.heap->GetCPUDescriptorHandleForHeapStart(); td.GPUDescriptorHandle=o.heap->GetGPUDescriptorHandleForHeapStart(); td.SizeInDescriptors=o.descN; dml->CreateBindingTable(&td,IID_PPV_ARGS(&o.bt));
    if(o.tmpB && ip.TemporaryResourceSize){ DML_BUFFER_BINDING b{o.tmp,0,o.tmpB}; DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER,&b}; o.bt->BindTemporaryResource(&d); }
    rec->RecordDispatch(cl,ini,o.bt); flush(); ini->Release();
    // rebind for execution
    td.Dispatchable=c; o.bt->Reset(&td);
    if(o.tmpB){ DML_BUFFER_BINDING b{o.tmp,0,o.tmpB}; DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER,&b}; o.bt->BindTemporaryResource(&d); }
    std::vector<DML_BUFFER_BINDING> ibb; for(auto* r: ins){ auto rd=r->GetDesc(); ibb.push_back({r,0,rd.Width}); }
    std::vector<DML_BINDING_DESC> ibd; for(auto& b: ibb) ibd.push_back({DML_BINDING_TYPE_BUFFER,&b});
    o.bt->BindInputs((UINT)ibd.size(), ibd.data());
    DML_BUFFER_BINDING ob{out,0,outB}; DML_BINDING_DESC od{DML_BINDING_TYPE_BUFFER,&ob}; o.bt->BindOutputs(1,&od);
    return o;
}
static void record(Op& o){ ID3D12DescriptorHeap* heaps[]={o.heap}; cl->SetDescriptorHeaps(1,heaps); rec->RecordDispatch(cl,o.c,o.bt); auto u=uavbar(); cl->ResourceBarrier(1,&u); }

static IDMLCompiledOperator* compile(const DML_OPERATOR_DESC& od,const char* name){ IDMLOperator* o=nullptr; if(FAILED(dml->CreateOperator(&od,IID_PPV_ARGS(&o)))){ printf("%s create FAILED\n",name); return nullptr; } IDMLCompiledOperator* c=nullptr; if(FAILED(dml->CompileOperator(o,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&c)))){ printf("%s compile FAILED\n",name); return nullptr; } o->Release(); return c; }

int main(){
    UINT S=0,E=0,H=0; { std::ifstream f("mlp_dims.txt"); std::stringstream d; d<<f.rdbuf(); d>>S>>E>>H; }
    if(!S||!E||!H){ printf("no dims (run from scratch/dml after mlp_prep.py)\n"); return 1; }
    auto x=readBin("mlp_x.bin",(size_t)S*E), lng=readBin("mlp_lng.bin",E), lnb=readBin("mlp_lnb.bin",E);
    auto wfc=readBin("mlp_wfc.bin",(size_t)E*H), bfc=readBin("mlp_bfc.bin",(size_t)S*H);
    auto wproj=readBin("mlp_wproj.bin",(size_t)H*E), bproj=readBin("mlp_bproj.bin",(size_t)S*E);
    auto ref=readBin("mlp_ref.bin",(size_t)S*E);

    IDXGIFactory4* fac; CreateDXGIFactory1(IID_PPV_ARGS(&fac)); IDXGIAdapter1* ad=nullptr; DXGI_ADAPTER_DESC1 dd{};
    for(UINT i=0; fac->EnumAdapters1(i,&ad)!=DXGI_ERROR_NOT_FOUND; ++i){ ad->GetDesc1(&dd); if(dd.Flags&DXGI_ADAPTER_FLAG_SOFTWARE){ad->Release();ad=nullptr;continue;} if(SUCCEEDED(D3D12CreateDevice(ad,D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&dev)))) break; ad->Release(); ad=nullptr; }
    if(!dev){ printf("no d3d12\n"); return 1; }
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type=D3D12_COMMAND_LIST_TYPE_DIRECT; dev->CreateCommandQueue(&qd,IID_PPV_ARGS(&q));
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&alloc)); dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,alloc,nullptr,IID_PPV_ARGS(&cl));
    dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)); fe=CreateEvent(nullptr,FALSE,FALSE,nullptr);
    DMLCreateDevice(dev,DML_CREATE_DEVICE_FLAG_NONE,IID_PPV_ARGS(&dml)); dml->CreateCommandRecorder(IID_PPV_ARGS(&rec));
    printf("[dev] %ls  MLP S=%u E=%u H=%u (fused on-device: ln->fc->gelu->proj->+res)\n",dd.Description,S,E,H);

    // input + intermediate buffers (intermediates stay resident on the GPU across ops)
    ID3D12Resource *bx=upload(x), *blng=upload(lng), *blnb=upload(lnb), *bwfc=upload(wfc), *bbfc=upload(bfc), *bwproj=upload(wproj), *bbproj=upload(bproj);
    ID3D12Resource *lnbuf=mkDef((UINT64)S*E*4), *fcbuf=mkDef((UINT64)S*H*4), *gbuf=mkDef((UINT64)S*H*4), *projbuf=mkDef((UINT64)S*E*4), *outbuf=mkDef((UINT64)S*E*4);

    // compile ops
    DML_BUFFER_TENSOR_DESC bxT,glnT,blnT,olnT; DML_TENSOR_DESC xT=T(bxT,1,1,S,E),lgT=T(glnT,1,1,1,E),lbT=T(blnT,1,1,1,E),lnoT=T(olnT,1,1,S,E);
    UINT axes[1]={3};
    DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC mvn{}; mvn.InputTensor=&xT; mvn.ScaleTensor=&lgT; mvn.BiasTensor=&lbT; mvn.OutputTensor=&lnoT; mvn.AxisCount=1; mvn.Axes=axes; mvn.NormalizeVariance=TRUE; mvn.Epsilon=1e-5f;
    DML_OPERATOR_DESC mvnD{DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1,&mvn};

    DML_BUFFER_TENSOR_DESC aT1,bT1,cT1,oT1; DML_TENSOR_DESC gA=T(aT1,1,1,S,E),gB=T(bT1,1,1,E,H),gC=T(cT1,1,1,S,H),gO=T(oT1,1,1,S,H);
    DML_GEMM_OPERATOR_DESC gfc{}; gfc.ATensor=&gA; gfc.BTensor=&gB; gfc.CTensor=&gC; gfc.OutputTensor=&gO; gfc.TransA=DML_MATRIX_TRANSFORM_NONE; gfc.TransB=DML_MATRIX_TRANSFORM_NONE; gfc.Alpha=1; gfc.Beta=1;
    DML_OPERATOR_DESC gfcD{DML_OPERATOR_GEMM,&gfc};

    DML_BUFFER_TENSOR_DESC geiT,geoT; DML_TENSOR_DESC geI=T(geiT,1,1,S,H),geO=T(geoT,1,1,S,H);
    DML_ACTIVATION_GELU_OPERATOR_DESC gelu{&geI,&geO}; DML_OPERATOR_DESC geluD{DML_OPERATOR_ACTIVATION_GELU,&gelu};

    DML_BUFFER_TENSOR_DESC aT2,bT2,cT2,oT2; DML_TENSOR_DESC pA=T(aT2,1,1,S,H),pB=T(bT2,1,1,H,E),pC=T(cT2,1,1,S,E),pO=T(oT2,1,1,S,E);
    DML_GEMM_OPERATOR_DESC gpr{}; gpr.ATensor=&pA; gpr.BTensor=&pB; gpr.CTensor=&pC; gpr.OutputTensor=&pO; gpr.TransA=DML_MATRIX_TRANSFORM_NONE; gpr.TransB=DML_MATRIX_TRANSFORM_NONE; gpr.Alpha=1; gpr.Beta=1;
    DML_OPERATOR_DESC gprD{DML_OPERATOR_GEMM,&gpr};

    DML_BUFFER_TENSOR_DESC raT,rbT,roT; DML_TENSOR_DESC rA=T(raT,1,1,S,E),rB=T(rbT,1,1,S,E),rO=T(roT,1,1,S,E);
    DML_ELEMENT_WISE_ADD1_OPERATOR_DESC add{&rA,&rB,&rO,nullptr}; DML_OPERATOR_DESC addD{DML_OPERATOR_ELEMENT_WISE_ADD1,&add};

    IDMLCompiledOperator *cMvn=compile(mvnD,"MVN1"), *cFc=compile(gfcD,"GEMM_fc"), *cGe=compile(geluD,"GELU"), *cPr=compile(gprD,"GEMM_proj"), *cAdd=compile(addD,"ADD1");
    if(!cMvn||!cFc||!cGe||!cPr||!cAdd){ return 2; }

    Op oMvn=setupOp(cMvn,{bx,blng,blnb},lnbuf,(UINT64)S*E*4);
    Op oFc =setupOp(cFc,{lnbuf,bwfc,bbfc},fcbuf,(UINT64)S*H*4);
    Op oGe =setupOp(cGe,{fcbuf},gbuf,(UINT64)S*H*4);
    Op oPr =setupOp(cPr,{gbuf,bwproj,bbproj},projbuf,(UINT64)S*E*4);
    Op oAdd=setupOp(cAdd,{projbuf,bx},outbuf,(UINT64)S*E*4);

    // ONE command list: whole MLP, activations resident, single flush.
    record(oMvn); record(oFc); record(oGe); record(oPr); record(oAdd);
    auto b1=trans(outbuf,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE); cl->ResourceBarrier(1,&b1);
    ID3D12Resource* rb=nullptr; auto rh=hp(D3D12_HEAP_TYPE_READBACK); auto rd=bd((UINT64)S*E*4); dev->CreateCommittedResource(&rh,D3D12_HEAP_FLAG_NONE,&rd,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&rb));
    cl->CopyResource(rb,outbuf); flush();
    std::vector<float> out(S*E); void*p; D3D12_RANGE rr{0,(SIZE_T)S*E*4}; rb->Map(0,&rr,&p); memcpy(out.data(),p,S*E*4); D3D12_RANGE nw{0,0}; rb->Unmap(0,&nw);

    double maxAbs=0,absmax=0; for(size_t i=0;i<(size_t)S*E;i++){ maxAbs=std::max(maxAbs,(double)std::fabs(out[i]-ref[i])); absmax=std::max(absmax,(double)std::fabs(ref[i])); }
    double nrm=maxAbs/absmax;
    printf("[out[0,0..2]] gpu=(%.5f,%.5f,%.5f) ref=(%.5f,%.5f,%.5f)\n",out[0],out[1],out[2],ref[0],ref[1],ref[2]);
    printf("[verify] max abs %.3e  scale %.3f  scale-norm %.2e\n",maxAbs,absmax,nrm);
    bool pass=nrm<1e-3;
    printf("=== %s: fused MLP block (5 ops, one command list, activations resident) on HD4600 vs numpy erf-gelu ===\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
