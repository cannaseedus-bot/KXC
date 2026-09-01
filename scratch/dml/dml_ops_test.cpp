// dml_ops_test.cpp — DE-RISK: do the non-GEMM ops a fused MLP graph needs actually EXECUTE with
// correct values on this HD 4600 (FL 11_1 tier-1)? API presence != execution here (the GEMM
// C-slot no-op taught us that). Value-checks GELU, LayerNorm (MVN1), and Add vs C++ references.
#define NOMINMAX
#define DML_TARGET_VERSION_USE_LATEST 1   // GELU/MVN1 are gated behind newer DML_TARGET_VERSION
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

// Run a single compiled DML op over the given input buffers; return the output (outN floats).
static std::vector<float> runOp(IDMLCompiledOperator* op, std::vector<ID3D12Resource*> ins, UINT outN){
    IDMLOperatorInitializer* ini=nullptr; IDMLCompiledOperator* cops[]={op};
    HR(dml->CreateOperatorInitializer(1,cops,IID_PPV_ARGS(&ini)));
    DML_BINDING_PROPERTIES ip=ini->GetBindingProperties(), ep=op->GetBindingProperties();
    UINT descN=std::max(ip.RequiredDescriptorCount,ep.RequiredDescriptorCount);
    UINT64 tmpB=std::max(ip.TemporaryResourceSize,ep.TemporaryResourceSize), perB=ep.PersistentResourceSize;
    ID3D12DescriptorHeap* heap=nullptr; D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors=descN; hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE; HR(dev->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap)));
    ID3D12DescriptorHeap* heaps[]={heap}; cl->SetDescriptorHeaps(1,heaps);
    ID3D12Resource* tmp = tmpB? mkDef(tmpB):nullptr; ID3D12Resource* per = perB? mkDef(perB):nullptr;
    IDMLBindingTable* bt=nullptr; DML_BINDING_TABLE_DESC td{}; td.Dispatchable=ini; td.CPUDescriptorHandle=heap->GetCPUDescriptorHandleForHeapStart(); td.GPUDescriptorHandle=heap->GetGPUDescriptorHandleForHeapStart(); td.SizeInDescriptors=descN; HR(dml->CreateBindingTable(&td,IID_PPV_ARGS(&bt)));
    if(tmpB && ip.TemporaryResourceSize){ DML_BUFFER_BINDING b{tmp,0,tmpB}; DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER,&b}; bt->BindTemporaryResource(&d); }
    if(perB){ DML_BUFFER_BINDING b{per,0,perB}; DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER,&b}; bt->BindOutputs(1,&d); }
    rec->RecordDispatch(cl,ini,bt); flush(); ini->Release();

    ID3D12Resource* out=mkDef((UINT64)outN*4);
    td.Dispatchable=op; bt->Reset(&td);
    if(tmpB){ DML_BUFFER_BINDING b{tmp,0,tmpB}; DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER,&b}; bt->BindTemporaryResource(&d); }
    if(perB){ DML_BUFFER_BINDING b{per,0,perB}; DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER,&b}; bt->BindPersistentResource(&d); }
    std::vector<DML_BUFFER_BINDING> ibb; std::vector<DML_BINDING_DESC> ibd;
    for(auto* r: ins){ D3D12_RESOURCE_DESC rd=r->GetDesc(); ibb.push_back({r,0,rd.Width}); }
    for(auto& b: ibb) ibd.push_back({DML_BINDING_TYPE_BUFFER,&b});
    bt->BindInputs((UINT)ibd.size(), ibd.data());
    DML_BUFFER_BINDING ob{out,0,(UINT64)outN*4}; DML_BINDING_DESC od{DML_BINDING_TYPE_BUFFER,&ob}; bt->BindOutputs(1,&od);
    cl->SetDescriptorHeaps(1,heaps);
    rec->RecordDispatch(cl,op,bt); flush();

    std::vector<float> res(outN); ID3D12Resource* rb=nullptr; auto h=hp(D3D12_HEAP_TYPE_READBACK); auto d=bd((UINT64)outN*4); dev->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&rb));
    auto b1=br(out,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE); cl->ResourceBarrier(1,&b1); cl->CopyResource(rb,out); flush();
    void*p; D3D12_RANGE rr{0,(SIZE_T)outN*4}; rb->Map(0,&rr,&p); memcpy(res.data(),p,outN*4); D3D12_RANGE nw{0,0}; rb->Unmap(0,&nw); rb->Release();
    return res;
}

static DML_TENSOR_DESC td4(DML_BUFFER_TENSOR_DESC& bt, UINT* sizes){ bt={}; bt.DataType=DML_TENSOR_DATA_TYPE_FLOAT32; bt.DimensionCount=4; bt.Sizes=sizes; UINT64 e=1; for(int i=0;i<4;i++) e*=sizes[i]; bt.TotalTensorSizeInBytes=e*4; DML_TENSOR_DESC d{DML_TENSOR_TYPE_BUFFER,&bt}; return d; }
static double snerr(const std::vector<float>& a, const std::vector<float>& b){ double m=0,s=0; for(size_t i=0;i<a.size();i++){ m=std::max(m,(double)std::fabs(a[i]-b[i])); s=std::max(s,(double)std::fabs(b[i])); } return m/s; }

int main(){
    IDXGIFactory4* fac; CreateDXGIFactory1(IID_PPV_ARGS(&fac)); IDXGIAdapter1* ad=nullptr; DXGI_ADAPTER_DESC1 dd{};
    for(UINT i=0; fac->EnumAdapters1(i,&ad)!=DXGI_ERROR_NOT_FOUND; ++i){ ad->GetDesc1(&dd); if(dd.Flags&DXGI_ADAPTER_FLAG_SOFTWARE){ad->Release();ad=nullptr;continue;} if(SUCCEEDED(D3D12CreateDevice(ad,D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&dev)))) break; ad->Release(); ad=nullptr; }
    if(!dev){ printf("no d3d12\n"); return 1; }
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type=D3D12_COMMAND_LIST_TYPE_DIRECT; dev->CreateCommandQueue(&qd,IID_PPV_ARGS(&q));
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&alloc)); dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,alloc,nullptr,IID_PPV_ARGS(&cl));
    dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)); fe=CreateEvent(nullptr,FALSE,FALSE,nullptr);
    DMLCreateDevice(dev,DML_CREATE_DEVICE_FLAG_NONE,IID_PPV_ARGS(&dml)); dml->CreateCommandRecorder(IID_PPV_ARGS(&rec));
    printf("[dev] %ls\n", dd.Description);
    int fails=0;

    // ---- GELU (exact/erf): y = 0.5 x (1 + erf(x/sqrt2)) ----
    { const UINT n=16; std::vector<float> x(n); for(UINT i=0;i<n;i++) x[i]=(float)((i%7)-3)*0.7f;
      UINT sz[4]={1,1,1,n}; DML_BUFFER_TENSOR_DESC bt; DML_TENSOR_DESC t=td4(bt,sz);
      DML_ACTIVATION_GELU_OPERATOR_DESC g{&t,&t}; DML_OPERATOR_DESC od{DML_OPERATOR_ACTIVATION_GELU,&g};
      IDMLOperator* o=nullptr; dml->CreateOperator(&od,IID_PPV_ARGS(&o)); IDMLCompiledOperator* c=nullptr;
      if(FAILED(dml->CompileOperator(o,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&c)))){ printf("[GELU] compile FAILED\n"); fails++; }
      else { auto out=runOp(c,{upload(x)},n); std::vector<float> ref(n); for(UINT i=0;i<n;i++) ref[i]=0.5f*x[i]*(1.0f+erff(x[i]*0.70710678f));
        double e=snerr(out,ref); printf("[GELU] scale-norm err %.2e  %s\n", e, e<1e-4?"PASS":"FAIL"); if(!(e<1e-4)) fails++; } }

    // ---- LayerNorm via MVN1 over last axis, with scale (gamma) + bias (beta) ----
    { const UINT S=4,E=8,n=S*E; std::vector<float> x(n),gamma(E),beta(E); for(UINT i=0;i<n;i++) x[i]=(float)((i*13%17)-8)*0.3f; for(UINT i=0;i<E;i++){gamma[i]=1.0f+0.1f*i; beta[i]=0.05f*i;}
      UINT xs[4]={1,1,S,E}, gs[4]={1,1,1,E}; DML_BUFFER_TENSOR_DESC xb,gb,bb,ob; DML_TENSOR_DESC xt=td4(xb,xs),gt=td4(gb,gs),bt2=td4(bb,gs),ot=td4(ob,xs);
      UINT axes[1]={3};
      DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC m{}; m.InputTensor=&xt; m.ScaleTensor=&gt; m.BiasTensor=&bt2; m.OutputTensor=&ot; m.AxisCount=1; m.Axes=axes; m.NormalizeVariance=TRUE; m.Epsilon=1e-5f;
      DML_OPERATOR_DESC od{DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1,&m};
      IDMLOperator* o=nullptr; dml->CreateOperator(&od,IID_PPV_ARGS(&o)); IDMLCompiledOperator* c=nullptr;
      if(FAILED(dml->CompileOperator(o,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&c)))){ printf("[LayerNorm/MVN1] compile FAILED\n"); fails++; }
      else { auto out=runOp(c,{upload(x),upload(gamma),upload(beta)},n);
        std::vector<float> ref(n); for(UINT s=0;s<S;s++){ double mu=0; for(UINT e=0;e<E;e++) mu+=x[s*E+e]; mu/=E; double var=0; for(UINT e=0;e<E;e++){ double d=x[s*E+e]-mu; var+=d*d; } var/=E; double inv=1.0/std::sqrt(var+1e-5); for(UINT e=0;e<E;e++) ref[s*E+e]=(float)((x[s*E+e]-mu)*inv*gamma[e]+beta[e]); }
        double e=snerr(out,ref); printf("[LayerNorm/MVN1] scale-norm err %.2e  %s\n", e, e<1e-3?"PASS":"FAIL"); if(!(e<1e-3)) fails++; } }

    // ---- Element-wise Add ----
    { const UINT n=16; std::vector<float> a(n),b(n); for(UINT i=0;i<n;i++){a[i]=(float)i*0.5f-3; b[i]=(float)(n-i)*0.25f;}
      UINT sz[4]={1,1,1,n}; DML_BUFFER_TENSOR_DESC ba,bbt,bo; DML_TENSOR_DESC ta=td4(ba,sz),tb=td4(bbt,sz),to=td4(bo,sz);
      DML_ELEMENT_WISE_ADD1_OPERATOR_DESC ad{&ta,&tb,&to,nullptr}; DML_OPERATOR_DESC od{DML_OPERATOR_ELEMENT_WISE_ADD1,&ad};
      IDMLOperator* o=nullptr; HRESULT chr=dml->CreateOperator(&od,IID_PPV_ARGS(&o)); IDMLCompiledOperator* c=nullptr;
      HRESULT phr = o ? dml->CompileOperator(o,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&c)) : chr;
      if(FAILED(phr)){ printf("[Add1] FAILED create=0x%08x compile=0x%08x\n",(unsigned)chr,(unsigned)phr); fails++; }
      else { auto out=runOp(c,{upload(a),upload(b)},n); std::vector<float> ref(n); for(UINT i=0;i<n;i++) ref[i]=a[i]+b[i];
        double e=snerr(out,ref); printf("[Add1] scale-norm err %.2e  %s\n", e, e<1e-5?"PASS":"FAIL"); if(!(e<1e-5)) fails++; } }

    printf("=== %s: DirectML non-GEMM ops on the HD 4600 (FL 11_1) ===\n", fails==0?"PASS":"FAIL");
    return fails;
}
