// #004-B1 de-risk (probe): is DirectML's capacity-shaped past available on THIS device?
// MHA1 + PastSequenceLengthsTensor is gated at DML feature level 6.3; this HD 4600 measured 6.2.
// Probe both routes: (A) MHA1 + PastSequenceLengths, (B) base MHA + fixed-capacity past + a
// validity mask (RelativePositionBias -inf on the unused capacity slots). Report which CREATES.
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

static ID3D12Device* dev; static IDMLDevice* dml;
static DML_TENSOR_DESC Td(DML_BUFFER_TENSOR_DESC& bt, std::vector<UINT> sz, DML_TENSOR_DATA_TYPE dt=DML_TENSOR_DATA_TYPE_FLOAT32){ bt={}; bt.DataType=dt; bt.DimensionCount=(UINT)sz.size(); static std::vector<std::vector<UINT>> keep; keep.push_back(sz); bt.Sizes=keep.back().data(); UINT64 e=1; for(UINT s:sz) e*=s; bt.TotalTensorSizeInBytes=e*((dt==DML_TENSOR_DATA_TYPE_FLOAT32||dt==DML_TENSOR_DATA_TYPE_UINT32)?4:2); DML_TENSOR_DESC t{DML_TENSOR_TYPE_BUFFER,&bt}; return t; }

int main(){
    IDXGIFactory4* fac; CreateDXGIFactory1(IID_PPV_ARGS(&fac)); IDXGIAdapter1* ad=nullptr; DXGI_ADAPTER_DESC1 dd{};
    for(UINT i=0; fac->EnumAdapters1(i,&ad)!=DXGI_ERROR_NOT_FOUND; ++i){ ad->GetDesc1(&dd); if(dd.Flags&DXGI_ADAPTER_FLAG_SOFTWARE){ad->Release();ad=nullptr;continue;} if(SUCCEEDED(D3D12CreateDevice(ad,D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&dev)))) break; ad->Release(); ad=nullptr; }
    if(!dev){ printf("no d3d12\n"); return 1; }
    DMLCreateDevice(dev,DML_CREATE_DEVICE_FLAG_NONE,IID_PPV_ARGS(&dml));
    printf("[dev] %ls\n", dd.Description);
    { DML_FEATURE_LEVEL want[]={DML_FEATURE_LEVEL_6_2,DML_FEATURE_LEVEL_6_3,DML_FEATURE_LEVEL_6_4};
      DML_FEATURE_QUERY_FEATURE_LEVELS qy{3,want}; DML_FEATURE_DATA_FEATURE_LEVELS data{};
      dml->CheckFeatureSupport(DML_FEATURE_FEATURE_LEVELS,sizeof(qy),&qy,sizeof(data),&data);
      printf("[dml] max feature level = 0x%04x   (MHA1+PastSequenceLengths needs 0x6300)\n",(unsigned)data.MaxSupportedFeatureLevel); }

    const UINT Hn=2, Hd=3, E=Hn*Hd, C=6, P=3;   // capacity C, logical extent P (< C)
    const float scale=1.0f/std::sqrt((float)Hd);

    // (A) MHA1 + PastSequenceLengths : physical past [1,Hn,C,Hd], valid length P
    {
        DML_BUFFER_TENSOR_DESC qd,kd,vd,pk,pv,psl,od,ok,ov;
        DML_TENSOR_DESC qt=Td(qd,{1,1,E}),kt=Td(kd,{1,1,E}),vt=Td(vd,{1,1,E}),
            pkt=Td(pk,{1,Hn,C,Hd}),pvt=Td(pv,{1,Hn,C,Hd}),pslt=Td(psl,{1},DML_TENSOR_DATA_TYPE_UINT32),
            ot=Td(od,{1,1,E}),okt=Td(ok,{1,Hn,C,Hd}),ovt=Td(ov,{1,Hn,C,Hd});
        DML_MULTIHEAD_ATTENTION1_OPERATOR_DESC m{}; m.QueryTensor=&qt; m.KeyTensor=&kt; m.ValueTensor=&vt;
        m.PastKeyTensor=&pkt; m.PastValueTensor=&pvt; m.PastSequenceLengthsTensor=&pslt;
        m.OutputTensor=&ot; m.OutputPresentKeyTensor=&okt; m.OutputPresentValueTensor=&ovt;
        m.Scale=scale; m.MaskFilterValue=-1e9f; m.QueryHeadCount=Hn; m.KeyValueHeadCount=Hn; m.MaskType=DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
        DML_OPERATOR_DESC od1{DML_OPERATOR_MULTIHEAD_ATTENTION1,&m};
        IDMLOperator* o=nullptr; HRESULT ch=dml->CreateOperator(&od1,IID_PPV_ARGS(&o)); HRESULT ph=E_FAIL;
        if(SUCCEEDED(ch)){ IDMLCompiledOperator* c=nullptr; ph=dml->CompileOperator(o,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&c)); if(c)c->Release(); o->Release(); }
        printf("[A MHA1+PastSeqLen] create=0x%08x compile=0x%08x %s\n",(unsigned)ch,(unsigned)ph,(SUCCEEDED(ch)&&SUCCEEDED(ph))?"<== AVAILABLE":"<== unavailable");
    }

    // (A') MHA1 desc-shape sweep — find what CreateOperator accepts (past/present fixed at capacity C)
    auto probeMHA1=[&](const char* name, std::vector<UINT> pslDims, std::vector<UINT> qkvDims, std::vector<UINT> presDims, bool present){
        DML_BUFFER_TENSOR_DESC qd,kd,vd,pk,pv,psl,od,ok,ov;
        DML_TENSOR_DESC qt=Td(qd,qkvDims),kt=Td(kd,qkvDims),vt=Td(vd,qkvDims),
            pkt=Td(pk,{1,Hn,C,Hd}),pvt=Td(pv,{1,Hn,C,Hd}),pslt=Td(psl,pslDims,DML_TENSOR_DATA_TYPE_UINT32),
            ot=Td(od,{1,1,E}),okt=Td(ok,presDims),ovt=Td(ov,presDims);
        DML_MULTIHEAD_ATTENTION1_OPERATOR_DESC m{}; m.QueryTensor=&qt; m.KeyTensor=&kt; m.ValueTensor=&vt;
        m.PastKeyTensor=&pkt; m.PastValueTensor=&pvt; m.PastSequenceLengthsTensor=&pslt;
        m.OutputTensor=&ot; if(present){ m.OutputPresentKeyTensor=&okt; m.OutputPresentValueTensor=&ovt; }
        m.Scale=scale; m.MaskFilterValue=-1e9f; m.QueryHeadCount=Hn; m.KeyValueHeadCount=Hn; m.MaskType=DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
        DML_OPERATOR_DESC od1{DML_OPERATOR_MULTIHEAD_ATTENTION1,&m};
        IDMLOperator* o=nullptr; HRESULT ch=dml->CreateOperator(&od1,IID_PPV_ARGS(&o)); HRESULT ph=E_FAIL;
        if(SUCCEEDED(ch)){ IDMLCompiledOperator* c=nullptr; ph=dml->CompileOperator(o,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&c)); if(c)c->Release(); o->Release(); }
        printf("  [A' %-30s] create=0x%08x compile=0x%08x %s\n",name,(unsigned)ch,(unsigned)ph,(SUCCEEDED(ch)&&SUCCEEDED(ph))?"<== OK":"");
    };
    probeMHA1("psl[1] present[1,Hn,C,Hd]",   {1},        {1,1,E}, {1,Hn,C,Hd}, true);
    probeMHA1("psl[1,1,1,1] pres[1,Hn,C,Hd]",{1,1,1,1},  {1,1,E}, {1,Hn,C,Hd}, true);
    probeMHA1("psl[1] present OFF",          {1},        {1,1,E}, {1,Hn,C,Hd}, false);
    probeMHA1("psl[1,1,1,1] present OFF",    {1,1,1,1},  {1,1,E}, {1,Hn,C,Hd}, false);
    probeMHA1("qkv[1,1,1,E] psl[1,1,1,1]",   {1,1,1,1},  {1,1,1,E},{1,Hn,C,Hd}, true);

    // (B) base MHA + capacity past [1,Hn,C,Hd] + validity mask via RelativePositionBias [1,Hn,1,C+1]
    {
        DML_BUFFER_TENSOR_DESC qd,kd,vd,pk,pv,rb,od,ok,ov;
        DML_TENSOR_DESC qt=Td(qd,{1,1,E}),kt=Td(kd,{1,1,E}),vt=Td(vd,{1,1,E}),
            pkt=Td(pk,{1,Hn,C,Hd}),pvt=Td(pv,{1,Hn,C,Hd}),rbt=Td(rb,{1,Hn,1,C+1}),
            ot=Td(od,{1,1,E}),okt=Td(ok,{1,Hn,C+1,Hd}),ovt=Td(ov,{1,Hn,C+1,Hd});
        DML_MULTIHEAD_ATTENTION_OPERATOR_DESC m{}; m.QueryTensor=&qt; m.KeyTensor=&kt; m.ValueTensor=&vt;
        m.PastKeyTensor=&pkt; m.PastValueTensor=&pvt; m.RelativePositionBiasTensor=&rbt;
        m.OutputTensor=&ot; m.OutputPresentKeyTensor=&okt; m.OutputPresentValueTensor=&ovt;
        m.Scale=scale; m.MaskFilterValue=-1e9f; m.HeadCount=Hn; m.MaskType=DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
        DML_OPERATOR_DESC od1{DML_OPERATOR_MULTIHEAD_ATTENTION,&m};
        IDMLOperator* o=nullptr; HRESULT ch=dml->CreateOperator(&od1,IID_PPV_ARGS(&o)); HRESULT ph=E_FAIL;
        if(SUCCEEDED(ch)){ IDMLCompiledOperator* c=nullptr; ph=dml->CompileOperator(o,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&c)); if(c)c->Release(); o->Release(); }
        printf("[B baseMHA+mask   ] create=0x%08x compile=0x%08x %s\n",(unsigned)ch,(unsigned)ph,(SUCCEEDED(ch)&&SUCCEEDED(ph))?"<== AVAILABLE":"<== unavailable");
    }
    return 0;
}
