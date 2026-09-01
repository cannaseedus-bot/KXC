// attn_run.cpp — dispatch KHANARY's G_ATTENTION cs_5_0 causal MHA kernel on the HD 4600.
// qkv is a REAL gpt2 QKV projection (seeded hidden @ transformer.h.0.attn.c_attn.weight+bias);
// verify attn_out[S,E] vs a numpy causal-MHA f64 reference. Promotes gpt2_attn_fwd.hlsl to a glyph.
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <sstream>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
static std::string readFile(const char*p){std::ifstream f(p,std::ios::binary);std::stringstream s;s<<f.rdbuf();return s.str();}
static std::vector<float> readBin(const char*p,size_t n){std::vector<float> v(n);std::ifstream f(p,std::ios::binary);f.read((char*)v.data(),n*4);return v;}

int main(){
    UINT S=0,E=0,H=0; float scale=0.f;
    { std::stringstream d(readFile("attn_dims.txt")); d>>S>>E>>H>>scale; }
    if(!S||!E||!H){printf("no dims\n");return 1;}
    UINT D=E/H;
    std::string hlsl=readFile("knu_attn.hlsl");
    if(hlsl.empty()){printf("no hlsl\n");return 1;}
    std::vector<float> qkv=readBin("attn_qkv.bin",(size_t)S*3*E), ref=readBin("attn_ref.bin",(size_t)S*E);

    D3D_FEATURE_LEVEL fl{}; ID3D11Device*dev=nullptr; ID3D11DeviceContext*ctx=nullptr;
    D3D_FEATURE_LEVEL want[]={D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0};
    if(FAILED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,want,2,D3D11_SDK_VERSION,&dev,&fl,&ctx))){printf("no d3d11\n");return 1;}
    printf("[dev] D3D11 FL 0x%x  causal MHA S=%u E=%u H=%u D=%u scale=%.4f (real gpt2 qkv)\n",(int)fl,S,E,H,D,scale);
    ID3DBlob*blob=nullptr;ID3DBlob*err=nullptr;
    if(FAILED(D3DCompile(hlsl.data(),hlsl.size(),"attn",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&err))){if(err)printf("HLSL err: %s\n",(char*)err->GetBufferPointer());return 1;}
    ID3D11ComputeShader*cs=nullptr;dev->CreateComputeShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&cs);

    auto structSRV=[&](const void*data,UINT count)->ID3D11ShaderResourceView*{
        D3D11_BUFFER_DESC bd{};bd.ByteWidth=count*4;bd.Usage=D3D11_USAGE_DEFAULT;bd.BindFlags=D3D11_BIND_SHADER_RESOURCE;bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;bd.StructureByteStride=4;
        D3D11_SUBRESOURCE_DATA sd{};sd.pSysMem=data;ID3D11Buffer*b=nullptr;dev->CreateBuffer(&bd,&sd,&b);
        D3D11_SHADER_RESOURCE_VIEW_DESC sv{};sv.Format=DXGI_FORMAT_UNKNOWN;sv.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;sv.Buffer.NumElements=count;
        ID3D11ShaderResourceView*srv=nullptr;dev->CreateShaderResourceView(b,&sv,&srv);return srv;};
    auto structUAV=[&](UINT count)->ID3D11UnorderedAccessView*{
        D3D11_BUFFER_DESC bd{};bd.ByteWidth=count*4;bd.Usage=D3D11_USAGE_DEFAULT;bd.BindFlags=D3D11_BIND_UNORDERED_ACCESS;bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;bd.StructureByteStride=4;
        ID3D11Buffer*b=nullptr;dev->CreateBuffer(&bd,nullptr,&b);
        D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};ud.Format=DXGI_FORMAT_UNKNOWN;ud.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;ud.Buffer.NumElements=count;
        ID3D11UnorderedAccessView*uav=nullptr;dev->CreateUnorderedAccessView(b,&ud,&uav);return uav;};

    ID3D11ShaderResourceView*srvQ=structSRV(qkv.data(),S*3*E);
    // attn_out UAV — need its buffer for readback, so create explicitly
    ID3D11Buffer*bOut=nullptr;{D3D11_BUFFER_DESC bd{};bd.ByteWidth=S*E*4;bd.Usage=D3D11_USAGE_DEFAULT;bd.BindFlags=D3D11_BIND_UNORDERED_ACCESS;bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;bd.StructureByteStride=4;dev->CreateBuffer(&bd,nullptr,&bOut);}
    ID3D11UnorderedAccessView*uavOut=nullptr;{D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};ud.Format=DXGI_FORMAT_UNKNOWN;ud.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;ud.Buffer.NumElements=S*E;dev->CreateUnorderedAccessView(bOut,&ud,&uavOut);}
    ID3D11UnorderedAccessView*uavP=structUAV(H*S*S);   // P_buf scratch
    struct CB{uint32_t seq_len,n_embd,head_dim;float scale;} cb{S,E,D,scale};
    ID3D11Buffer*cbuf=nullptr;{D3D11_BUFFER_DESC cbd{};cbd.ByteWidth=sizeof(CB);cbd.Usage=D3D11_USAGE_DEFAULT;cbd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;D3D11_SUBRESOURCE_DATA sd{};sd.pSysMem=&cb;dev->CreateBuffer(&cbd,&sd,&cbuf);}

    ID3D11UnorderedAccessView*uavs[2]={uavOut,uavP};
    ctx->CSSetShader(cs,nullptr,0);ctx->CSSetShaderResources(0,1,&srvQ);ctx->CSSetUnorderedAccessViews(0,2,uavs,nullptr);ctx->CSSetConstantBuffers(0,1,&cbuf);
    ctx->Dispatch(H,1,1);   // one group per head; threads = query positions

    std::vector<float> out(S*E);
    ID3D11Buffer*bStg=nullptr;{D3D11_BUFFER_DESC stg{};stg.ByteWidth=S*E*4;stg.Usage=D3D11_USAGE_STAGING;stg.CPUAccessFlags=D3D11_CPU_ACCESS_READ;dev->CreateBuffer(&stg,nullptr,&bStg);}
    ctx->CopyResource(bStg,bOut);D3D11_MAPPED_SUBRESOURCE ms;ctx->Map(bStg,0,D3D11_MAP_READ,0,&ms);memcpy(out.data(),ms.pData,S*E*4);ctx->Unmap(bStg,0);

    double maxAbs=0,absmax=0; size_t worst=0;
    for(size_t i=0;i<(size_t)S*E;i++){double a=std::fabs((double)out[i]-ref[i]);if(a>maxAbs){maxAbs=a;worst=i;}if(std::fabs((double)ref[i])>absmax)absmax=std::fabs((double)ref[i]);}
    double norm=maxAbs/absmax;
    printf("[out row0 0..2] gpu=(%.5f,%.5f,%.5f) ref=(%.5f,%.5f,%.5f)\n",out[0],out[1],out[2],ref[0],ref[1],ref[2]);
    printf("[worst @ %zu] gpu=%.6f ref=%.6f (abs %.2e)\n",worst,out[worst],ref[worst],maxAbs);
    printf("[verify] max abs err=%.3e  scale=|ref|max=%.3f  scale-normalized err=%.2e\n\n",maxAbs,absmax,norm);
    bool pass=norm<1e-5;
    printf("=== %s: KHANARY G_ATTENTION cs_5_0 causal MHA on real gpt2 qkv, dispatched on HD4600 -> %s vs numpy ===\n",pass?"PASS":"FAIL",pass?"matches (fp32-accurate)":"MISMATCH");
    return pass?0:1;
}
