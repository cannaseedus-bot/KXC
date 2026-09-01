// xform_run.cpp — dispatch KHANARY's D3D11 backend vertex-transform kernel on the HD 4600,
// verify vs CPU (q = M * [p,1]). Proves the backend emits a REAL geometry kernel that RUNS.
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
int main(){
    const UINT N=256;
    std::string hlsl=readFile("vertex_transform.hlsl");
    if(hlsl.empty()){printf("no hlsl\n");return 1;}
    D3D_FEATURE_LEVEL fl{}; ID3D11Device*dev=nullptr; ID3D11DeviceContext*ctx=nullptr;
    D3D_FEATURE_LEVEL want[]={D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0};
    if(FAILED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,want,2,D3D11_SDK_VERSION,&dev,&fl,&ctx))){printf("no d3d11\n");return 1;}
    printf("[dev] D3D11 FL 0x%x\n",(int)fl);
    ID3DBlob*blob=nullptr;ID3DBlob*err=nullptr;
    if(FAILED(D3DCompile(hlsl.data(),hlsl.size(),"vt",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&err))){if(err)printf("HLSL err: %s\n",(char*)err->GetBufferPointer());return 1;}
    ID3D11ComputeShader*cs=nullptr;dev->CreateComputeShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&cs);
    // input positions (tight float3): p_i=(i, i+0.5, i+1)
    std::vector<float> in(3*N),out(3*N),ref(3*N);
    for(UINT i=0;i<N;i++){in[i*3+0]=(float)i;in[i*3+1]=i+0.5f;in[i*3+2]=i+1.0f;}
    // M row-major: q=(2px+1, 3py+2, 4pz+3)
    for(UINT i=0;i<N;i++){ref[i*3+0]=2*in[i*3+0]+1;ref[i*3+1]=3*in[i*3+1]+2;ref[i*3+2]=4*in[i*3+2]+3;}
    auto rawBuf=[&](const void*data,UINT bytes,bool uav)->ID3D11Buffer*{
        D3D11_BUFFER_DESC bd{};bd.ByteWidth=bytes;bd.Usage=D3D11_USAGE_DEFAULT;
        bd.BindFlags=uav?D3D11_BIND_UNORDERED_ACCESS:D3D11_BIND_SHADER_RESOURCE;
        bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        D3D11_SUBRESOURCE_DATA sd{};sd.pSysMem=data;ID3D11Buffer*b=nullptr;dev->CreateBuffer(&bd,data?&sd:nullptr,&b);return b;};
    ID3D11Buffer*bIn=rawBuf(in.data(),3*N*4,false);
    ID3D11Buffer*bOut=rawBuf(nullptr,3*N*4,true);
    ID3D11ShaderResourceView*srv=nullptr;
    D3D11_SHADER_RESOURCE_VIEW_DESC sv{};sv.Format=DXGI_FORMAT_R32_TYPELESS;sv.ViewDimension=D3D11_SRV_DIMENSION_BUFFEREX;sv.BufferEx.NumElements=3*N;sv.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW;dev->CreateShaderResourceView(bIn,&sv,&srv);
    ID3D11UnorderedAccessView*uav=nullptr;
    D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};ud.Format=DXGI_FORMAT_R32_TYPELESS;ud.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;ud.Buffer.NumElements=3*N;ud.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;dev->CreateUnorderedAccessView(bOut,&ud,&uav);
    struct CB{float M[16];uint32_t count;uint32_t pad[3];} cb{};
    float Mrow[16]={2,0,0,1, 0,3,0,2, 0,0,4,3, 0,0,0,1}; for(int i=0;i<16;i++)cb.M[i]=Mrow[i]; cb.count=N;
    D3D11_BUFFER_DESC cbd{};cbd.ByteWidth=sizeof(CB);cbd.Usage=D3D11_USAGE_DEFAULT;cbd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA cs0{};cs0.pSysMem=&cb;ID3D11Buffer*cbuf=nullptr;dev->CreateBuffer(&cbd,&cs0,&cbuf);
    D3D11_BUFFER_DESC stg{};stg.ByteWidth=3*N*4;stg.Usage=D3D11_USAGE_STAGING;stg.CPUAccessFlags=D3D11_CPU_ACCESS_READ;ID3D11Buffer*bStg=nullptr;dev->CreateBuffer(&stg,nullptr,&bStg);
    ctx->CSSetShader(cs,nullptr,0);ctx->CSSetShaderResources(0,1,&srv);ctx->CSSetUnorderedAccessViews(0,1,&uav,nullptr);ctx->CSSetConstantBuffers(0,1,&cbuf);
    ctx->Dispatch((N+63)/64,1,1);
    ctx->CopyResource(bStg,bOut);D3D11_MAPPED_SUBRESOURCE ms;ctx->Map(bStg,0,D3D11_MAP_READ,0,&ms);memcpy(out.data(),ms.pData,3*N*4);ctx->Unmap(bStg,0);
    double maxAbs=0;for(UINT i=0;i<3*N;i++)maxAbs=std::max(maxAbs,std::fabs((double)out[i]-ref[i]));
    printf("[in]  v0=(%.1f,%.1f,%.1f) v1=(%.1f,%.1f,%.1f)\n",in[0],in[1],in[2],in[3],in[4],in[5]);
    printf("[out] v0=(%.1f,%.1f,%.1f) v1=(%.1f,%.1f,%.1f)\n",out[0],out[1],out[2],out[3],out[4],out[5]);
    printf("[ref] v0=(%.1f,%.1f,%.1f) v1=(%.1f,%.1f,%.1f)\n",ref[0],ref[1],ref[2],ref[3],ref[4],ref[5]);
    printf("[verify] max abs err=%.2e\n\n",maxAbs);
    bool pass=maxAbs<1e-4;
    printf("=== %s: KHANARY D3D11 vertex-transform kernel dispatched on HD4600 -> %s vs CPU ===\n",pass?"PASS":"FAIL",pass?"correct":"MISMATCH");
    return pass?0:1;
}
