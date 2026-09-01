// brain_xform_run.cpp — (c) the payoff: transform brain2's REAL birdsong mesh vertices
// (nodes = spectrogram ridge points, from the canary song -> khanary_brain.stb) on the HD 4600
// through KHANARY's KNU-glyph-driven vertex-transform kernel. Verify bit-exact vs CPU.
// Closes the loop: birdsong -> graph -> .stb -> GPU geometry, all on this iGPU.
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
    UINT N=(UINT)atoi(readFile("brain_N.txt").c_str());
    if(!N){printf("no brain_N\n");return 1;}
    std::string hlsl=readFile("knu_xform.hlsl");   // KNU-stream-lowered transform kernel
    if(hlsl.empty()){printf("no hlsl\n");return 1;}
    // load the real mesh vertices
    std::vector<float> in(3*N);
    { std::ifstream f("brain_verts.bin",std::ios::binary); f.read((char*)in.data(),3*N*4); }

    D3D_FEATURE_LEVEL fl{}; ID3D11Device*dev=nullptr; ID3D11DeviceContext*ctx=nullptr;
    D3D_FEATURE_LEVEL want[]={D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0};
    if(FAILED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,want,2,D3D11_SDK_VERSION,&dev,&fl,&ctx))){printf("no d3d11\n");return 1;}
    printf("[dev] D3D11 FL 0x%x  brain2 mesh N=%u vertices\n",(int)fl,N);
    ID3DBlob*blob=nullptr;ID3DBlob*err=nullptr;
    if(FAILED(D3DCompile(hlsl.data(),hlsl.size(),"bx",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&err))){if(err)printf("HLSL err: %s\n",(char*)err->GetBufferPointer());return 1;}
    ID3D11ComputeShader*cs=nullptr;dev->CreateComputeShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&cs);

    // Mesh normalization transform (row_major, as the kernel declares M): spectrogram axes are
    // time~[0,~1400], freq~[0,~150]; map to a normalized model box + recenter.
    float Mrow[16]={ 1.0f/1400, 0, 0, -0.5f,
                     0, 1.0f/150, 0, -0.5f,
                     0, 0, 1.0f,   0.0f,
                     0, 0, 0,      1.0f };
    std::vector<float> ref(3*N);
    for(UINT i=0;i<N;i++){
        float p[3]={in[i*3+0],in[i*3+1],in[i*3+2]};
        for(int r=0;r<3;r++) ref[i*3+r]=Mrow[r*4+0]*p[0]+Mrow[r*4+1]*p[1]+Mrow[r*4+2]*p[2]+Mrow[r*4+3];
    }

    auto rawBuf=[&](const void*data,UINT bytes,bool uav)->ID3D11Buffer*{
        D3D11_BUFFER_DESC bd{};bd.ByteWidth=bytes;bd.Usage=D3D11_USAGE_DEFAULT;
        bd.BindFlags=uav?D3D11_BIND_UNORDERED_ACCESS:D3D11_BIND_SHADER_RESOURCE;bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        D3D11_SUBRESOURCE_DATA sd{};sd.pSysMem=data;ID3D11Buffer*b=nullptr;dev->CreateBuffer(&bd,data?&sd:nullptr,&b);return b;};
    ID3D11Buffer*bIn=rawBuf(in.data(),3*N*4,false);
    ID3D11Buffer*bOut=rawBuf(nullptr,3*N*4,true);
    ID3D11ShaderResourceView*srv=nullptr;
    {D3D11_SHADER_RESOURCE_VIEW_DESC sv{};sv.Format=DXGI_FORMAT_R32_TYPELESS;sv.ViewDimension=D3D11_SRV_DIMENSION_BUFFEREX;sv.BufferEx.NumElements=3*N;sv.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW;dev->CreateShaderResourceView(bIn,&sv,&srv);}
    ID3D11UnorderedAccessView*uav=nullptr;
    {D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};ud.Format=DXGI_FORMAT_R32_TYPELESS;ud.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;ud.Buffer.NumElements=3*N;ud.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;dev->CreateUnorderedAccessView(bOut,&ud,&uav);}
    struct CB{float M[16];uint32_t count;uint32_t pad[3];} cb{};
    for(int i=0;i<16;i++)cb.M[i]=Mrow[i]; cb.count=N;
    ID3D11Buffer*cbuf=nullptr;{D3D11_BUFFER_DESC cbd{};cbd.ByteWidth=sizeof(CB);cbd.Usage=D3D11_USAGE_DEFAULT;cbd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;D3D11_SUBRESOURCE_DATA s0{};s0.pSysMem=&cb;dev->CreateBuffer(&cbd,&s0,&cbuf);}
    ID3D11Buffer*bStg=nullptr;{D3D11_BUFFER_DESC stg{};stg.ByteWidth=3*N*4;stg.Usage=D3D11_USAGE_STAGING;stg.CPUAccessFlags=D3D11_CPU_ACCESS_READ;dev->CreateBuffer(&stg,nullptr,&bStg);}

    ctx->CSSetShader(cs,nullptr,0);ctx->CSSetShaderResources(0,1,&srv);ctx->CSSetUnorderedAccessViews(0,1,&uav,nullptr);ctx->CSSetConstantBuffers(0,1,&cbuf);
    ctx->Dispatch((N+63)/64,1,1);
    std::vector<float> out(3*N);
    ctx->CopyResource(bStg,bOut);D3D11_MAPPED_SUBRESOURCE ms;ctx->Map(bStg,0,D3D11_MAP_READ,0,&ms);memcpy(out.data(),ms.pData,3*N*4);ctx->Unmap(bStg,0);

    double maxAbs=0;for(UINT i=0;i<3*N;i++)maxAbs=std::max(maxAbs,std::fabs((double)out[i]-ref[i]));
    printf("[node0]   raw=(%.2f,%.2f,%.2f) -> gpu=(%.4f,%.4f,%.4f)\n",in[0],in[1],in[2],out[0],out[1],out[2]);
    printf("[node%u] raw=(%.2f,%.2f,%.2f) -> gpu=(%.4f,%.4f,%.4f)\n",N-1,in[(N-1)*3],in[(N-1)*3+1],in[(N-1)*3+2],out[(N-1)*3],out[(N-1)*3+1],out[(N-1)*3+2]);
    printf("[verify] %u real birdsong mesh vertices, max abs err=%.2e\n\n",N,maxAbs);
    bool pass=maxAbs<1e-4;
    printf("=== %s: KHANARY transformed brain2's REAL birdsong mesh on HD4600 -> %s vs CPU ===\n",pass?"PASS":"FAIL",pass?"bit-exact":"MISMATCH");
    return pass?0:1;
}
