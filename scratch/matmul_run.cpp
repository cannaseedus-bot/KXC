// matmul_run.cpp — dispatch KHANARY's G_MATMUL cs_5_0 GEMM on the HD 4600 with a REAL
// gpt2 weight (transformer.h.0.attn.c_attn.weight from random_gpt2.safetensors as B),
// verify C = A @ B vs a float64 numpy reference. First compute glyph beyond the copy skeleton.
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
    UINT M=0,N=0,K=0;
    { std::stringstream d(readFile("gemm_dims.txt")); d>>M>>N>>K; }
    if(!M||!N||!K){printf("no dims\n");return 1;}
    std::string hlsl=readFile("knu_matmul.hlsl");   // KNU-stream-lowered GEMM (G_MATMUL)
    if(hlsl.empty()){printf("no hlsl\n");return 1;}
    std::vector<float> A=readBin("gemm_A.bin",(size_t)M*K), B=readBin("gemm_B.bin",(size_t)K*N), Cref=readBin("gemm_Cref.bin",(size_t)M*N);

    D3D_FEATURE_LEVEL fl{}; ID3D11Device*dev=nullptr; ID3D11DeviceContext*ctx=nullptr;
    D3D_FEATURE_LEVEL want[]={D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0};
    if(FAILED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,want,2,D3D11_SDK_VERSION,&dev,&fl,&ctx))){printf("no d3d11\n");return 1;}
    printf("[dev] D3D11 FL 0x%x  GEMM C[%u,%u] = A[%u,%u] @ B[%u,%u]  (%.0fM MACs, real gpt2 QKV weight)\n",(int)fl,M,N,M,K,K,N,(double)M*N*K/1e6);
    ID3DBlob*blob=nullptr;ID3DBlob*err=nullptr;
    if(FAILED(D3DCompile(hlsl.data(),hlsl.size(),"mm",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&err))){if(err)printf("HLSL err: %s\n",(char*)err->GetBufferPointer());return 1;}
    ID3D11ComputeShader*cs=nullptr;dev->CreateComputeShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&cs);

    auto structSRV=[&](const void*data,UINT count)->ID3D11ShaderResourceView*{
        D3D11_BUFFER_DESC bd{};bd.ByteWidth=count*4;bd.Usage=D3D11_USAGE_DEFAULT;bd.BindFlags=D3D11_BIND_SHADER_RESOURCE;bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;bd.StructureByteStride=4;
        D3D11_SUBRESOURCE_DATA sd{};sd.pSysMem=data;ID3D11Buffer*b=nullptr;dev->CreateBuffer(&bd,&sd,&b);
        D3D11_SHADER_RESOURCE_VIEW_DESC sv{};sv.Format=DXGI_FORMAT_UNKNOWN;sv.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;sv.Buffer.NumElements=count;
        ID3D11ShaderResourceView*srv=nullptr;dev->CreateShaderResourceView(b,&sv,&srv);return srv;};
    ID3D11ShaderResourceView*srvA=structSRV(A.data(),M*K);
    ID3D11ShaderResourceView*srvB=structSRV(B.data(),K*N);
    // output C: structured RW
    ID3D11Buffer*bC=nullptr;{D3D11_BUFFER_DESC bd{};bd.ByteWidth=M*N*4;bd.Usage=D3D11_USAGE_DEFAULT;bd.BindFlags=D3D11_BIND_UNORDERED_ACCESS;bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;bd.StructureByteStride=4;dev->CreateBuffer(&bd,nullptr,&bC);}
    ID3D11UnorderedAccessView*uav=nullptr;{D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};ud.Format=DXGI_FORMAT_UNKNOWN;ud.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;ud.Buffer.NumElements=M*N;dev->CreateUnorderedAccessView(bC,&ud,&uav);}
    struct CB{uint32_t M,N,K,pad;} cb{M,N,K,0};
    ID3D11Buffer*cbuf=nullptr;{D3D11_BUFFER_DESC cbd{};cbd.ByteWidth=sizeof(CB);cbd.Usage=D3D11_USAGE_DEFAULT;cbd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;D3D11_SUBRESOURCE_DATA sd{};sd.pSysMem=&cb;dev->CreateBuffer(&cbd,&sd,&cbuf);}

    ID3D11ShaderResourceView*srvs[2]={srvA,srvB};
    ctx->CSSetShader(cs,nullptr,0);ctx->CSSetShaderResources(0,2,srvs);ctx->CSSetUnorderedAccessViews(0,1,&uav,nullptr);ctx->CSSetConstantBuffers(0,1,&cbuf);
    ctx->Dispatch((N+15)/16,(M+15)/16,1);

    std::vector<float> C(M*N);
    ID3D11Buffer*bStg=nullptr;{D3D11_BUFFER_DESC stg{};stg.ByteWidth=M*N*4;stg.Usage=D3D11_USAGE_STAGING;stg.CPUAccessFlags=D3D11_CPU_ACCESS_READ;dev->CreateBuffer(&stg,nullptr,&bStg);}
    ctx->CopyResource(bStg,bC);D3D11_MAPPED_SUBRESOURCE ms;ctx->Map(bStg,0,D3D11_MAP_READ,0,&ms);memcpy(C.data(),ms.pData,M*N*4);ctx->Unmap(bStg,0);

    // --- timing A/B: naive one-thread-per-output vs the tiled (groupshared) kernel ---
    std::string NAIVE=
      "StructuredBuffer<float> A:register(t0);StructuredBuffer<float> B:register(t1);RWStructuredBuffer<float> C:register(u0);\n"
      "cbuffer G:register(b0){uint M;uint N;uint K;uint p;};\n"
      "[numthreads(16,16,1)] void main(uint3 t:SV_DispatchThreadID){uint r=t.y,c=t.x; if(r>=M||c>=N)return; float a=0; for(uint k=0;k<K;++k)a+=A[r*K+k]*B[k*N+c]; C[r*N+c]=a;}\n";
    ID3DBlob*nb=nullptr,*ne=nullptr; D3DCompile(NAIVE.data(),NAIVE.size(),"n",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&nb,&ne);
    ID3D11ComputeShader*csN=nullptr; dev->CreateComputeShader(nb->GetBufferPointer(),nb->GetBufferSize(),nullptr,&csN);
    auto timeIt=[&](ID3D11ComputeShader*sh,int iters)->double{
        ctx->CSSetShader(sh,nullptr,0);ctx->CSSetShaderResources(0,2,srvs);ctx->CSSetUnorderedAccessViews(0,1,&uav,nullptr);ctx->CSSetConstantBuffers(0,1,&cbuf);
        ctx->Dispatch((N+15)/16,(M+15)/16,1);ctx->CopyResource(bStg,bC);{D3D11_MAPPED_SUBRESOURCE m;ctx->Map(bStg,0,D3D11_MAP_READ,0,&m);ctx->Unmap(bStg,0);} // warmup+flush
        LARGE_INTEGER f,a,b;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&a);
        for(int i=0;i<iters;i++) ctx->Dispatch((N+15)/16,(M+15)/16,1);
        ctx->CopyResource(bStg,bC);{D3D11_MAPPED_SUBRESOURCE m;ctx->Map(bStg,0,D3D11_MAP_READ,0,&m);ctx->Unmap(bStg,0);}
        QueryPerformanceCounter(&b);return (double)(b.QuadPart-a.QuadPart)*1000.0/f.QuadPart/iters; };
    int IT=100; double tTiled=timeIt(cs,IT), tNaive=timeIt(csN,IT);
    printf("[perf] GEMM %ux%ux%u  naive %.3f ms  tiled %.3f ms  speedup %.2fx (%d iters)\n",M,K,N,tNaive,tTiled,tNaive/tTiled,IT);

    // Metric: error normalized to the tensor scale (max|ref|). Per-element relative error is
    // meaningless here — many outputs are ~0, so dividing a tiny abs error by them blows up.
    double maxAbs=0,absmax=0; size_t worst=0;
    for(size_t i=0;i<(size_t)M*N;i++){
        double a=std::fabs((double)C[i]-Cref[i]);
        if(a>maxAbs){maxAbs=a;worst=i;}
        if(std::fabs((double)Cref[i])>absmax)absmax=std::fabs((double)Cref[i]);
    }
    double normErr=maxAbs/absmax;
    printf("[C[0][0..2]] gpu=(%.5f,%.5f,%.5f) ref=(%.5f,%.5f,%.5f)\n",C[0],C[1],C[2],Cref[0],Cref[1],Cref[2]);
    printf("[worst @ %zu] gpu=%.6f ref=%.6f  (abs %.2e)\n",worst,C[worst],Cref[worst],maxAbs);
    printf("[verify] max abs err=%.3e  scale=|ref|max=%.3f  scale-normalized err=%.2e\n\n",maxAbs,absmax,normErr);
    bool pass=normErr<1e-5;   // fp32 GEMM (K=768 accumulation) vs f64 reference
    printf("=== %s: KHANARY G_MATMUL cs_5_0 GEMM on real gpt2 weight, dispatched on HD4600 -> %s vs numpy ===\n",pass?"PASS":"FAIL",pass?"matches (fp32-accurate)":"MISMATCH");
    return pass?0:1;
}
