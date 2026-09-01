// skin_run.cpp — dispatch KHANARY's D3D11 skinning kernel (G_VERTEX_SKIN) on the HD 4600,
// verify pos+normal bit-exact vs CPU. Proves the richer geometry op RUNS, not just compiles.
//
// The kernel uses StructuredBuffer<float4x4> (no row_major), so HLSL loads each matrix
// column-major. The CPU ref therefore mirrors mul(m, v) exactly as: out[i] = sum_j f[j*4+i]*v[j],
// i.e. affine col c of the 3x3 lives at f[c*4+i], translation at f[12+i]. Blend is linear:
// m = w.x*M[jx] + w.y*M[jy] + w.z*M[jz] + w.w*M[jw].
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
    const UINT N=128;         // vertices
    const UINT J=2;           // joints
    std::string hlsl=readFile("knu_skin.hlsl");   // the KNU-stream-lowered skinning kernel
    if(hlsl.empty()){printf("no hlsl (run the python emit first)\n");return 1;}
    D3D_FEATURE_LEVEL fl{}; ID3D11Device*dev=nullptr; ID3D11DeviceContext*ctx=nullptr;
    D3D_FEATURE_LEVEL want[]={D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0};
    if(FAILED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,want,2,D3D11_SDK_VERSION,&dev,&fl,&ctx))){printf("no d3d11\n");return 1;}
    printf("[dev] D3D11 FL 0x%x\n",(int)fl);
    ID3DBlob*blob=nullptr;ID3DBlob*err=nullptr;
    if(FAILED(D3DCompile(hlsl.data(),hlsl.size(),"skin",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&err))){if(err)printf("HLSL err: %s\n",(char*)err->GetBufferPointer());return 1;}
    ID3D11ComputeShader*cs=nullptr;dev->CreateComputeShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&cs);

    // --- two skin matrices, stored as flat 16-float arrays interpreted COLUMN-major by HLSL ---
    // M0: scale 2 + translate (1,2,3);  M1: scale 3 + translate (-1,-2,-3).
    // column-major flat: f[c*4 + r]. 3x3 diag at f[0],f[5],f[10]; translation at f[12],f[13],f[14]; f[15]=1.
    auto affine=[&](float s,float tx,float ty,float tz,float*f){
        for(int i=0;i<16;i++)f[i]=0.0f;
        f[0]=s; f[5]=s; f[10]=s;          // 3x3 scale on the diagonal
        f[12]=tx; f[13]=ty; f[14]=tz;     // translation column (col 3)
        f[15]=1.0f;
    };
    std::vector<float> M(J*16);
    affine(2.0f, 1.0f, 2.0f, 3.0f, &M[0]);
    affine(3.0f,-1.0f,-2.0f,-3.0f, &M[16]);

    // --- per-vertex inputs ---
    std::vector<float>    pos(3*N), nrm(3*N);
    std::vector<float>    wgt(4*N);
    std::vector<uint32_t> jnt(4*N);
    for(UINT i=0;i<N;i++){
        pos[i*3+0]=(float)i; pos[i*3+1]=i+0.5f; pos[i*3+2]=i+1.0f;
        nrm[i*3+0]=0.0f;     nrm[i*3+1]=1.0f;   nrm[i*3+2]=0.0f;
        float w0=(float)(i%5)/4.0f, w1=1.0f-w0;  // blend across joints, sums to 1
        wgt[i*4+0]=w0; wgt[i*4+1]=w1; wgt[i*4+2]=0.0f; wgt[i*4+3]=0.0f;
        jnt[i*4+0]=0;  jnt[i*4+1]=1;  jnt[i*4+2]=0;    jnt[i*4+3]=0;
    }

    // --- CPU reference (mirror col-major mul) ---
    std::vector<float> ref(6*N); // [pos3, nrm3] per vertex, matching outVerts 24B stride
    for(UINT i=0;i<N;i++){
        float w[4]={wgt[i*4+0],wgt[i*4+1],wgt[i*4+2],wgt[i*4+3]};
        uint32_t jj[4]={jnt[i*4+0],jnt[i*4+1],jnt[i*4+2],jnt[i*4+3]};
        float m[16]; for(int k=0;k<16;k++)m[k]=w[0]*M[jj[0]*16+k]+w[1]*M[jj[1]*16+k]+w[2]*M[jj[2]*16+k]+w[3]*M[jj[3]*16+k];
        float v[4]={pos[i*3+0],pos[i*3+1],pos[i*3+2],1.0f};
        for(int r=0;r<3;r++){ float acc=0; for(int c=0;c<4;c++)acc+=m[c*4+r]*v[c]; ref[i*6+r]=acc; }        // position
        float nv[3]={nrm[i*3+0],nrm[i*3+1],nrm[i*3+2]};
        for(int r=0;r<3;r++){ float acc=0; for(int c=0;c<3;c++)acc+=m[c*4+r]*nv[c]; ref[i*6+3+r]=acc; }      // normal (3x3)
    }

    // --- GPU buffers ---
    auto rawSRV=[&](const void*d,UINT bytes)->ID3D11ShaderResourceView*{
        D3D11_BUFFER_DESC bd{};bd.ByteWidth=bytes;bd.Usage=D3D11_USAGE_DEFAULT;bd.BindFlags=D3D11_BIND_SHADER_RESOURCE;bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        D3D11_SUBRESOURCE_DATA sd{};sd.pSysMem=d;ID3D11Buffer*b=nullptr;dev->CreateBuffer(&bd,&sd,&b);
        D3D11_SHADER_RESOURCE_VIEW_DESC sv{};sv.Format=DXGI_FORMAT_R32_TYPELESS;sv.ViewDimension=D3D11_SRV_DIMENSION_BUFFEREX;sv.BufferEx.NumElements=bytes/4;sv.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW;
        ID3D11ShaderResourceView*srv=nullptr;dev->CreateShaderResourceView(b,&sv,&srv);return srv;};
    ID3D11ShaderResourceView*srvPos=rawSRV(pos.data(),3*N*4);
    ID3D11ShaderResourceView*srvNrm=rawSRV(nrm.data(),3*N*4);
    ID3D11ShaderResourceView*srvWgt=rawSRV(wgt.data(),4*N*4);
    // joints: typed Buffer<uint4>
    ID3D11ShaderResourceView*srvJnt=nullptr;{
        D3D11_BUFFER_DESC bd{};bd.ByteWidth=4*N*4;bd.Usage=D3D11_USAGE_DEFAULT;bd.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA sd{};sd.pSysMem=jnt.data();ID3D11Buffer*b=nullptr;dev->CreateBuffer(&bd,&sd,&b);
        D3D11_SHADER_RESOURCE_VIEW_DESC sv{};sv.Format=DXGI_FORMAT_R32G32B32A32_UINT;sv.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;sv.Buffer.NumElements=N;dev->CreateShaderResourceView(b,&sv,&srvJnt);}
    // skinMatrices: StructuredBuffer<float4x4>, stride 64
    ID3D11ShaderResourceView*srvMat=nullptr;{
        D3D11_BUFFER_DESC bd{};bd.ByteWidth=J*64;bd.Usage=D3D11_USAGE_DEFAULT;bd.BindFlags=D3D11_BIND_SHADER_RESOURCE;bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;bd.StructureByteStride=64;
        D3D11_SUBRESOURCE_DATA sd{};sd.pSysMem=M.data();ID3D11Buffer*b=nullptr;dev->CreateBuffer(&bd,&sd,&b);
        D3D11_SHADER_RESOURCE_VIEW_DESC sv{};sv.Format=DXGI_FORMAT_UNKNOWN;sv.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;sv.Buffer.NumElements=J;dev->CreateShaderResourceView(b,&sv,&srvMat);}
    // output: RWByteAddressBuffer, 24 B/vertex
    ID3D11Buffer*bOut=nullptr;{
        D3D11_BUFFER_DESC bd{};bd.ByteWidth=6*N*4;bd.Usage=D3D11_USAGE_DEFAULT;bd.BindFlags=D3D11_BIND_UNORDERED_ACCESS;bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;dev->CreateBuffer(&bd,nullptr,&bOut);}
    ID3D11UnorderedAccessView*uav=nullptr;{
        D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};ud.Format=DXGI_FORMAT_R32_TYPELESS;ud.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;ud.Buffer.NumElements=6*N;ud.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;dev->CreateUnorderedAccessView(bOut,&ud,&uav);}
    // cbuffer VertexUniforms { posStride,posOff,nrmStride,nrmOff,vertexCount, float3 pad } = 32 B
    struct CB{uint32_t posStride,posOff,nrmStride,nrmOff,vertexCount,pad[3];} cb{3,0,3,0,N,{0,0,0}};
    ID3D11Buffer*cbuf=nullptr;{D3D11_BUFFER_DESC cbd{};cbd.ByteWidth=sizeof(CB);cbd.Usage=D3D11_USAGE_DEFAULT;cbd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;D3D11_SUBRESOURCE_DATA sd{};sd.pSysMem=&cb;dev->CreateBuffer(&cbd,&sd,&cbuf);}

    ID3D11ShaderResourceView*srvs[5]={srvPos,srvNrm,srvWgt,srvJnt,srvMat};
    ctx->CSSetShader(cs,nullptr,0);
    ctx->CSSetShaderResources(0,5,srvs);
    ctx->CSSetUnorderedAccessViews(0,1,&uav,nullptr);
    ctx->CSSetConstantBuffers(0,1,&cbuf);
    ctx->Dispatch((N+63)/64,1,1);

    std::vector<float> out(6*N);
    ID3D11Buffer*bStg=nullptr;{D3D11_BUFFER_DESC stg{};stg.ByteWidth=6*N*4;stg.Usage=D3D11_USAGE_STAGING;stg.CPUAccessFlags=D3D11_CPU_ACCESS_READ;dev->CreateBuffer(&stg,nullptr,&bStg);}
    ctx->CopyResource(bStg,bOut);D3D11_MAPPED_SUBRESOURCE ms;ctx->Map(bStg,0,D3D11_MAP_READ,0,&ms);memcpy(out.data(),ms.pData,6*N*4);ctx->Unmap(bStg,0);

    double maxAbs=0;for(UINT i=0;i<6*N;i++)maxAbs=std::max(maxAbs,std::fabs((double)out[i]-ref[i]));
    printf("[v0]  pos out=(%.3f,%.3f,%.3f) ref=(%.3f,%.3f,%.3f)\n",out[0],out[1],out[2],ref[0],ref[1],ref[2]);
    printf("[v3]  pos out=(%.3f,%.3f,%.3f) ref=(%.3f,%.3f,%.3f)  nrm out=(%.3f,%.3f,%.3f)\n",out[18],out[19],out[20],ref[18],ref[19],ref[20],out[21],out[22],out[23]);
    printf("[verify] max abs err=%.2e\n\n",maxAbs);
    bool pass=maxAbs<1e-4;
    printf("=== %s: KHANARY D3D11 skinning kernel dispatched on HD4600 -> %s vs CPU ===\n",pass?"PASS":"FAIL",pass?"correct (pos+normal)":"MISMATCH");
    return pass?0:1;
}
