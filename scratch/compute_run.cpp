// compute_run.cpp <layernorm|gelu|embed> — dispatch a promoted KHANARY compute glyph on the
// HD 4600, verify vs numpy f64. One harness, op selected by argv[1].
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
static std::vector<float> readF(const char*p,size_t n){std::vector<float> v(n);std::ifstream f(p,std::ios::binary);f.read((char*)v.data(),n*4);return v;}
static std::vector<int32_t> readI(const char*p,size_t n){std::vector<int32_t> v(n);std::ifstream f(p,std::ios::binary);f.read((char*)v.data(),n*4);return v;}

static ID3D11Device* dev; static ID3D11DeviceContext* ctx;
static ID3D11ShaderResourceView* srv(const void*d,UINT count,UINT stride){
    D3D11_BUFFER_DESC bd{};bd.ByteWidth=count*stride;bd.Usage=D3D11_USAGE_DEFAULT;bd.BindFlags=D3D11_BIND_SHADER_RESOURCE;bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;bd.StructureByteStride=stride;
    D3D11_SUBRESOURCE_DATA sd{};sd.pSysMem=d;ID3D11Buffer*b=nullptr;dev->CreateBuffer(&bd,&sd,&b);
    D3D11_SHADER_RESOURCE_VIEW_DESC v{};v.Format=DXGI_FORMAT_UNKNOWN;v.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;v.Buffer.NumElements=count;
    ID3D11ShaderResourceView*s=nullptr;dev->CreateShaderResourceView(b,&v,&s);return s;}
static ID3D11Buffer* rwbuf(UINT count){D3D11_BUFFER_DESC bd{};bd.ByteWidth=count*4;bd.Usage=D3D11_USAGE_DEFAULT;bd.BindFlags=D3D11_BIND_UNORDERED_ACCESS;bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;bd.StructureByteStride=4;ID3D11Buffer*b=nullptr;dev->CreateBuffer(&bd,nullptr,&b);return b;}
static ID3D11UnorderedAccessView* uav(ID3D11Buffer*b,UINT count){D3D11_UNORDERED_ACCESS_VIEW_DESC u{};u.Format=DXGI_FORMAT_UNKNOWN;u.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;u.Buffer.NumElements=count;ID3D11UnorderedAccessView*v=nullptr;dev->CreateUnorderedAccessView(b,&u,&v);return v;}
static ID3D11Buffer* cbuf(const void*d,UINT bytes){D3D11_BUFFER_DESC bd{};bd.ByteWidth=(bytes+15)&~15u;bd.Usage=D3D11_USAGE_DEFAULT;bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;D3D11_SUBRESOURCE_DATA sd{};sd.pSysMem=d;ID3D11Buffer*b=nullptr;dev->CreateBuffer(&bd,&sd,&b);return b;}
static ID3D11ComputeShader* compile(const std::string&hlsl){ID3DBlob*bl=nullptr;ID3DBlob*er=nullptr;if(FAILED(D3DCompile(hlsl.data(),hlsl.size(),"k",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&bl,&er))){if(er)printf("HLSL err: %s\n",(char*)er->GetBufferPointer());return nullptr;}ID3D11ComputeShader*cs=nullptr;dev->CreateComputeShader(bl->GetBufferPointer(),bl->GetBufferSize(),nullptr,&cs);return cs;}
static void readback(ID3D11Buffer*b,float*out,UINT n){D3D11_BUFFER_DESC bd{};bd.ByteWidth=n*4;bd.Usage=D3D11_USAGE_STAGING;bd.CPUAccessFlags=D3D11_CPU_ACCESS_READ;ID3D11Buffer*st=nullptr;dev->CreateBuffer(&bd,nullptr,&st);ctx->CopyResource(st,b);D3D11_MAPPED_SUBRESOURCE m;ctx->Map(st,0,D3D11_MAP_READ,0,&m);memcpy(out,m.pData,n*4);ctx->Unmap(st,0);}
static int verdict(const char*op,const std::vector<float>&o,const std::vector<float>&r){
    double mx=0,am=0;size_t w=0;for(size_t i=0;i<o.size();i++){double a=std::fabs((double)o[i]-r[i]);if(a>mx){mx=a;w=i;}if(std::fabs((double)r[i])>am)am=std::fabs((double)r[i]);}
    double nm=mx/(am+1e-9);printf("[%s] out[0..2]=(%.5f,%.5f,%.5f) ref=(%.5f,%.5f,%.5f)\n",op,o[0],o[1],o[2],r[0],r[1],r[2]);
    printf("[%s] max abs err=%.3e scale=%.3f norm=%.2e\n",op,mx,am,nm);bool p=nm<1e-5;
    printf("=== %s: KHANARY G_%s cs_5_0 on HD4600 -> %s vs numpy ===\n",p?"PASS":"FAIL",op,p?"matches":"MISMATCH");return p?0:1;}

int main(int argc,char**argv){
    std::string op=argc>1?argv[1]:"layernorm";
    D3D_FEATURE_LEVEL fl{};D3D_FEATURE_LEVEL want[]={D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0};
    if(FAILED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,want,2,D3D11_SDK_VERSION,&dev,&fl,&ctx))){printf("no d3d11\n");return 1;}
    printf("[dev] D3D11 FL 0x%x  op=%s\n",(int)fl,op.c_str());

    if(op=="layernorm"){
        UINT S,E;float eps;{std::stringstream d(readFile("ln_dims.txt"));d>>S>>E>>eps;}
        auto x=readF("ln_x.bin",(size_t)S*E),g=readF("ln_gamma.bin",E),b=readF("ln_beta.bin",E),ref=readF("ln_ref.bin",(size_t)S*E);
        auto cs=compile(readFile("knu_layernorm.hlsl"));if(!cs)return 1;
        ID3D11ShaderResourceView*sx=srv(x.data(),S*E,4),*sg=srv(g.data(),E,4),*sb=srv(b.data(),E,4);
        ID3D11Buffer*by=rwbuf(S*E),*bxh=rwbuf(S*E),*bis=rwbuf(S);
        ID3D11UnorderedAccessView*uy=uav(by,S*E),*uxh=uav(bxh,S*E),*uis=uav(bis,S);
        struct{uint32_t n_embd,seq_len;float eps;uint32_t pad;}cb{E,S,eps,0};ID3D11Buffer*cbf=cbuf(&cb,sizeof cb);
        ID3D11ShaderResourceView*ss[3]={sx,sg,sb};ID3D11UnorderedAccessView*us[3]={uy,uxh,uis};
        ctx->CSSetShader(cs,nullptr,0);ctx->CSSetShaderResources(0,3,ss);ctx->CSSetUnorderedAccessViews(0,3,us,nullptr);ctx->CSSetConstantBuffers(0,1,&cbf);
        ctx->Dispatch(S,1,1);std::vector<float> o(S*E);readback(by,o.data(),S*E);return verdict("LAYERNORM",o,ref);
    }
    if(op=="gelu"){
        UINT N;{std::stringstream d(readFile("gelu_dims.txt"));d>>N;}
        auto x=readF("gelu_x.bin",N),ref=readF("gelu_ref.bin",N);
        auto cs=compile(readFile("knu_gelu.hlsl"));if(!cs)return 1;
        ID3D11ShaderResourceView*sx=srv(x.data(),N,4);ID3D11Buffer*by=rwbuf(N);ID3D11UnorderedAccessView*uy=uav(by,N);
        struct{uint32_t numel,off;uint32_t pad[2];}cb{N,0,{0,0}};ID3D11Buffer*cbf=cbuf(&cb,sizeof cb);
        ctx->CSSetShader(cs,nullptr,0);ctx->CSSetShaderResources(0,1,&sx);ctx->CSSetUnorderedAccessViews(0,1,&uy,nullptr);ctx->CSSetConstantBuffers(0,1,&cbf);
        ctx->Dispatch((N+255)/256,1,1);std::vector<float> o(N);readback(by,o.data(),N);return verdict("GELU",o,ref);
    }
    if(op=="embed"){
        UINT S,E,V,C;{std::stringstream d(readFile("emb_dims.txt"));d>>S>>E>>V>>C;}
        auto tok=readI("emb_tokens.bin",S);auto wte=readF("emb_wte.bin",(size_t)V*E),wpe=readF("emb_wpe.bin",(size_t)C*E),ref=readF("emb_ref.bin",(size_t)S*E);
        auto cs=compile(readFile("knu_embed.hlsl"));if(!cs)return 1;
        ID3D11ShaderResourceView*st=srv(tok.data(),S,4),*sw=srv(wte.data(),V*E,4),*sp=srv(wpe.data(),C*E,4);
        ID3D11Buffer*bh=rwbuf(S*E);ID3D11UnorderedAccessView*uh=uav(bh,S*E);
        struct{uint32_t seq_len,n_embd;uint32_t pad[2];}cb{S,E,{0,0}};ID3D11Buffer*cbf=cbuf(&cb,sizeof cb);
        ID3D11ShaderResourceView*ss[3]={st,sw,sp};
        ctx->CSSetShader(cs,nullptr,0);ctx->CSSetShaderResources(0,3,ss);ctx->CSSetUnorderedAccessViews(0,1,&uh,nullptr);ctx->CSSetConstantBuffers(0,1,&cbf);
        ctx->Dispatch(S,1,1);std::vector<float> o(S*E);readback(bh,o.data(),S*E);return verdict("EMBED",o,ref);
    }
    printf("unknown op\n");return 1;
}
