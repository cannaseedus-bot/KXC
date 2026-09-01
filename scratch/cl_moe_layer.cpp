// cl_moe_layer.cpp — full OpenCL MoE layer on the HD 4600.
// Kernels: MXFP4 dequant (top-4 expert planes) + router (top-4) + expert matvec
// + weighted merge. Runs on real gpt-oss-20b layer-0 data, verified vs CPU.
// OpenCL setup is done FIRST (before model loading) to avoid a driver crash.
// Build (MSVC): cl /O2 cl_moe_layer.cpp /Fe:cl_moe_layer.exe
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>

// ── OpenCL dynamic load ────────────────────────────────────────────────────
typedef int cl_int; typedef unsigned int cl_uint; typedef void* cl_platform_id;
typedef void* cl_device_id; typedef void* cl_context; typedef void* cl_command_queue;
typedef void* cl_program; typedef void* cl_kernel; typedef void* cl_mem; typedef size_t cl_size_t;
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
#define CL_MEM_READ_ONLY 4
#define CL_MEM_WRITE_ONLY 2
#ifndef CL_API_CALL
#define CL_API_CALL __stdcall
#endif
typedef cl_int (CL_API_CALL *clGetPlatformIDs_t)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (CL_API_CALL *clGetDeviceIDs_t)(cl_platform_id, cl_uint, cl_uint, cl_device_id*, cl_uint*);
typedef cl_context (CL_API_CALL *clCreateContext_t)(void*, cl_uint, cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue (CL_API_CALL *clCreateCommandQueue_t)(cl_context, cl_device_id, cl_uint, cl_int*);
typedef cl_program (CL_API_CALL *clCreateProgramWithSource_t)(cl_context, cl_uint, const char**, const cl_size_t*, cl_int*);
typedef cl_int (CL_API_CALL *clBuildProgram_t)(cl_program, cl_uint, cl_device_id*, const char*, void*, void*);
typedef cl_kernel (CL_API_CALL *clCreateKernel_t)(cl_program, const char*, cl_int*);
typedef cl_mem (CL_API_CALL *clCreateBuffer_t)(cl_context, cl_uint, cl_size_t, void*, cl_int*);
typedef cl_int (CL_API_CALL *clSetKernelArg_t)(cl_kernel, cl_uint, cl_size_t, const void*);
typedef cl_int (CL_API_CALL *clEnqueueWriteBuffer_t)(cl_command_queue, cl_mem, cl_int, cl_size_t, cl_size_t, const void*, cl_uint, void**);
typedef cl_int (CL_API_CALL *clEnqueueReadBuffer_t)(cl_command_queue, cl_mem, cl_int, cl_size_t, cl_size_t, void*, cl_uint, void**);
typedef cl_int (CL_API_CALL *clEnqueueNDRangeKernel_t)(cl_command_queue, cl_kernel, cl_uint, const cl_size_t*, const cl_size_t*, const cl_size_t*, cl_uint, void**);
typedef cl_int (CL_API_CALL *clFinish_t)(cl_command_queue);
static clGetPlatformIDs_t f_clGetPlatformIDs; static clGetDeviceIDs_t f_clGetDeviceIDs;
static clCreateContext_t f_clCreateContext; static clCreateCommandQueue_t f_clCreateCommandQueue;
static clCreateProgramWithSource_t f_clCreateProgramWithSource; static clBuildProgram_t f_clBuildProgram;
static clCreateKernel_t f_clCreateKernel; static clCreateBuffer_t f_clCreateBuffer;
static clSetKernelArg_t f_clSetKernelArg; static clEnqueueWriteBuffer_t f_clEnqueueWriteBuffer;
static clEnqueueReadBuffer_t f_clEnqueueReadBuffer; static clEnqueueNDRangeKernel_t f_clEnqueueNDRangeKernel;
static clFinish_t f_clFinish;
#define LOAD(fn,name) *(void**)&fn=(void*)GetProcAddress(cl,name)

// ── model loading ─────────────────────────────────────────────────────────
struct TensorInfo { std::string name; std::vector<int64_t> logical; std::string dtype; int64_t off0, off1; };
static std::vector<int64_t> parse_num_array(const std::string& s){
    std::vector<int64_t> r; std::string cur;
    for(char c:s){ if(c==','||c==']'){ if(!cur.empty()) r.push_back(std::stoll(cur)); cur.clear(); } else if(c!='['&&c!=' '&&c!='\n'&&c!='\r'&&c!='\t') cur+=c; }
    return r;
}
static std::vector<TensorInfo> load_tensor_map(const std::string& path){
    std::ifstream f(path); std::stringstream ss; ss<<f.rdbuf(); std::string s=ss.str();
    std::vector<TensorInfo> out; size_t pos=0;
    while((pos=s.find("\"name\"",pos))!=std::string::npos){
        size_t os=s.rfind('{',pos), oe=s.find('}',pos);
        if(os==std::string::npos||oe==std::string::npos) break;
        std::string o=s.substr(os,oe-os+1); TensorInfo t;
        size_t nq=o.find("\"name\""), vq=o.find(':',nq), v1=o.find('"',vq), v2=o.find('"',v1+1);
        t.name=o.substr(v1+1,v2-v1-1);
        size_t ls=o.find("\"logical_shape\""); if(ls!=std::string::npos){ size_t lb=o.find('[',ls),le=o.find(']',lb); t.logical=parse_num_array(o.substr(lb,le-lb+1)); }
        size_t gt=o.find("\"ggml_type_name\""); if(gt!=std::string::npos){ size_t gq=o.find(':',gt),g1=o.find('"',gq),g2=o.find('"',g1+1); t.dtype=o.substr(g1+1,g2-g1-1); }
        size_t dof=o.find("\"data_offsets\""); if(dof!=std::string::npos){ size_t db=o.find('[',dof),de=o.find(']',db); auto r=parse_num_array(o.substr(db,de-db+1)); if(r.size()>=2){ t.off0=r[0]; t.off1=r[1]; } }
        out.push_back(t); pos=oe+1;
    }
    return out;
}
static int64_t payload_origin(const std::string& path){
    std::ifstream f(path,std::ios::binary); uint64_t n=0; f.read((char*)&n,8); return 8+(int64_t)n;
}
static const int8_t MXFP4_K[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
static float e8m0_to_fp32(uint8_t x){ uint32_t bits=(x<2)?(0x00200000u<<x):((uint32_t)(x-1)<<23); float f; memcpy(&f,&bits,4); return f; }
static float silu(float x){ return x/(1.0f+std::exp(-x)); }
static void matvecT(const std::vector<float>& A,size_t K,size_t M,const std::vector<float>& x,std::vector<float>& y){
    y.assign(M,0); for(size_t i=0;i<M;i++){ float s=0; for(size_t k=0;k<K;k++) s+=A[k*M+i]*x[k]; y[i]=s; }
}

int main(){
    const size_t H=2880;
    // ── OpenCL setup FIRST (before model loading) ────────────────────────
    HMODULE cl=LoadLibraryA("OpenCL.dll"); if(!cl){printf("FAIL: no OpenCL.dll\n");return 1;}
    LOAD(f_clGetPlatformIDs,"clGetPlatformIDs"); LOAD(f_clGetDeviceIDs,"clGetDeviceIDs");
    LOAD(f_clCreateContext,"clCreateContext"); LOAD(f_clCreateCommandQueue,"clCreateCommandQueue");
    LOAD(f_clCreateProgramWithSource,"clCreateProgramWithSource"); LOAD(f_clBuildProgram,"clBuildProgram");
    LOAD(f_clCreateKernel,"clCreateKernel"); LOAD(f_clCreateBuffer,"clCreateBuffer");
    LOAD(f_clSetKernelArg,"clSetKernelArg"); LOAD(f_clEnqueueWriteBuffer,"clEnqueueWriteBuffer");
    LOAD(f_clEnqueueReadBuffer,"clEnqueueReadBuffer"); LOAD(f_clEnqueueNDRangeKernel,"clEnqueueNDRangeKernel");
    LOAD(f_clFinish,"clFinish");
    cl_uint np=0; f_clGetPlatformIDs(0,nullptr,&np);
    std::vector<cl_platform_id> plats(np); f_clGetPlatformIDs(np,plats.data(),nullptr);
    cl_device_id dev=nullptr;
    for(auto p:plats){ cl_uint nd=0; f_clGetDeviceIDs(p,CL_DEVICE_TYPE_ALL,0,nullptr,&nd);
        if(nd){ std::vector<cl_device_id> devs(nd); f_clGetDeviceIDs(p,CL_DEVICE_TYPE_ALL,nd,devs.data(),nullptr); dev=devs[0]; break; } }
    cl_int err=0; cl_context ctx=f_clCreateContext(nullptr,1,&dev,nullptr,nullptr,&err);
    cl_command_queue cq=f_clCreateCommandQueue(ctx,dev,0,&err);
    const char* src=
      "constant int kvalues[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};\n"
      "float e8m0(uint x){ uint bits=(x<2)?(0x00200000u<<x):((x-1)<<23); return as_float(bits); }\n"
      "__kernel void dequant_plane(__global const uchar* raw, __global float* plane, uint e, uint nblocks){\n"
      "  uint k=get_global_id(0); if(k>=nblocks) return;\n"
      "  float d=e8m0(raw[k*17]); uchar byte=raw[k*17+1+(e%16)];\n"
      "  int nib=(e<16)?(int)(byte&0x0F):(int)((byte>>4)&0x0F); plane[k]=d*kvalues[nib]; }\n"
      "__kernel void router(__global const float* W, __global const float* x, __global float* logits, uint K){\n"
      "  int i=get_global_id(0); if(i>=32) return; float s=0; for(int j=0;j<(int)K;j++) s+=W[j*32+i]*x[j]; logits[i]=s; }\n"
      "__kernel void expert_matvec(__global const float* plane, __global const float* x, __global float* y, uint K){\n"
      "  int i=get_global_id(0); if(i>=(int)K) return; float s=0;\n"
      "  for(int j=0;j<(int)K;j++) s+=plane[j*K+i]*x[j]; y[i]=s; }\n"
      "__kernel void merge(__global float* moe, __global const float* down, float w, uint K){\n"
      "  int i=get_global_id(0); if(i>=(int)K) return; moe[i]+=w*down[i]; }\n";
    cl_program prog=f_clCreateProgramWithSource(ctx,1,&src,nullptr,&err);
    err=f_clBuildProgram(prog,1,&dev,"-cl-std=CL1.2",nullptr,nullptr);
    if(err){printf("FAIL: build rc=%d\n",err);return 1;}
    cl_kernel kd=f_clCreateKernel(prog,"dequant_plane",&err);
    cl_kernel kr=f_clCreateKernel(prog,"router",&err);
    cl_kernel km=f_clCreateKernel(prog,"expert_matvec",&err);
    cl_kernel kg=f_clCreateKernel(prog,"merge",&err);
    printf("OPENCL READY ctx=%p cq=%p write=%p\n",(void*)ctx,(void*)cq,(void*)f_clEnqueueWriteBuffer); fflush(stdout);

    // ── model loading ────────────────────────────────────────────────────
    const std::string ST="E:/models/GPT-DDS/GPT-OSS/gpt-oss-20b-MXFP4.safetensors";
    const std::string TM="E:/models/GPT-DDS/GPT-OSS/gpt-oss-20b-MXFP4.tensor_map.json";
    auto tensors=load_tensor_map(TM);
    std::map<std::string,TensorInfo> ts; for(auto&t:tensors) ts[t.name]=t;
    int64_t origin=payload_origin(ST);
    std::ifstream sf(ST,std::ios::binary);
    auto read_raw=[&](const std::string& name)->std::vector<uint8_t>{
        auto&t=ts[name]; std::vector<uint8_t> raw(t.off1-t.off0);
        sf.seekg(origin+t.off0); sf.read((char*)raw.data(),raw.size()); return raw;
    };
    auto read_f32=[&](const std::string& name)->std::vector<float>{
        auto&t=ts[name]; size_t n=1; for(auto d:t.logical) n*=d;
        std::vector<uint8_t> raw(t.off1-t.off0);
        sf.seekg(origin+t.off0); sf.read((char*)raw.data(),raw.size());
        std::vector<float> out(n); memcpy(out.data(),raw.data(),n*4); return out;
    };
    std::vector<float> h_moe(H); for(size_t i=0;i<H;i++) h_moe[i]=(float)(rand()%1000)/1000.f-0.5f;
    auto gate_inp=read_f32("blk.0.ffn_gate_inp.weight");
    auto raw_gate=read_raw("blk.0.ffn_gate_exps.weight");
    auto raw_up=read_raw("blk.0.ffn_up_exps.weight");
    auto raw_down=read_raw("blk.0.ffn_down_exps.weight");
    printf("MODEL LOADED\n"); fflush(stdout);

    // ── CPU reference ────────────────────────────────────────────────────
    std::vector<float> logits(32); matvecT(gate_inp,2880,32,h_moe,logits);
    std::vector<int> idx(32); for(int i=0;i<32;i++) idx[i]=i;
    std::partial_sort(idx.begin(),idx.begin()+4,idx.end(),[&](int a,int b){return logits[a]>logits[b];});
    int top4[4]={idx[0],idx[1],idx[2],idx[3]};
    float mx=logits[top4[0]],sum=0,wts[4];
    for(int j=0;j<4;j++) wts[j]=std::exp(logits[top4[j]]-mx),sum+=wts[j];
    for(int j=0;j<4;j++) wts[j]/=sum;
    auto dequant_plane=[&](const std::vector<uint8_t>& raw,int e)->std::vector<float>{
        std::vector<float> p(2880*2880);
        for(size_t k=0;k<2880*2880;k++){ const uint8_t* b=&raw[k*17]; float d=e8m0_to_fp32(b[0]);
            uint8_t byte=b[1+(e%16)]; int8_t nib=(e<16)?(int8_t)(byte&0x0F):(int8_t)((byte>>4)&0x0F); p[k]=d*MXFP4_K[nib]; }
        return p;
    };
    std::vector<float> moe_cpu(2880,0);
    for(int j=0;j<4;j++){
        int e=top4[j];
        auto pg=dequant_plane(raw_gate,e), pu=dequant_plane(raw_up,e), pd=dequant_plane(raw_down,e);
        std::vector<float> ge(2880),up(2880),inter(2880),down(2880);
        for(int i=0;i<2880;i++){ float sg=0,su=0; for(int jj=0;jj<2880;jj++){ sg+=pg[jj*2880+i]*h_moe[jj]; su+=pu[jj*2880+i]*h_moe[jj]; } ge[i]=sg; up[i]=su; }
        for(int i=0;i<2880;i++) inter[i]=silu(ge[i])*up[i];
        for(int i=0;i<2880;i++){ float sd=0; for(int jj=0;jj<2880;jj++) sd+=pd[jj*2880+i]*inter[jj]; down[i]=sd; }
        for(int i=0;i<2880;i++) moe_cpu[i]+=wts[j]*down[i];
    }
    printf("CPU top4=[%d,%d,%d,%d]\n",top4[0],top4[1],top4[2],top4[3]); fflush(stdout);

    // ── OpenCL MoE ───────────────────────────────────────────────────────
    size_t nblocks=2880*2880;
    cl_mem bGate=f_clCreateBuffer(ctx,CL_MEM_READ_ONLY,raw_gate.size(),nullptr,&err);
    cl_mem bUp=f_clCreateBuffer(ctx,CL_MEM_READ_ONLY,raw_up.size(),nullptr,&err);
    cl_mem bDown=f_clCreateBuffer(ctx,CL_MEM_READ_ONLY,raw_down.size(),nullptr,&err);
    cl_mem bH=f_clCreateBuffer(ctx,CL_MEM_READ_ONLY,2880*4,nullptr,&err);
    cl_mem bGI=f_clCreateBuffer(ctx,CL_MEM_READ_ONLY,2880*32*4,nullptr,&err);
    cl_mem bLogits=f_clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,32*4,nullptr,&err);
    cl_mem bPlane=f_clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,nblocks*4,nullptr,&err);
    cl_mem bGe=f_clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,2880*4,nullptr,&err);
    cl_mem bUpv=f_clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,2880*4,nullptr,&err);
    cl_mem bInter=f_clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,2880*4,nullptr,&err);
    cl_mem bDownv=f_clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,2880*4,nullptr,&err);
    cl_mem bMoe=f_clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,2880*4,nullptr,&err);
    printf("BUFFERS err=%d\n",err); fflush(stdout);
    auto chunk_write=[&](cl_mem buf,const std::vector<uint8_t>& data){
        const size_t CH=8*1024*1024;
        for(size_t off=0; off<data.size(); off+=CH){
            size_t n=(CH<data.size()-off)?CH:(data.size()-off);
            f_clEnqueueWriteBuffer(cq,buf,1,off,n,data.data()+off,0,nullptr);
        }
    };
    chunk_write(bGate,raw_gate); chunk_write(bUp,raw_up); chunk_write(bDown,raw_down);
    f_clEnqueueWriteBuffer(cq,bH,1,0,2880*4,h_moe.data(),0,nullptr);
    f_clEnqueueWriteBuffer(cq,bGI,1,0,2880*32*4,gate_inp.data(),0,nullptr);
    std::vector<float> moe_gpu(2880,0);
    f_clEnqueueWriteBuffer(cq,bMoe,1,0,2880*4,moe_gpu.data(),0,nullptr);
    printf("WRITES OK\n"); fflush(stdout);

    f_clSetKernelArg(kr,0,sizeof(cl_mem),&bGI); f_clSetKernelArg(kr,1,sizeof(cl_mem),&bH);
    f_clSetKernelArg(kr,2,sizeof(cl_mem),&bLogits); cl_uint K=2880; f_clSetKernelArg(kr,3,sizeof(cl_uint),&K);
    cl_size_t g32=32,l32=32; f_clEnqueueNDRangeKernel(cq,kr,1,nullptr,&g32,&l32,0,nullptr);
    std::vector<float> logits_gpu(32); f_clEnqueueReadBuffer(cq,bLogits,1,0,32*4,logits_gpu.data(),0,nullptr);
    f_clFinish(cq);
    std::vector<int> gidx(32); for(int i=0;i<32;i++) gidx[i]=i;
    std::partial_sort(gidx.begin(),gidx.begin()+4,gidx.end(),[&](int a,int b){return logits_gpu[a]>logits_gpu[b];});
    int gtop4[4]={gidx[0],gidx[1],gidx[2],gidx[3]};
    float gmx=logits_gpu[gtop4[0]],gsum=0,gwts[4];
    for(int j=0;j<4;j++) gwts[j]=std::exp(logits_gpu[gtop4[j]]-gmx),gsum+=gwts[j];
    for(int j=0;j<4;j++) gwts[j]/=gsum;
    printf("GPU  top4=[%d,%d,%d,%d]\n",gtop4[0],gtop4[1],gtop4[2],gtop4[3]); fflush(stdout);

    cl_size_t gb=nblocks, lb=64;
    cl_size_t g2880=2880, l2880=64;
    cl_uint nb=nblocks;
    for(int j=0;j<4;j++){
        cl_uint e=gtop4[j];
        f_clSetKernelArg(kd,0,sizeof(cl_mem),&bGate); f_clSetKernelArg(kd,1,sizeof(cl_mem),&bPlane); f_clSetKernelArg(kd,2,sizeof(cl_uint),&e); f_clSetKernelArg(kd,3,sizeof(cl_uint),&nb);
        f_clEnqueueNDRangeKernel(cq,kd,1,nullptr,&gb,&lb,0,nullptr);
        f_clSetKernelArg(km,0,sizeof(cl_mem),&bPlane); f_clSetKernelArg(km,1,sizeof(cl_mem),&bH); f_clSetKernelArg(km,2,sizeof(cl_mem),&bGe); f_clSetKernelArg(km,3,sizeof(cl_uint),&K);
        f_clEnqueueNDRangeKernel(cq,km,1,nullptr,&g2880,&l2880,0,nullptr);
        f_clSetKernelArg(kd,0,sizeof(cl_mem),&bUp); f_clEnqueueNDRangeKernel(cq,kd,1,nullptr,&gb,&lb,0,nullptr);
        f_clSetKernelArg(km,0,sizeof(cl_mem),&bPlane); f_clSetKernelArg(km,2,sizeof(cl_mem),&bUpv); f_clEnqueueNDRangeKernel(cq,km,1,nullptr,&g2880,&l2880,0,nullptr);
        std::vector<float> ge(2880),upv(2880),inter(2880);
        f_clEnqueueReadBuffer(cq,bGe,1,0,2880*4,ge.data(),0,nullptr);
        f_clEnqueueReadBuffer(cq,bUpv,1,0,2880*4,upv.data(),0,nullptr);
        f_clFinish(cq);
        for(int i=0;i<2880;i++) inter[i]=silu(ge[i])*upv[i];
        f_clEnqueueWriteBuffer(cq,bInter,1,0,2880*4,inter.data(),0,nullptr);
        f_clSetKernelArg(kd,0,sizeof(cl_mem),&bDown); f_clEnqueueNDRangeKernel(cq,kd,1,nullptr,&gb,&lb,0,nullptr);
        f_clSetKernelArg(km,0,sizeof(cl_mem),&bPlane); f_clSetKernelArg(km,1,sizeof(cl_mem),&bInter); f_clSetKernelArg(km,2,sizeof(cl_mem),&bDownv); f_clEnqueueNDRangeKernel(cq,km,1,nullptr,&g2880,&l2880,0,nullptr);
        f_clSetKernelArg(kg,0,sizeof(cl_mem),&bMoe); f_clSetKernelArg(kg,1,sizeof(cl_mem),&bDownv); f_clSetKernelArg(kg,2,sizeof(float),&gwts[j]); f_clSetKernelArg(kg,3,sizeof(cl_uint),&K);
        f_clEnqueueNDRangeKernel(cq,kg,1,nullptr,&g2880,&l2880,0,nullptr);
    }
    f_clEnqueueReadBuffer(cq,bMoe,1,0,2880*4,moe_gpu.data(),0,nullptr);
    f_clFinish(cq);

    double maxerr=0; for(int i=0;i<2880;i++){ double e=fabs(moe_gpu[i]-moe_cpu[i]); if(e>maxerr)maxerr=e; }
    printf("OpenCL MoE layer vs CPU: max-err=%.3e\n", maxerr);
    printf("RESULT: %s\n", maxerr<1e-3 ? "PASS — full OpenCL MoE layer (dequant+router+matvec+merge) matches CPU on HD 4600" : "FAIL");
    return maxerr<1e-3?0:1;
}
