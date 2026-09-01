// OpenCL 1.2 sidecar for the pi-KUHUL field corrective.
// Inputs are raw little-endian float32: A=5x4 (20 floats), B=4x5 (20 floats).
// Output is raw float32 C=5x4 (20 floats). The main trainer remains PyTorch.
#include <windows.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using cl_int=int; using cl_uint=unsigned int; using cl_ulong=unsigned long;
using cl_platform_id=void*; using cl_device_id=void*; using cl_context=void*;
using cl_command_queue=void*; using cl_program=void*; using cl_kernel=void*; using cl_mem=void*;
using cl_bool=cl_uint; using cl_size_t=size_t;
using GetPlatforms=cl_int (__stdcall*)(cl_uint,cl_platform_id*,cl_uint*);
using GetDevices=cl_int (__stdcall*)(cl_platform_id,cl_ulong,cl_uint,cl_device_id*,cl_uint*);
using GetInfo=cl_int (__stdcall*)(cl_device_id,cl_uint,size_t,void*,size_t*);
using CreateContext=cl_context (__stdcall*)(void*,cl_uint,cl_device_id*,void*,void*,cl_int*);
using CreateQueue=cl_command_queue (__stdcall*)(cl_context,cl_device_id,cl_ulong,cl_int*);
using CreateSource=cl_program (__stdcall*)(cl_context,cl_uint,const char**,const size_t*,cl_int*);
using Build=cl_int (__stdcall*)(cl_program,cl_uint,cl_device_id*,const char*,void*,void*);
using CreateKernel=cl_kernel (__stdcall*)(cl_program,const char*,cl_int*);
using CreateBuffer=cl_mem (__stdcall*)(cl_context,cl_ulong,size_t,void*,cl_int*);
using SetArg=cl_int (__stdcall*)(cl_kernel,cl_uint,size_t,const void*);
using EnqueueND=cl_int (__stdcall*)(cl_command_queue,cl_kernel,cl_uint,const size_t*,const size_t*,const size_t*,cl_uint,void**,void**);
using ReadBuffer=cl_int (__stdcall*)(cl_command_queue,cl_mem,cl_bool,size_t,size_t,void*,cl_uint,void**,void**);
using Finish=cl_int (__stdcall*)(cl_command_queue);

static bool read20(const char* path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary); if (!f) return false;
    out.resize(20); f.read(reinterpret_cast<char*>(out.data()), 20*sizeof(float));
    return f.gcount() == static_cast<std::streamsize>(20*sizeof(float));
}
static bool write20(const char* path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary); if (!f) return false;
    f.write(reinterpret_cast<const char*>(v.data()), 20*sizeof(float)); return !!f;
}
static void replay(const char* path, const char* device, const std::vector<float>& out) {
    if (!path) return;
    std::ofstream f(path, std::ios::app); if (!f) return;
    double sum=0.0; for (float x : out) sum += x;
    f << "{\"op\":\"arc.replay\",\"kernel\":\"matrix5x4_matmul\","
         "\"backend\":\"opencl_1_2\",\"device\":\"" << device
      << "\",\"shape\":[5,4,4,5],\"output_sum\":" << sum
      << ",\"status\":\"complete\"}\n";
}

int main(int argc, char** argv) {
    const char* aPath = argc > 1 ? argv[1] : nullptr;
    const char* bPath = argc > 2 ? argv[2] : nullptr;
    const char* outPath = argc > 3 ? argv[3] : "scratch/matrix5x4_out.bin";
    const char* replayPath = argc > 4 ? argv[4] : "scratch/arc_replay.jsonl";
    const bool cpu = argc > 5 && std::strcmp(argv[5], "cpu") == 0;
    std::vector<float> a(20), b(20), out(20, 0.0f);
    if (aPath && bPath) {
        if (!read20(aPath, a) || !read20(bPath, b)) { std::puts("FAIL: expected 20-float inputs"); return 2; }
    } else {
        for (int i=0;i<20;i++) { a[i] = 0.01f * (i+1); b[i] = 0.02f * (i+1); }
    }

    HMODULE lib=LoadLibraryA("OpenCL.dll"); if(!lib){std::puts("FAIL: no OpenCL.dll");return 3;}
    auto gp=(GetPlatforms)GetProcAddress(lib,"clGetPlatformIDs"); auto gd=(GetDevices)GetProcAddress(lib,"clGetDeviceIDs");
    auto gi=(GetInfo)GetProcAddress(lib,"clGetDeviceInfo"); auto cc=(CreateContext)GetProcAddress(lib,"clCreateContext");
    auto cq=(CreateQueue)GetProcAddress(lib,"clCreateCommandQueue"); auto cs=(CreateSource)GetProcAddress(lib,"clCreateProgramWithSource");
    auto bp=(Build)GetProcAddress(lib,"clBuildProgram"); auto ck=(CreateKernel)GetProcAddress(lib,"clCreateKernel");
    auto cb=(CreateBuffer)GetProcAddress(lib,"clCreateBuffer"); auto sa=(SetArg)GetProcAddress(lib,"clSetKernelArg");
    auto nd=(EnqueueND)GetProcAddress(lib,"clEnqueueNDRangeKernel"); auto rb=(ReadBuffer)GetProcAddress(lib,"clEnqueueReadBuffer");
    auto fn=(Finish)GetProcAddress(lib,"clFinish");
    cl_uint np=0; if(gp(0,nullptr,&np)!=0 || !np){std::puts("FAIL: no OpenCL platform");return 4;}
    std::vector<cl_platform_id> ps(np); gp(np,ps.data(),nullptr); cl_device_id dev=nullptr;
    const cl_ulong type=cpu ? 2u : 4u;
    for(auto p:ps){cl_uint n=0;if(gd(p,type,0,nullptr,&n)==0&&n){std::vector<cl_device_id> ds(n);gd(p,type,n,ds.data(),nullptr);dev=ds[0];break;}}
    if(!dev){std::puts("FAIL: requested OpenCL device unavailable");return 5;}
    char name[256]={}; gi(dev,0x102B,sizeof(name),name,nullptr); std::printf("OpenCL matrix5x4 device: %s\n",name);
    cl_int err=0; cl_context ctx=cc(nullptr,1,&dev,nullptr,nullptr,&err); if(!ctx||err){std::printf("FAIL: context rc=%d\n",err);return 6;}
    cl_command_queue q=cq(ctx,dev,0,&err); if(!q||err){std::printf("FAIL: queue rc=%d\n",err);return 7;}
    std::ifstream sf("drivers/opencl_1_2/matrix5x4.cl"); std::string src((std::istreambuf_iterator<char>(sf)),{});
    if(src.empty()){std::puts("FAIL: matrix5x4.cl not found from project root");return 8;}
    const char* text=src.c_str(); cl_program p=cs(ctx,1,&text,nullptr,&err); err=bp(p,1,&dev,"-cl-std=CL1.2",nullptr,nullptr);
    if(err){std::printf("FAIL: kernel build rc=%d\n",err);return 9;}
    cl_kernel k=ck(p,"matrix5x4_matmul",&err); cl_mem ba=cb(ctx,4|32,80,a.data(),&err); cl_mem bb=cb(ctx,4|32,80,b.data(),&err); cl_mem bo=cb(ctx,2,80,nullptr,&err);
    sa(k,0,sizeof(cl_mem),&ba); sa(k,1,sizeof(cl_mem),&bb); sa(k,2,sizeof(cl_mem),&bo);
    size_t global[2]={5,4}, local[2]={1,1}; err=nd(q,k,2,nullptr,global,local,0,nullptr,nullptr); if(err){std::printf("FAIL: dispatch rc=%d\n",err);return 10;}
    err=rb(q,bo,1,0,80,out.data(),0,nullptr,nullptr); fn(q); if(err){std::printf("FAIL: readback rc=%d\n",err);return 11;}
    if(!write20(outPath,out)){std::puts("FAIL: output write");return 12;} replay(replayPath,name,out);
    std::printf("RESULT: PASS — matrix5x4_matmul output=%s replay=%s\n",outPath,replayPath); return 0;
}
