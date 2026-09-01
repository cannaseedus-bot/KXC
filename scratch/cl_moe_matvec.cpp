// cl_moe_matvec.cpp — OpenCL MoE expert matvec on the HD 4600.
// Proves the GPU does the core MoE op (transpose matvec y[i]=sum_j W[j*K+i]*x[j])
// correctly, matching the CPU. This is the GPU-resident inference kernel.
// Build (MSVC): cl /O2 cl_moe_matvec.cpp /Fe:cl_moe_matvec.exe
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

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

int main(){
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

    const char* src="__kernel void expert_matvec(__global const float* W, __global const float* x, __global float* y, uint K){"
        "int i=get_global_id(0); if(i>=(int)K) return;"
        "float s=0; for(int j=0;j<(int)K;j++) s+=W[j*K+i]*x[j]; y[i]=s; }";
    cl_program prog=f_clCreateProgramWithSource(ctx,1,&src,nullptr,&err);
    err=f_clBuildProgram(prog,1,&dev,"-cl-std=CL1.2",nullptr,nullptr);
    if(err){printf("FAIL: build rc=%d\n",err);return 1;}
    cl_kernel k=f_clCreateKernel(prog,"expert_matvec",&err);

    const int K=2880;   // one expert plane [2880,2880] @ [2880]
    std::vector<float> W(K*K), x(K), yg(K), yc(K);
    for(size_t i=0;i<W.size();i++) W[i]=(float)(rand()%1000)/1000.f-0.5f;
    for(size_t i=0;i<K;i++) x[i]=(float)(rand()%1000)/1000.f-0.5f;
    // CPU reference (transpose matvec)
    for(int i=0;i<K;i++){ float s=0; for(int j=0;j<K;j++) s+=W[j*K+i]*x[j]; yc[i]=s; }

    cl_mem bufW=f_clCreateBuffer(ctx,CL_MEM_READ_ONLY,K*K*4,nullptr,&err);
    cl_mem bufX=f_clCreateBuffer(ctx,CL_MEM_READ_ONLY,K*4,nullptr,&err);
    cl_mem bufY=f_clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,K*4,nullptr,&err);
    f_clEnqueueWriteBuffer(cq,bufW,1,0,K*K*4,W.data(),0,nullptr);
    f_clEnqueueWriteBuffer(cq,bufX,1,0,K*4,x.data(),0,nullptr);
    f_clSetKernelArg(k,0,sizeof(cl_mem),&bufW);
    f_clSetKernelArg(k,1,sizeof(cl_mem),&bufX);
    f_clSetKernelArg(k,2,sizeof(cl_mem),&bufY);
    cl_uint kk=K; f_clSetKernelArg(k,3,sizeof(cl_uint),&kk);
    cl_size_t global=K, local=64;
    f_clEnqueueNDRangeKernel(cq,k,1,nullptr,&global,&local,0,nullptr);
    f_clEnqueueReadBuffer(cq,bufY,1,0,K*4,yg.data(),0,nullptr);
    f_clFinish(cq);

    double errsum=0, maxerr=0; for(int i=0;i<K;i++){ double e=fabs(yg[i]-yc[i]); errsum+=e; if(e>maxerr)maxerr=e; }
    printf("OpenCL expert matvec [%dx%d]@[%d]: max-err=%.3e avg-err=%.3e\n", K,K,K, maxerr, errsum/K);
    printf("RESULT: %s\n", maxerr<1e-3 ? "PASS — OpenCL MoE matvec matches CPU on HD 4600" : "FAIL");
    return maxerr<1e-3?0:1;
}
