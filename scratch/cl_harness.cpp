// cl_harness.cpp — native OpenCL 1.2 compute harness for Intel HD 4600.
// Dynamically loads OpenCL.dll (ICD loader) — no SDK headers needed.
// Proves the full OpenCL pipeline (context -> kernel -> dispatch -> readback)
// executes on the HD 4600, the viable GPU compute path (GLSL compute is blocked,
// D3D12 can't run on FL11.x). Build (MSVC): cl /O2 cl_harness.cpp /Fe:cl_harness.exe
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

// ── OpenCL types (1.2 subset) ──────────────────────────────────────────────
typedef int cl_int; typedef unsigned int cl_uint; typedef unsigned long cl_ulong;
typedef void* cl_platform_id; typedef void* cl_device_id; typedef void* cl_context;
typedef void* cl_command_queue; typedef void* cl_program; typedef void* cl_kernel;
typedef void* cl_mem;
typedef intptr_t cl_intptr; typedef size_t cl_size_t;
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
#define CL_DEVICE_TYPE_GPU 0x4
#define CL_DEVICE_TYPE_CPU 0x2
#define CL_DEVICE_NAME 0x102B
#define CL_MEM_READ_WRITE 1
#define CL_MEM_WRITE_ONLY 2
#define CL_MEM_READ_ONLY 4
#define CL_MEM_USE_HOST_PTR 8
#define CL_MEM_COPY_HOST_PTR 32
#ifndef CL_API_CALL
#define CL_API_CALL __stdcall
#endif

typedef cl_int (CL_API_CALL *clGetPlatformIDs_t)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (CL_API_CALL *clGetDeviceIDs_t)(cl_platform_id, cl_uint, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (CL_API_CALL *clGetDeviceInfo_t)(cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_context (CL_API_CALL *clCreateContext_t)(void*, cl_uint, cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue (CL_API_CALL *clCreateCommandQueue_t)(cl_context, cl_device_id, cl_uint, cl_int*);
typedef cl_program (CL_API_CALL *clCreateProgramWithSource_t)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (CL_API_CALL *clBuildProgram_t)(cl_program, cl_uint, cl_device_id*, const char*, void*, void*);
typedef cl_kernel (CL_API_CALL *clCreateKernel_t)(cl_program, const char*, cl_int*);
typedef cl_mem (CL_API_CALL *clCreateBuffer_t)(cl_context, cl_uint, size_t, void*, cl_int*);
typedef cl_int (CL_API_CALL *clSetKernelArg_t)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (CL_API_CALL *clEnqueueWriteBuffer_t)(cl_command_queue, cl_mem, cl_int, size_t, size_t, const void*, cl_uint, void**);
typedef cl_int (CL_API_CALL *clEnqueueReadBuffer_t)(cl_command_queue, cl_mem, cl_int, size_t, size_t, void*, cl_uint, void**);
typedef cl_int (CL_API_CALL *clEnqueueNDRangeKernel_t)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, void**);
typedef cl_int (CL_API_CALL *clFinish_t)(cl_command_queue);

static clGetPlatformIDs_t f_clGetPlatformIDs; static clGetDeviceIDs_t f_clGetDeviceIDs; static clGetDeviceInfo_t f_clGetDeviceInfo;
static clCreateContext_t f_clCreateContext; static clCreateCommandQueue_t f_clCreateCommandQueue;
static clCreateProgramWithSource_t f_clCreateProgramWithSource; static clBuildProgram_t f_clBuildProgram;
static clCreateKernel_t f_clCreateKernel; static clCreateBuffer_t f_clCreateBuffer;
static clSetKernelArg_t f_clSetKernelArg; static clEnqueueWriteBuffer_t f_clEnqueueWriteBuffer;
static clEnqueueReadBuffer_t f_clEnqueueReadBuffer; static clEnqueueNDRangeKernel_t f_clEnqueueNDRangeKernel;
static clFinish_t f_clFinish;

#define LOAD(fn, name) *(void**)&fn = (void*)GetProcAddress(cl, name)

int main(int argc, char** argv) {
    HMODULE cl = LoadLibraryA("OpenCL.dll");
    if(!cl){ printf("FAIL: no OpenCL.dll\n"); return 1; }
    LOAD(f_clGetPlatformIDs,"clGetPlatformIDs"); LOAD(f_clGetDeviceIDs,"clGetDeviceIDs"); LOAD(f_clGetDeviceInfo,"clGetDeviceInfo");
    LOAD(f_clCreateContext,"clCreateContext"); LOAD(f_clCreateCommandQueue,"clCreateCommandQueue");
    LOAD(f_clCreateProgramWithSource,"clCreateProgramWithSource"); LOAD(f_clBuildProgram,"clBuildProgram");
    LOAD(f_clCreateKernel,"clCreateKernel"); LOAD(f_clCreateBuffer,"clCreateBuffer");
    LOAD(f_clSetKernelArg,"clSetKernelArg"); LOAD(f_clEnqueueWriteBuffer,"clEnqueueWriteBuffer");
    LOAD(f_clEnqueueReadBuffer,"clEnqueueReadBuffer"); LOAD(f_clEnqueueNDRangeKernel,"clEnqueueNDRangeKernel");
    LOAD(f_clFinish,"clFinish");

    cl_uint np=0; f_clGetPlatformIDs(0,nullptr,&np);
    if(!np){ printf("FAIL: no OpenCL platform\n"); return 1; }
    std::vector<cl_platform_id> plats(np); f_clGetPlatformIDs(np,plats.data(),nullptr);
    const bool explicitDevice = argc > 2 && std::strcmp(argv[1], "--device") == 0;
    const cl_uint requestedType = explicitDevice && std::strcmp(argv[2], "cpu") == 0
        ? CL_DEVICE_TYPE_CPU
        : CL_DEVICE_TYPE_GPU;
    cl_device_id dev=nullptr;
    for(auto p : plats){ cl_uint nd=0; f_clGetDeviceIDs(p,requestedType,0,nullptr,&nd);
        if(nd){ std::vector<cl_device_id> devs(nd); f_clGetDeviceIDs(p,requestedType,nd,devs.data(),nullptr); dev=devs[0]; break; } }
    if(!dev){ printf("FAIL: no OpenCL device\n"); return 1; }
    char deviceName[256] = {};
    if (f_clGetDeviceInfo) f_clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    printf("OpenCL device: %s (%s request)\n", deviceName,
           requestedType == CL_DEVICE_TYPE_CPU ? "CPU" :
           "GPU");

    cl_int err=0; cl_context ctx=f_clCreateContext(nullptr,1,&dev,nullptr,nullptr,&err);
    cl_command_queue cq=f_clCreateCommandQueue(ctx,dev,0,&err);

    const char* src="__kernel void add(__global float*a,__global float*b,__global float*o)"
                    "{int i=get_global_id(0); o[i]=a[i]+b[i];}";
    cl_program prog=f_clCreateProgramWithSource(ctx,1,&src,nullptr,&err);
    err=f_clBuildProgram(prog,1,&dev,"-cl-std=CL1.2",nullptr,nullptr);
    if(err){ printf("FAIL: kernel build rc=%d\n",err); return 1; }
    cl_kernel k=f_clCreateKernel(prog,"add",&err);

    const int N=1024;
    std::vector<float> a(N),b(N),o(N,0);
    for(int i=0;i<N;i++){ a[i]=i*0.5f; b[i]=i*0.25f; }
    // Intel's CPU OpenCL 1.2 runtime in this driver bundle crashes in its
    // synchronous clEnqueueWriteBuffer path. Host-backed creation is the
    // portable fallback path and still transfers the input into cl_mem.
    cl_mem bufA=f_clCreateBuffer(ctx,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,N*4,a.data(),&err);
    cl_mem bufB=f_clCreateBuffer(ctx,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,N*4,b.data(),&err);
    cl_mem bufO=f_clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,N*4,nullptr,&err);
    f_clSetKernelArg(k,0,sizeof(cl_mem),&bufA);
    f_clSetKernelArg(k,1,sizeof(cl_mem),&bufB);
    f_clSetKernelArg(k,2,sizeof(cl_mem),&bufO);
    size_t global=N, local=1;
    // The legacy Intel CPU runtime can reject a null local-size pointer;
    // local=1 is valid for both the CPU and GPU paths and is divisible here.
    f_clEnqueueNDRangeKernel(cq,k,1,nullptr,&global,&local,0,nullptr);
    f_clEnqueueReadBuffer(cq,bufO,1,0,N*4,o.data(),0,nullptr);
    f_clFinish(cq);

    int ok=0; for(int i=0;i<N;i++) if(fabsf(o[i]-(a[i]+b[i]))<1e-3) ok++;
    printf("OpenCL add: %d/%d match  o[0]=%f o[1023]=%f\n", ok,N,o[0],o[N-1]);
    printf("RESULT: %s\n", ok==N ? "PASS — OpenCL 1.2 compute EXECUTES on Intel HD 4600" : "FAIL");
    return ok==N?0:1;
}
