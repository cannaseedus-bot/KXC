// Compile every repository OpenCL C 1.2 source and create each registered kernel.
// Dynamically loads the ICD, so no OpenCL SDK headers or import library are needed.
#include <windows.h>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using cl_int = int;
using cl_uint = unsigned int;
using cl_platform_id = void*;
using cl_device_id = void*;
using cl_context = void*;
using cl_program = void*;
using cl_kernel = void*;
using cl_command_queue = void*;
using cl_size_t = size_t;

constexpr cl_uint CL_DEVICE_TYPE_ALL = 0xFFFFFFFFu;
constexpr cl_int CL_SUCCESS = 0;
constexpr cl_uint CL_PROGRAM_BUILD_LOG = 0x1183;

#ifndef CL_API_CALL
#define CL_API_CALL __stdcall
#endif

using GetPlatforms = cl_int (CL_API_CALL*)(cl_uint, cl_platform_id*, cl_uint*);
using GetDevices = cl_int (CL_API_CALL*)(cl_platform_id, cl_uint, cl_uint, cl_device_id*, cl_uint*);
using CreateContext = cl_context (CL_API_CALL*)(void*, cl_uint, cl_device_id*, void*, void*, cl_int*);
using CreateProgram = cl_program (CL_API_CALL*)(cl_context, cl_uint, const char**, const cl_size_t*, cl_int*);
using BuildProgram = cl_int (CL_API_CALL*)(cl_program, cl_uint, cl_device_id*, const char*, void*, void*);
using CreateKernel = cl_kernel (CL_API_CALL*)(cl_program, const char*, cl_int*);
using GetBuildInfo = cl_int (CL_API_CALL*)(cl_program, cl_device_id, cl_uint, cl_size_t, void*, cl_size_t*);
using ReleaseProgram = cl_int (CL_API_CALL*)(cl_program);
using ReleaseKernel = cl_int (CL_API_CALL*)(cl_kernel);
using ReleaseContext = cl_int (CL_API_CALL*)(cl_context);

#define LOAD(name) reinterpret_cast<name##Fn>(GetProcAddress(cl, #name))

static std::string readText(const char* path) {
    std::ifstream file(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
}

struct SourceSpec {
    const char* path;
    std::vector<const char*> kernels;
};

int main() {
    HMODULE cl = LoadLibraryA("OpenCL.dll");
    if (!cl) { std::puts("FAIL: no OpenCL.dll"); return 1; }

    auto getPlatforms = reinterpret_cast<GetPlatforms>(GetProcAddress(cl, "clGetPlatformIDs"));
    auto getDevices = reinterpret_cast<GetDevices>(GetProcAddress(cl, "clGetDeviceIDs"));
    auto createContext = reinterpret_cast<CreateContext>(GetProcAddress(cl, "clCreateContext"));
    auto createProgram = reinterpret_cast<CreateProgram>(GetProcAddress(cl, "clCreateProgramWithSource"));
    auto buildProgram = reinterpret_cast<BuildProgram>(GetProcAddress(cl, "clBuildProgram"));
    auto createKernel = reinterpret_cast<CreateKernel>(GetProcAddress(cl, "clCreateKernel"));
    auto getBuildInfo = reinterpret_cast<GetBuildInfo>(GetProcAddress(cl, "clGetProgramBuildInfo"));
    auto releaseProgram = reinterpret_cast<ReleaseProgram>(GetProcAddress(cl, "clReleaseProgram"));
    auto releaseKernel = reinterpret_cast<ReleaseKernel>(GetProcAddress(cl, "clReleaseKernel"));
    auto releaseContext = reinterpret_cast<ReleaseContext>(GetProcAddress(cl, "clReleaseContext"));
    if (!getPlatforms || !getDevices || !createContext || !createProgram ||
        !buildProgram || !createKernel || !getBuildInfo) {
        std::puts("FAIL: incomplete OpenCL 1.2 ICD"); return 1;
    }

    cl_uint platformCount = 0;
    if (getPlatforms(0, nullptr, &platformCount) != CL_SUCCESS || platformCount == 0) {
        std::puts("FAIL: no OpenCL platform"); return 1;
    }
    std::vector<cl_platform_id> platforms(platformCount);
    getPlatforms(platformCount, platforms.data(), nullptr);
    cl_device_id device = nullptr;
    for (auto platform : platforms) {
        cl_uint count = 0;
        if (getDevices(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &count) == CL_SUCCESS && count) {
            std::vector<cl_device_id> devices(count);
            getDevices(platform, CL_DEVICE_TYPE_ALL, count, devices.data(), nullptr);
            device = devices[0];
            break;
        }
    }
    if (!device) { std::puts("FAIL: no OpenCL device"); return 1; }

    cl_int error = 0;
    cl_context context = createContext(nullptr, 1, &device, nullptr, nullptr, &error);
    if (!context || error != CL_SUCCESS) { std::printf("FAIL: context rc=%d\n", error); return 1; }

    const SourceSpec sources[] = {
        {"drivers/opencl_1_2/sobel.cl", {"sobel_x", "sobel_y", "sobel_magnitude", "sobel_threshold", "elementwise_mul", "reduce_pair_sum"}},
        {"drivers/opencl_1_2/arc_replay.cl", {"arc_replay"}},
        {"drivers/opencl_1_2/geodesic_distance.cl", {"geodesic_distance"}},
        {"drivers/opencl_1_2/kson_fold_node_expert.cl", {"kson_fold_node_gate"}},
        {"drivers/opencl_1_2/tensor_evolve.cl", {"tensor_evolve"}},
        {"drivers/opencl_1_2/matrix5x4.cl", {"matrix5x4_mul", "matrix5x4_matmul", "matrix5x4_add", "matrix5x4_swiglu"}},
        {"drivers/opencl_1_2/training.cl", {"swiglu_forward", "rms_norm_forward", "softmax_forward", "cross_entropy_loss_back", "rms_norm_back", "adamw_step"}}
    };

    int failures = 0;
    for (const auto& source : sources) {
        const std::string text = readText(source.path);
        if (text.empty()) { std::printf("FAIL: cannot read %s\n", source.path); ++failures; continue; }
        const char* ptr = text.c_str();
        cl_program program = createProgram(context, 1, &ptr, nullptr, &error);
        error = buildProgram(program, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
        if (error != CL_SUCCESS) {
            size_t bytes = 0;
            getBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &bytes);
            std::string log(bytes, '\0');
            if (bytes) getBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, bytes, &log[0], nullptr);
            std::printf("FAIL: %s build rc=%d\n%s\n", source.path, error, log.c_str());
            ++failures;
            if (releaseProgram) releaseProgram(program);
            continue;
        }
        int created = 0;
        for (const char* name : source.kernels) {
            cl_kernel kernel = createKernel(program, name, &error);
            if (!kernel || error != CL_SUCCESS) {
                std::printf("FAIL: %s :: %s create rc=%d\n", source.path, name, error);
                ++failures;
            } else {
                ++created;
                if (releaseKernel) releaseKernel(kernel);
            }
        }
        std::printf("%s: compiled %d kernels\n", source.path, created);
        if (releaseProgram) releaseProgram(program);
    }
    if (releaseContext) releaseContext(context);
    std::printf("RESULT: %s\n", failures == 0 ? "PASS — all OpenCL 1.2 sources compiled" : "FAIL");
    return failures == 0 ? 0 : 1;
}
