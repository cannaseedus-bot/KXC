#include "opencl_error.h"
#include <cstdio>

namespace kuhul {

const char* opencl_error_name(int code) {
    switch (code) {
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -30: return "CL_INVALID_VALUE";
        case -34: return "CL_INVALID_CONTEXT";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -43: return "CL_INVALID_PROGRAM";
        case -44: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -48: return "CL_INVALID_KERNEL";
        case -52: return "CL_INVALID_ARG_INDEX";
        case -54: return "CL_INVALID_ARG_SIZE";
        case -57: return "CL_INVALID_WORK_GROUP_SIZE";
        default: return "CL_UNKNOWN_ERROR";
    }
}

bool check_opencl(int code, const char* operation, const char* file, int line, std::string* message) {
    if (code == 0) return true;
    char buffer[512] = {};
    std::snprintf(buffer, sizeof(buffer), "%s failed: %s (%d) at %s:%d", operation,
                  opencl_error_name(code), code, file, line);
    if (message) *message = buffer;
    std::fprintf(stderr, "%s\n", buffer);
    return false;
}

}
