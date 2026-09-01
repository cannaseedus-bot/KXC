#ifndef KUHUL_OPENCL_ERROR_H
#define KUHUL_OPENCL_ERROR_H

#include <string>

namespace kuhul {

const char* opencl_error_name(int code);
bool check_opencl(int code, const char* operation, const char* file, int line, std::string* message = nullptr);

}

#define KUHUL_CL_CHECK(code, operation) \
    ::kuhul::check_opencl((code), (operation), __FILE__, __LINE__)

#endif
