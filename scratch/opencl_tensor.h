#pragma once

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Project tensor dtype; intentionally separate from OpenCL image/channel enums.
enum class OpenCLDType : std::uint8_t { F32, F16, I32, I64, U8, BF16 };
enum class OpenCLLayout : std::uint8_t { RowMajor, ColumnMajor, CustomStrides };
enum class OpenCLSemanticRole : std::uint8_t {
    Input, Weight, Gradient, Output, Scratch, Embedding, Attention, Cache, Replay
};

struct OpenCLTensor {
    cl_mem buffer = nullptr;
    OpenCLDType dtype = OpenCLDType::F32;
    OpenCLLayout layout = OpenCLLayout::RowMajor;
    std::vector<std::size_t> shape;
    std::vector<std::size_t> strides;
    std::size_t num_elements = 0;
    OpenCLSemanticRole role = OpenCLSemanticRole::Input;
    std::string name;
    std::string fold_phase = "Pop";
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    bool persistent = false;

    std::size_t element_size() const noexcept;
    std::size_t total_bytes() const noexcept;
    std::size_t index(const std::vector<std::size_t>& coordinates) const;
    void set_shape(const std::vector<std::size_t>& new_shape);
};

inline std::size_t opencl_dtype_size(OpenCLDType dtype) noexcept {
    switch (dtype) {
    case OpenCLDType::F32: return 4;
    case OpenCLDType::F16: return 2;
    case OpenCLDType::I32: return 4;
    case OpenCLDType::I64: return 8;
    case OpenCLDType::U8: return 1;
    case OpenCLDType::BF16: return 2;
    }
    return 0;
}
