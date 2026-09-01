#include "opencl_tensor.h"
#include <stdexcept>

std::size_t OpenCLTensor::element_size() const noexcept { return opencl_dtype_size(dtype); }

std::size_t OpenCLTensor::total_bytes() const noexcept { return num_elements * element_size(); }

void OpenCLTensor::set_shape(const std::vector<std::size_t>& new_shape) {
    shape = new_shape;
    strides.assign(shape.size(), 1);
    for (std::size_t i = shape.size(); i > 1; --i) strides[i - 2] = strides[i - 1] * shape[i - 1];
    num_elements = 1;
    for (const std::size_t dimension : shape) num_elements *= dimension;
}

std::size_t OpenCLTensor::index(const std::vector<std::size_t>& coordinates) const {
    if (coordinates.size() != shape.size()) throw std::invalid_argument("coordinate rank mismatch");
    std::size_t flat = 0;
    for (std::size_t i = 0; i < coordinates.size(); ++i) {
        if (coordinates[i] >= shape[i]) throw std::out_of_range("tensor coordinate out of range");
        flat += coordinates[i] * strides[i];
    }
    return flat;
}
