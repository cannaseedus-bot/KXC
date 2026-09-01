#include "kuhul_glyph_api.hpp"
#include "glyph_backend_abi.h"

#include <algorithm>
#include <stdexcept>

namespace KuhulGlyph {

Host::Host() = default;

uint32_t Host::abi_version() {
    return kuhul_glyph_abi_version();
}

uint32_t Host::lane_count() {
    return kuhul_glyph_lane_count();
}

bool Host::get_lane(uint32_t index, KuhulGlyphLaneDesc* out_desc) {
    return kuhul_glyph_get_lane(index, out_desc) == KUHUL_GLYPH_OK;
}

int32_t Host::execute_f32(
    uint32_t opcode,
    const float* input_a,
    const float* input_b,
    float* output,
    uint64_t element_count
) {
    return kuhul_glyph_execute_f32(opcode, input_a, input_b, output, element_count);
}

std::vector<float> Host::execute_f32(
    uint32_t opcode,
    const std::vector<float>& input_a,
    const std::vector<float>& input_b
) {
    if (!input_b.empty() && input_a.size() != input_b.size()) {
        throw std::invalid_argument("execute_f32: input_a and input_b must have the same size");
    }
    std::vector<float> output(input_a.size());
    const int32_t rc = kuhul_glyph_execute_f32(opcode, input_a.data(), input_b.data(), output.data(), input_a.size());
    if (rc != KUHUL_GLYPH_OK) {
        throw std::runtime_error(std::string("execute_f32 failed with code ") + std::to_string(rc));
    }
    return output;
}

std::vector<float> Host::execute_f32(
    uint32_t opcode,
    const std::vector<float>& input_a
) {
    std::vector<float> output(input_a.size());
    const int32_t rc = kuhul_glyph_execute_f32(opcode, input_a.data(), nullptr, output.data(), input_a.size());
    if (rc != KUHUL_GLYPH_OK) {
        throw std::runtime_error(std::string("execute_f32 failed with code ") + std::to_string(rc));
    }
    return output;
}

int32_t Host::matmul_f32(
    const float* lhs,
    const float* rhs,
    float* output,
    uint32_t rows_lhs,
    uint32_t shared_dim,
    uint32_t cols_rhs
) {
    return kuhul_glyph_matmul_f32(lhs, rhs, output, rows_lhs, shared_dim, cols_rhs);
}

std::vector<float> Host::matmul_f32(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    uint32_t rows_lhs,
    uint32_t shared_dim,
    uint32_t cols_rhs
) {
    if (lhs.size() != static_cast<size_t>(rows_lhs * shared_dim)) {
        throw std::invalid_argument("matmul_f32: lhs size mismatch");
    }
    if (rhs.size() != static_cast<size_t>(shared_dim * cols_rhs)) {
        throw std::invalid_argument("matmul_f32: rhs size mismatch");
    }
    std::vector<float> output(static_cast<size_t>(rows_lhs * cols_rhs));
    const int32_t rc = kuhul_glyph_matmul_f32(lhs.data(), rhs.data(), output.data(), rows_lhs, shared_dim, cols_rhs);
    if (rc != KUHUL_GLYPH_OK) {
        throw std::runtime_error(std::string("matmul_f32 failed with code ") + std::to_string(rc));
    }
    return output;
}

std::vector<KuhulGlyphLaneDesc> Host::list_lanes() {
    std::vector<KuhulGlyphLaneDesc> result;
    const uint32_t n = lane_count();
    for (uint32_t i = 0; i < n; ++i) {
        KuhulGlyphLaneDesc desc{};
        if (get_lane(i, &desc)) {
            result.push_back(desc);
        }
    }
    return result;
}

} // namespace KuhulGlyph

extern "C" {

KUHUL_GLYPH_API uint32_t kuhul_glyph_host_abi_version() {
    return KuhulGlyph::Host::abi_version();
}

KUHUL_GLYPH_API int32_t kuhul_glyph_host_execute_f32_vec(
    uint32_t opcode,
    const float* input_a,
    uint64_t count_a,
    const float* input_b,
    uint64_t count_b,
    float* output,
    uint64_t* out_count
) {
    if (!input_a || !output || !out_count) return KUHUL_GLYPH_ERR_NULL;
    const uint64_t count = count_a;
    if (count_b != 0 && count_b != count) return KUHUL_GLYPH_ERR_BAD_SHAPE;
    *out_count = count;
    return kuhul_glyph_execute_f32(opcode, input_a, count_b ? input_b : nullptr, output, count);
}

KUHUL_GLYPH_API int32_t kuhul_glyph_host_matmul_f32_vec(
    const float* lhs,
    const float* rhs,
    float* output,
    uint32_t rows_lhs,
    uint32_t shared_dim,
    uint32_t cols_rhs
) {
    return kuhul_glyph_matmul_f32(lhs, rhs, output, rows_lhs, shared_dim, cols_rhs);
}

} // extern "C"
