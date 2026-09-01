#include "glyph_backend_abi.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

constexpr uint32_t ABI_VERSION = 2;

constexpr KuhulGlyphLaneDesc LANES[] = {
    {KUHUL_GLYPH_WO_ADD, "WO_ADD", "+", KUHUL_GLYPH_LANE_SCALAR, 2, KUHUL_GLYPH_DTYPE_F32, 0},
    {KUHUL_GLYPH_WO_SUB, "WO_SUB", "-", KUHUL_GLYPH_LANE_SCALAR, 2, KUHUL_GLYPH_DTYPE_F32, 0},
    {KUHUL_GLYPH_WO_MUL, "WO_MUL", "*", KUHUL_GLYPH_LANE_SCALAR, 2, KUHUL_GLYPH_DTYPE_F32, 0},
    {KUHUL_GLYPH_WO_DIV, "WO_DIV", "/", KUHUL_GLYPH_LANE_SCALAR, 2, KUHUL_GLYPH_DTYPE_F32, 0},
    {KUHUL_GLYPH_WO_MATMUL, "WO_MATMUL", "matmul", KUHUL_GLYPH_LANE_TENSOR, 2, KUHUL_GLYPH_DTYPE_F32, 0},
    {KUHUL_GLYPH_WO_SILU, "WO_SILU", "silu", KUHUL_GLYPH_LANE_TENSOR, 1, KUHUL_GLYPH_DTYPE_F32, 0},
    {KUHUL_GLYPH_WO_GELU, "WO_GELU", "gelu", KUHUL_GLYPH_LANE_TENSOR, 1, KUHUL_GLYPH_DTYPE_F32, 0},
};

constexpr float GELU_K = 0.7978845608028654f;
constexpr float GELU_C = 0.044715f;

bool needs_input_b(uint32_t opcode) {
    return opcode == KUHUL_GLYPH_WO_ADD ||
           opcode == KUHUL_GLYPH_WO_SUB ||
           opcode == KUHUL_GLYPH_WO_MUL ||
           opcode == KUHUL_GLYPH_WO_DIV;
}

} // namespace

extern "C" {

KUHUL_GLYPH_API uint32_t kuhul_glyph_abi_version() {
    return ABI_VERSION;
}

KUHUL_GLYPH_API uint32_t kuhul_glyph_lane_count() {
    return static_cast<uint32_t>(std::size(LANES));
}

KUHUL_GLYPH_API int32_t kuhul_glyph_get_lane(uint32_t index, KuhulGlyphLaneDesc* out_desc) {
    if (!out_desc) {
        return KUHUL_GLYPH_ERR_NULL;
    }
    if (index >= kuhul_glyph_lane_count()) {
        return KUHUL_GLYPH_ERR_BAD_INDEX;
    }
    std::memcpy(out_desc, &LANES[index], sizeof(KuhulGlyphLaneDesc));
    return KUHUL_GLYPH_OK;
}

KUHUL_GLYPH_API int32_t kuhul_glyph_execute_f32(
    uint32_t opcode,
    const float* input_a,
    const float* input_b,
    float* output,
    uint64_t element_count
) {
    if (!input_a || !output || (needs_input_b(opcode) && !input_b)) {
        return KUHUL_GLYPH_ERR_NULL;
    }

    switch (opcode) {
        case KUHUL_GLYPH_WO_ADD:
            for (uint64_t i = 0; i < element_count; ++i) output[i] = input_a[i] + input_b[i];
            return KUHUL_GLYPH_OK;
        case KUHUL_GLYPH_WO_SUB:
            for (uint64_t i = 0; i < element_count; ++i) output[i] = input_a[i] - input_b[i];
            return KUHUL_GLYPH_OK;
        case KUHUL_GLYPH_WO_MUL:
            for (uint64_t i = 0; i < element_count; ++i) output[i] = input_a[i] * input_b[i];
            return KUHUL_GLYPH_OK;
        case KUHUL_GLYPH_WO_DIV:
            for (uint64_t i = 0; i < element_count; ++i) output[i] = input_b[i] == 0.0f ? 0.0f : input_a[i] / input_b[i];
            return KUHUL_GLYPH_OK;
        case KUHUL_GLYPH_WO_MATMUL:
            return KUHUL_GLYPH_ERR_BAD_ARITY;
        case KUHUL_GLYPH_WO_SILU:
            for (uint64_t i = 0; i < element_count; ++i) output[i] = input_a[i] / (1.0f + std::exp(-input_a[i]));
            return KUHUL_GLYPH_OK;
        case KUHUL_GLYPH_WO_GELU:
            for (uint64_t i = 0; i < element_count; ++i) {
                const float x = input_a[i];
                output[i] = 0.5f * x * (1.0f + std::tanh(GELU_K * (x + GELU_C * x * x * x)));
            }
            return KUHUL_GLYPH_OK;
        default:
            return KUHUL_GLYPH_ERR_UNSUPPORTED_OPCODE;
    }
}

KUHUL_GLYPH_API int32_t kuhul_glyph_matmul_f32(
    const float* lhs,
    const float* rhs,
    float* output,
    uint32_t rows_lhs,
    uint32_t shared_dim,
    uint32_t cols_rhs
) {
    if (!lhs || !rhs || !output) {
        return KUHUL_GLYPH_ERR_NULL;
    }
    if (rows_lhs == 0 || shared_dim == 0 || cols_rhs == 0) {
        return KUHUL_GLYPH_ERR_BAD_SHAPE;
    }

    for (uint32_t r = 0; r < rows_lhs; ++r) {
        for (uint32_t c = 0; c < cols_rhs; ++c) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < shared_dim; ++k) {
                sum += lhs[r * shared_dim + k] * rhs[k * cols_rhs + c];
            }
            output[r * cols_rhs + c] = sum;
        }
    }
    return KUHUL_GLYPH_OK;
}

}
