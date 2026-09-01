#pragma once

#include "glyph_backend_abi.h"
#include <vector>
#include <string>
#include <functional>

namespace KuhulGlyph {

// C++ wrapper around the Kuhul glyph backend ABI.
class Host {
public:
    Host();

    // ABI metadata
    static uint32_t abi_version();
    static uint32_t lane_count();

    // Get a lane descriptor by index.
    bool get_lane(uint32_t index, KuhulGlyphLaneDesc* out_desc);

    // Element-wise f32 execution. input_b may be null for unary lanes.
    int32_t execute_f32(
        uint32_t opcode,
        const float* input_a,
        const float* input_b,
        float* output,
        uint64_t element_count
    );

    // Convenience wrappers that allocate vectors for you.
    std::vector<float> execute_f32(
        uint32_t opcode,
        const std::vector<float>& input_a,
        const std::vector<float>& input_b
    );
    std::vector<float> execute_f32(
        uint32_t opcode,
        const std::vector<float>& input_a
    );

    // Row-major matrix multiply: lhs(rows x shared) @ rhs(shared x cols).
    int32_t matmul_f32(
        const float* lhs,
        const float* rhs,
        float* output,
        uint32_t rows_lhs,
        uint32_t shared_dim,
        uint32_t cols_rhs
    );

    std::vector<float> matmul_f32(
        const std::vector<float>& lhs,
        const std::vector<float>& rhs,
        uint32_t rows_lhs,
        uint32_t shared_dim,
        uint32_t cols_rhs
    );

    // List all lane descriptors.
    std::vector<KuhulGlyphLaneDesc> list_lanes();
};

} // namespace KuhulGlyph

// Additional C-compatible exports beyond glyph_backend_abi.h
extern "C" {
    KUHUL_GLYPH_API uint32_t kuhul_glyph_host_abi_version();
    KUHUL_GLYPH_API int32_t  kuhul_glyph_host_execute_f32_vec(
        uint32_t opcode,
        const float* input_a,
        uint64_t count_a,
        const float* input_b,
        uint64_t count_b,
        float* output,
        uint64_t* out_count
    );
    KUHUL_GLYPH_API int32_t kuhul_glyph_host_matmul_f32_vec(
        const float* lhs,
        const float* rhs,
        float* output,
        uint32_t rows_lhs,
        uint32_t shared_dim,
        uint32_t cols_rhs
    );
}
