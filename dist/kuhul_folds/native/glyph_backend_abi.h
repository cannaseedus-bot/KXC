#pragma once

#include <cstdint>

#if defined(_WIN32)
#  if defined(KUHUL_GLYPH_BACKEND_BUILD)
#    define KUHUL_GLYPH_API __declspec(dllexport)
#  else
#    define KUHUL_GLYPH_API __declspec(dllimport)
#  endif
#else
#  define KUHUL_GLYPH_API
#endif

extern "C" {

enum KuhulGlyphStatus : int32_t {
    KUHUL_GLYPH_OK = 0,
    KUHUL_GLYPH_ERR_NULL = 1,
    KUHUL_GLYPH_ERR_BAD_INDEX = 2,
    KUHUL_GLYPH_ERR_UNSUPPORTED_OPCODE = 3,
    KUHUL_GLYPH_ERR_BAD_ARITY = 4,
    KUHUL_GLYPH_ERR_BAD_SHAPE = 5,
};

enum KuhulGlyphLaneKind : uint32_t {
    KUHUL_GLYPH_LANE_SCALAR = 1,
    KUHUL_GLYPH_LANE_TENSOR = 2,
    KUHUL_GLYPH_LANE_ATTENTION = 3,
    KUHUL_GLYPH_LANE_PHASE = 4,
};

enum KuhulGlyphDType : uint32_t {
    KUHUL_GLYPH_DTYPE_F32 = 1u << 0,
};

enum KuhulGlyphOpcode : uint32_t {
    KUHUL_GLYPH_WO_ADD = 0x02010001u,
    KUHUL_GLYPH_WO_SUB = 0x02010002u,
    KUHUL_GLYPH_WO_MUL = 0x02010003u,
    KUHUL_GLYPH_WO_DIV = 0x02010004u,
    KUHUL_GLYPH_WO_MATMUL = 0x02020000u,
    KUHUL_GLYPH_WO_SILU = 0x02020001u,
    KUHUL_GLYPH_WO_GELU = 0x02020002u,
    KUHUL_GLYPH_WO_BIMODAL_ATTN = 0x02030001u,
    KUHUL_GLYPH_WO_GEODESIC_ATTN = 0x02030002u,
    KUHUL_GLYPH_WO_FOLD_ATTN = 0x02030003u,
    KUHUL_GLYPH_WO_NODE_ATTN = 0x02030004u,
    KUHUL_GLYPH_WO_MICRO_ATTN = 0x02030005u,
};

struct KuhulGlyphLaneDesc {
    uint32_t opcode;
    char name[32];
    char glyph[16];
    uint32_t lane_kind;
    uint32_t arity;
    uint32_t dtype_mask;
    uint32_t flags;
};

KUHUL_GLYPH_API uint32_t kuhul_glyph_abi_version();
KUHUL_GLYPH_API uint32_t kuhul_glyph_lane_count();
KUHUL_GLYPH_API int32_t kuhul_glyph_get_lane(uint32_t index, KuhulGlyphLaneDesc* out_desc);
KUHUL_GLYPH_API int32_t kuhul_glyph_execute_f32(
    uint32_t opcode,
    const float* input_a,
    const float* input_b,
    float* output,
    uint64_t element_count
);
KUHUL_GLYPH_API int32_t kuhul_glyph_matmul_f32(
    const float* lhs,
    const float* rhs,
    float* output,
    uint32_t rows_lhs,
    uint32_t shared_dim,
    uint32_t cols_rhs
);

}
