#pragma once
//
// khanary_glyph_driver.h — Unified K'UHUL glyph + lane opcode registry
//
// Combines:
//   12 phase/fold glyphs  (glyph_contract.json, tokens 50257–50268)
//   13 compute lanes       (glyph_ggml_lanes.json, GGML → K'UHUL opcode mapping)
//
// Flat C ABI for ffi-napi loading. Compiles with MSVC.
//

#ifdef KHANARY_GLYPH_EXPORTS
#define KHANARY_GLYPH_API __declspec(dllexport)
#else
#define KHANARY_GLYPH_API __declspec(dllimport)
#endif

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ── Lane classes ──────────────────────────────────────────────────────

enum KhanaryLaneKind : uint32_t {
    KHANARY_LANE_SCALAR    = 1,  // element-wise arithmetic
    KHANARY_LANE_TENSOR    = 2,  // matmul, norm, activation
    KHANARY_LANE_ATTENTION = 3,  // softmax, rope, flash-attn
    KHANARY_LANE_PHASE     = 4,  // execution phase gate
};

enum KhanaryFoldClass : uint32_t {
    KHANARY_FOLD_ARC    = 0x01,  // scalar arithmetic fold
    KHANARY_FOLD_TENSOR = 0x02,  // tensor compute fold
    KHANARY_FOLD_KERNEL = 0x03,  // fused kernel fold
    KHANARY_FOLD_PHASE  = 0x04,  // control-flow phase fold
};

// ── Phase/fold glyph opcodes (from glyph_contract.json) ───────────────

enum KhanaryGlyphPhaseOpcode : uint32_t {
    KHANARY_PHASE_POP   = 0x5000,  // observe
    KHANARY_PHASE_WO    = 0x5001,  // schedule/allocate
    KHANARY_PHASE_YAX   = 0x5002,  // branch/plan
    KHANARY_PHASE_SEK   = 0x5003,  // execute/dispatch
    KHANARY_PHASE_CHEN  = 0x5004,  // verify/collapse-emit
    KHANARY_PHASE_XUL   = 0x5005,  // emit/finalize
    KHANARY_SEP         = 0x5006,  // prompt/response boundary
    KHANARY_FOLD_0      = 0x6000,  // artifact.creation
    KHANARY_FOLD_1      = 0x6001,  // code.generation
    KHANARY_FOLD_2      = 0x6002,  // general.interpolation
    KHANARY_FOLD_3      = 0x6003,  // reasoning.math
    KHANARY_FOLD_4      = 0x6004,  // tool.routing
};

// ── Compute lane opcodes (from glyph_ggml_lanes.json) ─────────────────

enum KhanaryLaneOpcode : uint32_t {
    KHANARY_LANE_ADD         = 0x02010001,  // scalar add
    KHANARY_LANE_SUB         = 0x02010002,  // scalar sub
    KHANARY_LANE_MUL         = 0x02010003,  // scalar mul
    KHANARY_LANE_DIV         = 0x02010004,  // scalar div
    KHANARY_LANE_MATMUL      = 0x02020000,  // tensor matmul
    KHANARY_LANE_SOFTMAX     = 0x02030001,  // attention softmax
    KHANARY_LANE_RMS_NORM    = 0x02040001,  // tensor rms_norm
    KHANARY_LANE_ROPE        = 0x02030002,  // attention rope
    KHANARY_LANE_FLASH_ATTN  = 0x02030003,  // attention flash-attn
    KHANARY_LANE_SILU        = 0x02020001,  // tensor silu
    KHANARY_LANE_GELU        = 0x02020002,  // tensor gelu
    KHANARY_LANE_CONCAT      = 0x02050001,  // tensor concat
    KHANARY_LANE_PHASE_GATE  = 0x00010001,  // phase control gate
};

// ── Descriptors ───────────────────────────────────────────────────────

struct KhanaryPhaseDesc {
    uint32_t opcode;
    uint32_t token_id;
    char name[32];
    char glyph[32];
    uint32_t phase_order;
    uint32_t flags;          // 0x01=PHASE, 0x02=SEP, 0x04=FOLD
    char role[64];
    char provider[32];
};

struct KhanaryLaneDesc {
    uint32_t opcode;
    char name[32];           // e.g. "WO_MATMUL", "WO_GELU"
    char glyph[8];           // e.g. "⨀", "Φ", "✦", "⟁"
    char ggml_op[32];        // e.g. "MUL_MAT", "GELU", "FLASH_ATTN_EXT"
    uint32_t lane_kind;      // KhanaryLaneKind
    uint32_t fold_class;     // KhanaryFoldClass
    uint32_t arity;
    char webgpu_shader[128]; // e.g. "wgsl-shaders/mul_mat.wgsl"
    uint32_t dtype_mask;     // 1<<0 = f32, 1<<1 = f16, 1<<2 = bf16
};

struct KhanaryGlyphResult {
    uint32_t opcode;
    char name[32];
    char status[32];
    char detail[512];
};

// ── Phase/fold glyph API ──────────────────────────────────────────────

KHANARY_GLYPH_API uint32_t kd_glyph_phase_count();
KHANARY_GLYPH_API int32_t  kd_glyph_get_phase(uint32_t index, KhanaryPhaseDesc* out_desc);
KHANARY_GLYPH_API KhanaryGlyphResult kd_glyph_dispatch_phase(uint32_t opcode, const char* provider_json);
KHANARY_GLYPH_API char*    kd_glyph_process_sequence(const uint32_t* opcodes, uint32_t count);

// ── Compute lane API ──────────────────────────────────────────────────

KHANARY_GLYPH_API uint32_t kd_glyph_lane_count();
KHANARY_GLYPH_API int32_t  kd_glyph_get_lane(uint32_t index, KhanaryLaneDesc* out_desc);
KHANARY_GLYPH_API int32_t  kd_glyph_find_lane_by_opcode(uint32_t opcode, KhanaryLaneDesc* out_desc);
KHANARY_GLYPH_API int32_t  kd_glyph_find_lane_by_ggml(const char* ggml_op, KhanaryLaneDesc* out_desc);

// ── Unified query ─────────────────────────────────────────────────────

KHANARY_GLYPH_API uint32_t kd_glyph_total_entries();
KHANARY_GLYPH_API char*    kd_glyph_dump_registry();  // JSON dump of full registry

KHANARY_GLYPH_API void     kd_glyph_free_string(char* str);

#ifdef __cplusplus
}
#endif
