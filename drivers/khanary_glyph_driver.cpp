//
// khanary_glyph_driver.cpp — Unified K'UHUL glyph + lane opcode registry
//
// 12 phase/fold glyphs  (glyph_contract.json)
// 13 compute lanes       (glyph_ggml_lanes.json)
//
// Flat C ABI. Compiles with MSVC (`cl /LD /EHsc /O2 /I.`).
//

#define KHANARY_GLYPH_EXPORTS
#include "khanary_glyph_driver.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>

namespace {

// ========================================================================
// Phase/fold glyph registry (from glyph_contract.json)
// ========================================================================

struct PhaseEntry {
    uint32_t opcode, token_id, phase_order, flags;
    const char* name;
    const char* glyph;
    const char* role;
    const char* provider;
};

constexpr PhaseEntry PHASES[] = {
    { KHANARY_PHASE_POP,  50257, 0, 0x01, "POP",   "[Pop]",               "observe",             "micronaut-base" },
    { KHANARY_PHASE_WO,   50258, 1, 0x01, "WO",    "[Wo]",                "schedule/allocate",   "micronaut-factory" },
    { KHANARY_PHASE_YAX,  50259, 2, 0x01, "YAX",   "[Yax]",               "branch/plan",         "micronaut-base" },
    { KHANARY_PHASE_SEK,  50260, 3, 0x01, "SEK",   "[Sek]",               "execute/dispatch",    "micronaut-coder" },
    { KHANARY_PHASE_CHEN, 50261, 4, 0x01, "CHEN",  "[Ch'en]",             "verify/collapse-emit","micronaut-base" },
    { KHANARY_PHASE_XUL,  50262, 5, 0x01, "XUL",   "[Xul]",               "emit/finalize",       "json-runtime" },
    { KHANARY_SEP,        50263, 255, 0x02, "SEP",   "[Sep]",              "boundary",            "kuhul-engine" },
    { KHANARY_FOLD_0,     50264, 0, 0x04, "FOLD_0","[Fold:artifact.creation]",     "artifact.creation",     "micronaut-factory" },
    { KHANARY_FOLD_1,     50265, 1, 0x04, "FOLD_1","[Fold:code.generation]",       "code.generation",       "micronaut-coder" },
    { KHANARY_FOLD_2,     50266, 2, 0x04, "FOLD_2","[Fold:general.interpolation]", "general.interpolation", "micronaut-base" },
    { KHANARY_FOLD_3,     50267, 3, 0x04, "FOLD_3","[Fold:reasoning.math]",        "reasoning.math",        "kuhul-engine" },
    { KHANARY_FOLD_4,     50268, 4, 0x04, "FOLD_4","[Fold:tool.routing]",          "tool.routing",          "kuhul-engine" },
};

// ========================================================================
// Compute lane registry (from glyph_ggml_lanes.json)
// ========================================================================

struct LaneEntry {
    uint32_t opcode;
    uint32_t lane_kind;
    uint32_t fold_class;
    uint32_t arity;
    uint32_t dtype_mask;
    const char* name;
    const char* glyph;
    const char* ggml_op;
    const char* webgpu_shader;
};

constexpr uint32_t DTYPE_F32  = 1u << 0;
constexpr uint32_t DTYPE_F16  = 1u << 1;
constexpr uint32_t DTYPE_BF16 = 1u << 2;

constexpr LaneEntry LANES[] = {
    { KHANARY_LANE_ADD,        KHANARY_LANE_SCALAR, KHANARY_FOLD_ARC,    2, DTYPE_F32 | DTYPE_F16 | DTYPE_BF16, "WO_ADD",        "+",  "ADD",            "wgsl-shaders/binary.wgsl" },
    { KHANARY_LANE_SUB,        KHANARY_LANE_SCALAR, KHANARY_FOLD_ARC,    2, DTYPE_F32 | DTYPE_F16 | DTYPE_BF16, "WO_SUB",        "-",  "SUB",            "wgsl-shaders/binary.wgsl" },
    { KHANARY_LANE_MUL,        KHANARY_LANE_SCALAR, KHANARY_FOLD_ARC,    2, DTYPE_F32 | DTYPE_F16 | DTYPE_BF16, "WO_MUL",        "*",  "MUL",            "wgsl-shaders/binary.wgsl" },
    { KHANARY_LANE_DIV,        KHANARY_LANE_SCALAR, KHANARY_FOLD_ARC,    2, DTYPE_F32 | DTYPE_F16 | DTYPE_BF16, "WO_DIV",        "/",  "DIV",            "wgsl-shaders/binary.wgsl" },
    { KHANARY_LANE_MATMUL,     KHANARY_LANE_TENSOR, KHANARY_FOLD_TENSOR, 2, DTYPE_F32 | DTYPE_F16 | DTYPE_BF16, "WO_MATMUL",     "⨀", "MUL_MAT",        "wgsl-shaders/mul_mat.wgsl" },
    { KHANARY_LANE_SOFTMAX,    KHANARY_LANE_ATTENTION, KHANARY_FOLD_TENSOR, 1, DTYPE_F32 | DTYPE_F16 | DTYPE_BF16, "WO_SOFTMAX",    "⟐", "SOFT_MAX",       "wgsl-shaders/soft_max.wgsl" },
    { KHANARY_LANE_RMS_NORM,   KHANARY_LANE_TENSOR, KHANARY_FOLD_TENSOR, 1, DTYPE_F32 | DTYPE_F16 | DTYPE_BF16, "WO_RMS_NORM",   "⌁", "RMS_NORM",       "wgsl-shaders/rms_norm_mul.wgsl" },
    { KHANARY_LANE_ROPE,       KHANARY_LANE_ATTENTION, KHANARY_FOLD_TENSOR, 2, DTYPE_F32 | DTYPE_F16 | DTYPE_BF16, "WO_ROTARY",     "⌬", "ROPE",           "wgsl-shaders/rope.wgsl" },
    { KHANARY_LANE_FLASH_ATTN, KHANARY_LANE_ATTENTION, KHANARY_FOLD_KERNEL, 4, DTYPE_F32 | DTYPE_F16 | DTYPE_BF16, "WO_FLASH_ATTN", "✦", "FLASH_ATTN_EXT", "wgsl-shaders/flash_attn.wgsl" },
    { KHANARY_LANE_SILU,       KHANARY_LANE_TENSOR, KHANARY_FOLD_TENSOR, 1, DTYPE_F32 | DTYPE_F16 | DTYPE_BF16, "WO_SILU",       "σ", "SILU",           "wgsl-shaders/unary.wgsl" },
    { KHANARY_LANE_GELU,       KHANARY_LANE_TENSOR, KHANARY_FOLD_TENSOR, 1, DTYPE_F32 | DTYPE_F16 | DTYPE_BF16, "WO_GELU",       "Φ", "GELU",           "wgsl-shaders/unary.wgsl" },
    { KHANARY_LANE_CONCAT,     KHANARY_LANE_TENSOR, KHANARY_FOLD_TENSOR, 0, DTYPE_F32 | DTYPE_F16 | DTYPE_BF16, "WO_CONCAT",     "⧉", "CONCAT",         "wgsl-shaders/concat.wgsl" },
    { KHANARY_LANE_PHASE_GATE, KHANARY_LANE_PHASE,  KHANARY_FOLD_PHASE,  1, 0u,                                "PHASE_GATE",    "⟁", nullptr,          nullptr },
};

constexpr uint32_t PHASE_COUNT = static_cast<uint32_t>(std::size(PHASES));
constexpr uint32_t LANE_COUNT  = static_cast<uint32_t>(std::size(LANES));

// ========================================================================
// Helpers
// ========================================================================

std::string jsonEscape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

std::string jsonString(const std::string& obj, const char* key) {
    std::string marker = std::string("\"") + key + "\"";
    size_t pos = obj.find(marker);
    if (pos == std::string::npos) return {};
    size_t colon = obj.find(':', pos + marker.size());
    size_t quote = obj.find('"', colon == std::string::npos ? colon : colon + 1);
    if (colon == std::string::npos || quote == std::string::npos) return {};
    size_t end = obj.find('"', quote + 1);
    return end == std::string::npos ? std::string{} : obj.substr(quote + 1, end - quote - 1);
}

} // namespace

// ========================================================================
// Phase/fold glyph API
// ========================================================================

extern "C" {

KHANARY_GLYPH_API uint32_t kd_glyph_phase_count() {
    return PHASE_COUNT;
}

KHANARY_GLYPH_API int32_t kd_glyph_get_phase(uint32_t index, KhanaryPhaseDesc* out_desc) {
    if (!out_desc) return -1;
    if (index >= PHASE_COUNT) return -2;
    const auto& e = PHASES[index];
    out_desc->opcode      = e.opcode;
    out_desc->token_id    = e.token_id;
    out_desc->phase_order = e.phase_order;
    out_desc->flags       = e.flags;
    std::strncpy(out_desc->name,  e.name,  sizeof(out_desc->name) - 1);
    std::strncpy(out_desc->glyph, e.glyph, sizeof(out_desc->glyph) - 1);
    std::strncpy(out_desc->role,  e.role,  sizeof(out_desc->role) - 1);
    std::strncpy(out_desc->provider, e.provider, sizeof(out_desc->provider) - 1);
    return 0;
}

KHANARY_GLYPH_API KhanaryGlyphResult kd_glyph_dispatch_phase(
    uint32_t opcode, const char* provider_json
) {
    KhanaryGlyphResult r{};
    r.opcode = opcode;

    const PhaseEntry* entry = nullptr;
    for (const auto& e : PHASES) {
        if (e.opcode == opcode) { entry = &e; break; }
    }
    if (!entry) {
        std::strncpy(r.name, "UNKNOWN", sizeof(r.name) - 1);
        std::strncpy(r.status, "not_registered", sizeof(r.status) - 1);
        std::strncpy(r.detail, "opcode_not_found", sizeof(r.detail) - 1);
        return r;
    }

    std::strncpy(r.name, entry->name, sizeof(r.name) - 1);
    std::strncpy(r.status, "admitted", sizeof(r.status) - 1);

    std::ostringstream d;
    d << "phase=" << entry->name << " token=" << entry->token_id
      << " role=" << entry->role << " provider=" << entry->provider;
    if (provider_json && provider_json[0]) {
        std::string pj(provider_json);
        std::string action = jsonString(pj, "action");
        std::string prov = jsonString(pj, "provider");
        if (!action.empty()) d << " action=" << action;
        if (!prov.empty()) d << " dispatch_to=" << prov;
    }
    std::strncpy(r.detail, d.str().c_str(), sizeof(r.detail) - 1);
    return r;
}

KHANARY_GLYPH_API char* kd_glyph_process_sequence(const uint32_t* opcodes, uint32_t count) {
    std::ostringstream out;
    out << "[";
    for (uint32_t i = 0; i < count; ++i) {
        if (i > 0) out << ",";
        const PhaseEntry* e = nullptr;
        for (const auto& p : PHASES) { if (p.opcode == opcodes[i]) { e = &p; break; } }
        if (e) {
            out << "{\"opcode\":\"0x" << std::hex << opcodes[i] << std::dec
                << "\",\"name\":\"" << jsonEscape(e->name)
                << "\",\"token\":" << e->token_id
                << ",\"glyph\":\"" << jsonEscape(e->glyph)
                << "\",\"role\":\"" << jsonEscape(e->role)
                << "\",\"provider\":\"" << jsonEscape(e->provider)
                << "\",\"status\":\"admitted\"}";
        } else {
            out << "{\"opcode\":\"0x" << std::hex << opcodes[i] << std::dec
                << "\",\"name\":\"UNKNOWN\",\"status\":\"not_registered\"}";
        }
    }
    out << "]";
    std::string json = out.str();
    char* result = new char[json.size() + 1];
    std::memcpy(result, json.data(), json.size());
    result[json.size()] = '\0';
    return result;
}

// ========================================================================
// Compute lane API
// ========================================================================

KHANARY_GLYPH_API uint32_t kd_glyph_lane_count() {
    return LANE_COUNT;
}

KHANARY_GLYPH_API int32_t kd_glyph_get_lane(uint32_t index, KhanaryLaneDesc* out_desc) {
    if (!out_desc) return -1;
    if (index >= LANE_COUNT) return -2;
    const auto& e = LANES[index];
    out_desc->opcode     = e.opcode;
    out_desc->lane_kind  = e.lane_kind;
    out_desc->fold_class = e.fold_class;
    out_desc->arity      = e.arity;
    out_desc->dtype_mask = e.dtype_mask;
    std::strncpy(out_desc->name,          e.name,          sizeof(out_desc->name) - 1);
    std::strncpy(out_desc->glyph,         e.glyph,         sizeof(out_desc->glyph) - 1);
    std::strncpy(out_desc->ggml_op,       e.ggml_op ? e.ggml_op : "", sizeof(out_desc->ggml_op) - 1);
    std::strncpy(out_desc->webgpu_shader, e.webgpu_shader ? e.webgpu_shader : "",
                 sizeof(out_desc->webgpu_shader) - 1);
    return 0;
}

KHANARY_GLYPH_API int32_t kd_glyph_find_lane_by_opcode(uint32_t opcode, KhanaryLaneDesc* out_desc) {
    for (uint32_t i = 0; i < LANE_COUNT; ++i) {
        if (LANES[i].opcode == opcode) return kd_glyph_get_lane(i, out_desc);
    }
    return -3;
}

KHANARY_GLYPH_API int32_t kd_glyph_find_lane_by_ggml(const char* ggml_op, KhanaryLaneDesc* out_desc) {
    if (!ggml_op) return -1;
    for (uint32_t i = 0; i < LANE_COUNT; ++i) {
        if (LANES[i].ggml_op && std::strcmp(LANES[i].ggml_op, ggml_op) == 0)
            return kd_glyph_get_lane(i, out_desc);
    }
    return -3;
}

// ========================================================================
// Unified query
// ========================================================================

KHANARY_GLYPH_API uint32_t kd_glyph_total_entries() {
    return PHASE_COUNT + LANE_COUNT;
}

KHANARY_GLYPH_API char* kd_glyph_dump_registry() {
    std::ostringstream out;
    out << "{\"phases\":[";
    for (uint32_t i = 0; i < PHASE_COUNT; ++i) {
        if (i > 0) out << ",";
        const auto& e = PHASES[i];
        out << "{\"opcode\":\"0x" << std::hex << e.opcode << std::dec
            << "\",\"name\":\"" << jsonEscape(e.name)
            << "\",\"token\":" << e.token_id
            << ",\"glyph\":\"" << jsonEscape(e.glyph)
            << "\",\"role\":\"" << jsonEscape(e.role)
            << "\",\"provider\":\"" << jsonEscape(e.provider)
            << "\",\"kind\":\"phase\"}";
    }
    out << "],\"lanes\":[";
    for (uint32_t i = 0; i < LANE_COUNT; ++i) {
        if (i > 0) out << ",";
        const auto& e = LANES[i];
        out << "{\"opcode\":\"0x" << std::hex << e.opcode << std::dec
            << "\",\"name\":\"" << jsonEscape(e.name)
            << "\",\"glyph\":\"" << jsonEscape(e.glyph)
            << "\",\"ggml_op\":\"" << (e.ggml_op ? jsonEscape(e.ggml_op) : "null")
            << "\",\"lane_kind\":" << e.lane_kind
            << ",\"fold_class\":" << e.fold_class
            << ",\"arity\":" << e.arity
            << ",\"webgpu_shader\":\"" << (e.webgpu_shader ? jsonEscape(e.webgpu_shader) : "null")
            << "\",\"kind\":\"lane\"}";
    }
    out << "],\"total\":" << (PHASE_COUNT + LANE_COUNT) << "}";
    std::string json = out.str();
    char* result = new char[json.size() + 1];
    std::memcpy(result, json.data(), json.size());
    result[json.size()] = '\0';
    return result;
}

KHANARY_GLYPH_API void kd_glyph_free_string(char* str) {
    delete[] str;
}

} // extern "C"
