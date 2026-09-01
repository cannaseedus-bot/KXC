#pragma once
//
// qwen_infer_driver.h — Qwen 1.8B D3D11 compute shader inference backend
//
// Qwen architecture differences from GPT-2:
//   - Rotary Position Embeddings (RoPE) instead of learned position embeddings
//   - RMSNorm instead of LayerNorm
//   - SiLU activation instead of GELU
//   - No learned position embedding table (wpe)
//   - Attention: Q/K/V projection with RoPE applied before Q·K^T
//
// Flat C ABI. Compiles with MSVC. Loaded by kuhul-server via ffi-napi.
//

#ifdef QWEN_INFER_EXPORTS
#define QWEN_INFER_API __declspec(dllexport)
#else
#define QWEN_INFER_API __declspec(dllimport)
#endif

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ── Shader op IDs (maps to D3D11 compute shader dispatch) ────────────

enum QwenInferOp : uint32_t {
    QWEN_OP_EMBED      = 0,  // token embedding (no position — RoPE handles it)
    QWEN_OP_RMS_NORM   = 1,  // RMS normalization
    QWEN_OP_MATMUL     = 2,  // matrix multiply (GEMM)
    QWEN_OP_ROPE       = 3,  // rotary position embedding (applied to Q and K)
    QWEN_OP_ATTENTION  = 4,  // Q·K^T + softmax + V accumulation
    QWEN_OP_SILU       = 5,  // SiLU activation
    QWEN_OP_ADD_BIAS   = 6,  // add bias vector
    QWEN_OP_RESIDUAL   = 7,  // residual add (x + y)
    QWEN_OP_LM_HEAD    = 8,  // unembedding (logits)
    QWEN_OP_COUNT      = 9,
};

// ── Model config (Qwen 1.8B) ─────────────────────────────────────────

struct QwenInferConfig {
    uint32_t vocab_size;     // 151936
    uint32_t max_seq_len;    // 32768
    uint32_t embed_dim;      // 2048
    uint32_t num_heads;      // 16
    uint32_t num_kv_heads;   // 16 (no GQA in 1.8B)
    uint32_t num_layers;     // 24
    uint32_t ff_dim;         // 5504
    uint32_t head_dim;       // 128
    float    rope_theta;     // 1000000.0f
};

// ── C ABI exports ────────────────────────────────────────────────────

/// Create inference context. config is JSON or NULL for Qwen 1.8B defaults.
QWEN_INFER_API void* qw_create(const char* config_json);

/// Destroy context.
QWEN_INFER_API void qw_destroy(void* ctx);

/// Load weights from GGUF. Returns 1 on success.
QWEN_INFER_API int qw_load_model(void* ctx, const char* gguf_path,
                                  char* error_buf, int error_buf_size);

/// Run forward pass on token sequence. Returns logits JSON.
QWEN_INFER_API char* qw_forward(void* ctx, const int32_t* tokens,
                                 uint32_t token_count);

/// Single-token sample. Returns JSON: {"token":1234,"logit":-0.5,"top_k":[...]}
QWEN_INFER_API char* qw_sample(void* ctx, const int32_t* tokens,
                                uint32_t token_count, int k);

/// Get model config as JSON.
QWEN_INFER_API char* qw_get_config(void* ctx);

/// Check if D3D11 compute shader support is available (cs_5_0).
QWEN_INFER_API int qw_probe(void);

QWEN_INFER_API void qw_free_string(char* str);

#ifdef __cplusplus
}
#endif
