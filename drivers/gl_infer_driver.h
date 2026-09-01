#pragma once
//
// gl_infer_driver.h — OpenGL 4.3 compute shader inference backend
//
// KHANARY's universal GPU path: runs on every GPU since 2012 via
// GL_ARB_compute_shader + SSBOs. No CUDA, no Metal, no Vulkan, no DirectML.
// Just OpenGL 4.3 — present on every Intel, AMD, and NVIDIA GPU from the
// last decade.
//
// Flat C ABI. Compiles with MSVC. Loaded by kuhul-server via ffi-napi.
//

#ifdef GL_INFER_EXPORTS
#define GL_INFER_API __declspec(dllexport)
#else
#define GL_INFER_API __declspec(dllimport)
#endif

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ── Shader op IDs (maps to GLSL compute shader dispatch) ─────────────

enum GLInferOp : uint32_t {
    GL_INFER_OP_EMBED      = 0,  // token + position embedding
    GL_INFER_OP_LAYERNORM  = 1,  // layer normalization
    GL_INFER_OP_MATMUL     = 2,  // matrix multiply (GEMM)
    GL_INFER_OP_ATTENTION  = 3,  // QK^T + softmax + V
    GL_INFER_OP_GELU       = 4,  // GELU activation
    GL_INFER_OP_ADD_BIAS   = 5,  // add bias vector
    GL_INFER_OP_RESIDUAL   = 6,  // residual add (x + y)
    GL_INFER_OP_LM_HEAD    = 7,  // unembedding (logits)
    GL_INFER_OP_COUNT      = 8,
};

// ── Model config ─────────────────────────────────────────────────────

struct GLInferConfig {
    uint32_t vocab_size;     // 50257 for GPT-2
    uint32_t max_seq_len;    // 1024
    uint32_t embed_dim;      // 768
    uint32_t num_heads;      // 12
    uint32_t num_layers;     // 12
    uint32_t ff_dim;         // 3072
};

// ── C ABI exports ────────────────────────────────────────────────────

/// Create inference context. config is a JSON string or NULL for defaults.
GL_INFER_API void* gli_create(const char* config_json);

/// Destroy context.
GL_INFER_API void gli_destroy(void* ctx);

/// Load weights from a GGUF file. Returns 1 on success.
/// Writes error to error_buf on failure.
GL_INFER_API int gli_load_model(void* ctx, const char* gguf_path,
                                 char* error_buf, int error_buf_size);

/// Run a full forward pass on a token sequence. Returns logits as JSON
/// string (array of vocab_size floats). Caller frees with gli_free_string.
GL_INFER_API char* gli_forward(void* ctx, const int32_t* tokens,
                                uint32_t token_count);

/// Single-token inference: run forward pass, return top-k token IDs.
/// k: number of top tokens to return (default 1 for greedy).
/// Returns JSON: {"token": 1234, "logit": -0.5, "top_k": [...]}
GL_INFER_API char* gli_sample(void* ctx, const int32_t* tokens,
                               uint32_t token_count, int k);

/// Get model config as JSON.
GL_INFER_API char* gli_get_config(void* ctx);

/// Check if OpenGL 4.3 is available on this system.
GL_INFER_API int gli_probe(void);

/// Get GPU info string (vendor, renderer, GL version).
GL_INFER_API char* gli_gpu_info(void);

GL_INFER_API void gli_free_string(char* str);

#ifdef __cplusplus
}
#endif
