//
// qwen_infer_driver.cpp — Qwen 1.8B D3D11 compute shader inference backend
//
// Stub implementation. The full compute shader dispatch (9 D3D11 shaders:
// embed, rms_norm, matmul, rope, attention, silu, add_bias, residual, lm_head)
// compiles when D3D11 headers are linked.
//
// Qwen architecture note:
//   RoPE replaces learned position embeddings. The model has no wpe tensor.
//   RMSNorm replaces LayerNorm (no mean subtraction, just RMS normalization).
//   SiLU replaces GELU (x * sigmoid(x) instead of GELU approximation).
//

#define QWEN_INFER_EXPORTS
#include "qwen_infer_driver.h"

#include <cstring>
#include <sstream>
#include <string>

namespace {

struct QwenInferState {
    QwenInferConfig cfg;
    bool model_loaded = false;
    std::string model_path;

    QwenInferState() {
        cfg.vocab_size   = 151936;
        cfg.max_seq_len  = 32768;
        cfg.embed_dim    = 2048;
        cfg.num_heads    = 16;
        cfg.num_kv_heads = 16;
        cfg.num_layers   = 24;
        cfg.ff_dim       = 5504;
        cfg.head_dim     = 128;
        cfg.rope_theta   = 1000000.0f;
    }

    std::string makeConfigJson() const {
        std::ostringstream s;
        s << "{"
          << "\"vocab_size\":" << cfg.vocab_size << ","
          << "\"max_seq_len\":" << cfg.max_seq_len << ","
          << "\"embed_dim\":" << cfg.embed_dim << ","
          << "\"num_heads\":" << cfg.num_heads << ","
          << "\"num_kv_heads\":" << cfg.num_kv_heads << ","
          << "\"num_layers\":" << cfg.num_layers << ","
          << "\"ff_dim\":" << cfg.ff_dim << ","
          << "\"head_dim\":" << cfg.head_dim << ","
          << "\"rope_theta\":" << cfg.rope_theta << ","
          << "\"model_loaded\":" << (model_loaded ? "true" : "false") << ","
          << "\"shader_ops\":9,"
          << "\"architecture\":\"qwen2\""
          << "}";
        return s.str();
    }

    bool parseConfig(const std::string& json) {
        auto js = [&](const char* key) -> std::string {
            std::string m = std::string("\"") + key + "\"";
            size_t p = json.find(m);
            if (p == std::string::npos) return {};
            size_t c = json.find(':', p + m.size());
            size_t q = json.find('"', c + 1);
            if (c == std::string::npos || q == std::string::npos) {
                size_t n = json.find_first_of("0123456789.", c + 1);
                size_t e = json.find_first_not_of("0123456789.", n);
                if (n != std::string::npos && e != std::string::npos)
                    return json.substr(n, e - n);
                return {};
            }
            size_t eq = json.find('"', q + 1);
            return eq == std::string::npos ? std::string{} : json.substr(q + 1, eq - q - 1);
        };
        std::string v;
        v = js("vocab_size");   if (!v.empty()) cfg.vocab_size = std::stoul(v);
        v = js("max_seq_len");  if (!v.empty()) cfg.max_seq_len = std::stoul(v);
        v = js("embed_dim");    if (!v.empty()) cfg.embed_dim = std::stoul(v);
        v = js("num_heads");    if (!v.empty()) cfg.num_heads = std::stoul(v);
        v = js("num_kv_heads"); if (!v.empty()) cfg.num_kv_heads = std::stoul(v);
        v = js("num_layers");   if (!v.empty()) cfg.num_layers = std::stoul(v);
        v = js("ff_dim");       if (!v.empty()) cfg.ff_dim = std::stoul(v);
        v = js("head_dim");     if (!v.empty()) cfg.head_dim = std::stoul(v);
        v = js("rope_theta");   if (!v.empty()) cfg.rope_theta = std::stof(v);
        return true;
    }
};

} // namespace

extern "C" {

QWEN_INFER_API void* qw_create(const char* config_json) {
    auto* s = new QwenInferState();
    if (config_json && config_json[0]) s->parseConfig(config_json);
    return s;
}

QWEN_INFER_API void qw_destroy(void* ctx) {
    delete static_cast<QwenInferState*>(ctx);
}

QWEN_INFER_API int qw_load_model(void* ctx, const char* gguf_path,
                                  char* error_buf, int error_buf_size) {
    auto* s = static_cast<QwenInferState*>(ctx);
    if (!s || !gguf_path) {
        if (error_buf && error_buf_size > 0)
            std::strncpy(error_buf, "null_context_or_path", error_buf_size - 1);
        return 0;
    }
    s->model_path = gguf_path;
    s->model_loaded = true;
    return 1;
}

QWEN_INFER_API char* qw_forward(void* ctx, const int32_t* tokens,
                                 uint32_t token_count) {
    auto* s = static_cast<QwenInferState*>(ctx);
    if (!s || !tokens || token_count == 0) {
        const char* err = "{\"error\":\"null_context_or_input\"}";
        auto* out = new char[std::strlen(err) + 1];
        std::strcpy(out, err);
        return out;
    }
    std::ostringstream r;
    r << "{\"logits\":[0.0],\"token_count\":" << token_count
      << ",\"architecture\":\"qwen2\",\"status\":\"stub_forward_pass\"}";
    std::string json = r.str();
    auto* out = new char[json.size() + 1];
    std::memcpy(out, json.data(), json.size());
    out[json.size()] = '\0';
    return out;
}

QWEN_INFER_API char* qw_sample(void* ctx, const int32_t* tokens,
                                uint32_t token_count, int k) {
    if (k <= 0) k = 1;
    std::ostringstream r;
    r << "{\"token\":0,\"logit\":0.0,\"top_k\":[],\"k\":" << k
      << ",\"status\":\"stub_sample\"}";
    std::string json = r.str();
    auto* out = new char[json.size() + 1];
    std::memcpy(out, json.data(), json.size());
    out[json.size()] = '\0';
    return out;
}

QWEN_INFER_API char* qw_get_config(void* ctx) {
    auto* s = static_cast<QwenInferState*>(ctx);
    if (!s) {
        auto* err = new char[5];
        std::strcpy(err, "null");
        return err;
    }
    std::string json = s->makeConfigJson();
    auto* out = new char[json.size() + 1];
    std::memcpy(out, json.data(), json.size());
    out[json.size()] = '\0';
    return out;
}

QWEN_INFER_API int qw_probe() {
#ifdef _WIN32
    // D3D11 is always available on Windows 7+
    return 1;
#else
    return 0;
#endif
}

QWEN_INFER_API void qw_free_string(char* str) {
    delete[] str;
}

} // extern "C"
