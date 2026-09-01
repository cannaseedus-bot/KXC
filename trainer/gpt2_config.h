#pragma once
#include <cstdint>
#include <cmath>
#include <string>

struct GPT2Config {
    uint32_t vocab_size = 50282;   // 50257 GPT-2 base + <|endoftext|> + 10 KUHUL fold tokens + 12 TS grammar tokens
    uint32_t n_ctx      = 1024;
    uint32_t n_embd     = 768;
    uint32_t n_head     = 12;
    uint32_t n_layer    = 6;
    uint32_t d_head     = 64;      // n_embd / n_head
    uint32_t d_ff       = 3072;    // 4 * n_embd
    float    attn_scale = 0.125f;  // 1/sqrt(d_head)

    // ── Presets ───────────────────────────────────────────────────────────────
    // distilgpt2  82M  6L  12H  768E  — default, fits HD 4600 full-GPU training
    // gpt2-small  117M 12L 12H  768E  — Phase 1 (CPU fwd/bwd) required
    // gpt2-medium 345M 24L 16H 1024E  — Phase 1 only; target for TS coder
    // gpt2-large  774M 36L 20H 1280E  — CPU training only
    static GPT2Config distilgpt2() { return GPT2Config{}; }

    static GPT2Config gpt2_small() {
        GPT2Config c;
        c.n_layer = 12; c.n_head = 12; c.n_embd = 768;
        c.d_head = 64; c.d_ff = 3072; c.attn_scale = 0.125f;
        return c;
    }

    static GPT2Config gpt2_medium() {
        GPT2Config c;
        c.n_layer = 24; c.n_head = 16; c.n_embd = 1024;
        c.d_head = 64; c.d_ff = 4096; c.attn_scale = 0.125f;
        return c;
    }

    static GPT2Config gpt2_large() {
        GPT2Config c;
        c.n_layer = 36; c.n_head = 20; c.n_embd = 1280;
        c.d_head = 64; c.d_ff = 5120; c.attn_scale = 0.125f;
        return c;
    }

    // Parse --model flag: "distilgpt2" | "gpt2-small" | "gpt2-medium" | "gpt2-large"
    static GPT2Config from_preset(const std::string& name) {
        if (name == "gpt2-medium") return gpt2_medium();
        if (name == "gpt2-small")  return gpt2_small();
        if (name == "gpt2-large")  return gpt2_large();
        return distilgpt2();
    }

    // Human-readable name derived from shape
    std::string preset_name_str() const {
        if (n_layer == 24 && n_embd == 1024) return "gpt2-medium";
        if (n_layer == 36 && n_embd == 1280) return "gpt2-large";
        if (n_layer == 12 && n_embd == 768)  return "gpt2-small";
        return "distilgpt2";
    }

    // Estimated VRAM for weights+grads+Adam in MB (float16 weights, float32 moments)
    uint32_t vram_estimate_mb() const {
        uint64_t params = (uint64_t)vocab_size * n_embd * 2       // wte + wpe
            + (uint64_t)n_layer * (12ull * n_embd * n_embd + 13ull * n_embd);
        // weights f16=2B, grads f32=4B, Adam m+v f32=8B → 14B/param
        return (uint32_t)((params * 14ull) / (1024 * 1024));
    }
};
