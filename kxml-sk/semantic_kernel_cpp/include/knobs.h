#pragma once
#include <atomic>
#include <vector>

// Simple knobs and metrics for self-optimizing control loop
struct Knobs {
    std::atomic<int> M{4};            // speculative window
    std::atomic<int> max_batch{4};
    std::atomic<int> credits_verify{2};
    std::atomic<int> credits_kv{2};
    enum QuantMode { FP16=0, INT8=1, INT4=2 };
    std::atomic<int> quant_mode{INT8};

    // Make copy semantics explicit by loading atomic values
    Knobs() = default;
    Knobs(const Knobs &o){ M.store(o.M.load()); max_batch.store(o.max_batch.load()); credits_verify.store(o.credits_verify.load()); credits_kv.store(o.credits_kv.load()); quant_mode.store(o.quant_mode.load()); }
    Knobs& operator=(const Knobs &o){ if (this!=&o){ M.store(o.M.load()); max_batch.store(o.max_batch.load()); credits_verify.store(o.credits_verify.load()); credits_kv.store(o.credits_kv.load()); quant_mode.store(o.quant_mode.load()); } return *this; }
};

struct Metrics {
    std::atomic<float> acceptance_rate{0.0f};
    std::atomic<float> p95_ms{0.0f};
    std::atomic<float> gpu_util{0.0f};
    std::atomic<float> tokens_per_sec{0.0f};
    std::atomic<float> link_bw_util{0.0f};

    Metrics() = default;
    Metrics(const Metrics &o){ acceptance_rate.store(o.acceptance_rate.load()); p95_ms.store(o.p95_ms.load()); gpu_util.store(o.gpu_util.load()); tokens_per_sec.store(o.tokens_per_sec.load()); link_bw_util.store(o.link_bw_util.load()); }
    Metrics& operator=(const Metrics &o){ if (this!=&o){ acceptance_rate.store(o.acceptance_rate.load()); p95_ms.store(o.p95_ms.load()); gpu_util.store(o.gpu_util.load()); tokens_per_sec.store(o.tokens_per_sec.load()); link_bw_util.store(o.link_bw_util.load()); } return *this; }
};

extern Knobs g_knobs;
extern Metrics g_metrics;

inline float gpu_util_avg(){ return g_metrics.gpu_util.load(); }
