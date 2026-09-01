#include "../include/kv_cache.h"
#include "../include/compiler.h"
#include "../include/planner.h"
#include "../include/quant.h"
#include "../include/scheduler.h"
#include "../include/knobs.h"
#include "../include/runtime_loop.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <thread>

// Global scheduler instance for demo
static CreditScheduler g_scheduler;

// Define globals from knobs.h
Knobs g_knobs;
Metrics g_metrics;

struct RuntimeState {
    KVCache kv;
    uint32_t vocab_size = 50257;
    uint32_t max_tokens = 128;
    std::vector<float> logits;
    std::vector<int> tokens;
    uint32_t cur_token = 0;
    // kv layout params for the fallback runtime
    uint32_t kv_layers = 0;
    uint32_t kv_heads = 0;
    uint32_t kv_D = 0;
    RuntimeState():logits(vocab_size, 0.0f){}
};

namespace {
struct FallbackState {
    int token = 0;
    uint32_t layer = 0;
    std::vector<float> hidden;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> batch_biases;
    float attention_bias = 0.0f;
    float ffn_bias = 0.0f;
    float batch_bias = 0.0f;

    FallbackState() : hidden(64, 0.0f), q(64, 0.0f), k(64, 0.0f), v(64, 0.0f) {}
};

static thread_local FallbackState g_fallback;

static float stable_unit_float(int token, uint32_t layer, uint32_t index, uint32_t salt){
    uint32_t x = static_cast<uint32_t>(token) * 2654435761u;
    x ^= (layer + 1u) * 2246822519u;
    x ^= (index + 1u) * 3266489917u;
    x ^= (salt + 1u) * 374761393u;
    x ^= x >> 15;
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    x ^= x >> 16;
    return static_cast<float>(x & 0x00FFFFFFu) / 16777215.0f;
}

static float mean_vector(const std::vector<float>& values){
    if (values.empty()) return 0.0f;
    float total = 0.0f;
    for (float v : values) total += v;
    return total / static_cast<float>(values.size());
}

static void refresh_hidden_from_token(){
    for (size_t i = 0; i < g_fallback.hidden.size(); ++i){
        float base = stable_unit_float(g_fallback.token, 0, static_cast<uint32_t>(i), 17u);
        g_fallback.hidden[i] = std::tanh((base * 2.0f - 1.0f) + static_cast<float>((g_fallback.token % 11)) * 0.01f);
    }
}
} // namespace

int sample_top_k(std::vector<float>& logits, int k, float temperature){
    int V = (int)logits.size();
    if (V == 0) return 0;
    k = std::max(1, std::min(k, V));
    if (temperature <= 0.0f) temperature = 1.0f;
    std::vector<int> idx(V);
    for (int i=0;i<V;i++) idx[i]=i;
    std::partial_sort(idx.begin(), idx.begin()+k, idx.end(), [&](int a,int b){ return logits[a]>logits[b]; });
    std::vector<float> probs(k);
    float sum=0.0f;
    for (int i=0;i<k;i++){ float v = logits[idx[i]]/temperature; probs[i]=expf(v); sum+=probs[i]; }
    for (int i=0;i<k;i++) probs[i]/=sum;
    static std::mt19937 rng(1337u);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int pick = dist(rng);
    return idx[pick];
}

// CPU fallback for the GPU ops used by the demo runtime.
void runEmbedding(int token){
    g_fallback.token = token;
    g_fallback.layer = 0;
    refresh_hidden_from_token();
    g_fallback.attention_bias = 0.0f;
    g_fallback.ffn_bias = 0.0f;
    g_fallback.batch_bias = 0.0f;
}

void runQKV(int layer){
    g_fallback.layer = static_cast<uint32_t>(layer);
    for (size_t i = 0; i < g_fallback.hidden.size(); ++i){
        float h = g_fallback.hidden[i];
        float phase = stable_unit_float(g_fallback.token, g_fallback.layer, static_cast<uint32_t>(i), 29u);
        g_fallback.q[i] = std::tanh(h * 0.75f + phase * 0.25f + static_cast<float>(layer) * 0.01f);
        g_fallback.k[i] = std::tanh(h * 0.55f + phase * 0.45f);
        g_fallback.v[i] = std::tanh(h * 0.35f + phase * 0.65f);
    }
}

void runAttention(int layer, KVCache& kv){
    uint32_t seq = kv.current_seq;
    uint32_t inspected = (seq < kv.max_seq) ? seq + 1 : kv.max_seq;
    float total = 0.0f;
    uint32_t count = 0;
    for (uint32_t head = 0; head < kv.heads; ++head){
        for (uint32_t pos = 0; pos < inspected; ++pos){
            uint32_t base = kv.offset(static_cast<uint32_t>(layer), head, pos);
            for (uint32_t d = 0; d < kv.D; ++d){
                total += kv.K[base + d] * 0.5f + kv.V[base + d] * 0.5f;
                ++count;
            }
        }
    }
    g_fallback.attention_bias = count ? (total / static_cast<float>(count)) : 0.0f;
}

void runFFN(int layer){
    float total = 0.0f;
    for (size_t i = 0; i < g_fallback.hidden.size(); ++i){
        float mixed = std::tanh(
            g_fallback.q[i] * 0.7f +
            g_fallback.k[i] * 0.2f +
            g_fallback.v[i] * 0.1f +
            g_fallback.attention_bias * 0.05f +
            static_cast<float>(layer) * 0.005f
        );
        g_fallback.hidden[i] = mixed;
        total += mixed;
    }
    g_fallback.ffn_bias = total / static_cast<float>(g_fallback.hidden.size());
}

void runLogits(std::vector<float>& out){
    if (out.empty()){
        out.assign(50257, 0.0f);
    }
    int vocab = static_cast<int>(out.size());
    int favored = (g_fallback.token + static_cast<int>(g_fallback.layer) * 31 + static_cast<int>(std::lround(g_fallback.attention_bias * 10.0f))) % vocab;
    if (favored < 0) favored += vocab;
    for (int i = 0; i < vocab; ++i){
        float periodic = stable_unit_float(g_fallback.token, g_fallback.layer, static_cast<uint32_t>(i), 91u);
        out[i] = periodic * 0.25f + g_fallback.ffn_bias * 0.15f + g_fallback.attention_bias * 0.1f;
        out[i] -= std::abs(i - favored) * 0.0001f;
    }
    out[favored] += 6.0f;
}

void prefill(RuntimeState &rt, const std::vector<int> &prompt){
    for (int t=0;t<prompt.size();++t){
        int token = prompt[t];
        runEmbedding(token);
        for (uint32_t l=0;l<rt.kv.layers;++l){
            runQKV(l);
            // read k/v from the current position and append
            // here we simulate with zeros
            std::vector<float> k(rt.kv.D, 0.0f), v(rt.kv.D, 0.0f);
            for (uint32_t h=0;h<rt.kv.heads;++h) rt.kv.append(l,h,k.data(),v.data());
            runAttention(l, rt.kv);
            runFFN(l);
        }
        rt.kv.current_seq++;
    }
}

void decode_loop(RuntimeState &rt, int start_token){
    int token = start_token;
    for (uint32_t step = 0; step < rt.max_tokens; ++step){
        runEmbedding(token);
        for (uint32_t l=0; l<rt.kv.layers; ++l){
            runQKV(l);
            std::vector<float> k(rt.kv.D, 0.0f), v(rt.kv.D, 0.0f);
            for (uint32_t h=0; h<rt.kv.heads; ++h) rt.kv.append(l,h,k.data(),v.data());
            runAttention(l, rt.kv);
            runFFN(l);
        }
        runLogits(rt.logits);
        token = sample_top_k(rt.logits, 40, 0.8f);
        rt.tokens.push_back(token);
        // emit
        std::cout<<token<<" "<<std::flush;
        rt.kv.current_seq++;
    }
}

// --- Speculative batched verifier and adaptive scheduler ---

static void softmax_row(const float* logits, float* probs, int vocab){
    float maxv = logits[0];
    for (int i=1;i<vocab;i++) if (logits[i] > maxv) maxv = logits[i];
    double sum = 0.0;
    for (int i=0;i<vocab;i++){ double e = exp((double)logits[i] - maxv); probs[i] = (float)e; sum += e; }
    for (int i=0;i<vocab;i++) probs[i] = (float)(probs[i] / (float)sum);
}

// deterministic logits batch generation
static void runLogitsBatch(RuntimeState& rt, int M, std::vector<float>& out_logits){
    out_logits.assign((size_t)M * rt.vocab_size, 0.0f);
    int base = rt.tokens.empty() ? 0 : rt.tokens.back();
    float bias = g_fallback.batch_bias;
    for (int i=0;i<M;i++){
        int favored = (base + 1 + i + static_cast<int>(std::lround(bias * 3.0f))) % rt.vocab_size;
        if (favored < 0) favored += rt.vocab_size;
        float row_bias = (i < static_cast<int>(g_fallback.batch_biases.size())) ? g_fallback.batch_biases[i] : 0.0f;
        for (uint32_t v = 0; v < rt.vocab_size; ++v){
            float periodic = stable_unit_float(base + i, static_cast<uint32_t>(i), v, 123u);
            out_logits[(size_t)i * rt.vocab_size + v] = periodic * 0.2f + row_bias * 0.1f + bias * 0.05f;
        }
        out_logits[(size_t)i * rt.vocab_size + favored] += 5.0f;
    }
}

struct VerifyResult { int accepted; int next_token; };

static VerifyResult verifyPrefixProb(RuntimeState& rt, const std::vector<int>& draft, const std::vector<float>& logitsMxV, float threshold){
    VerifyResult out{}; out.accepted=0; out.next_token=-1;
    std::vector<float> probs(rt.vocab_size);
    for (uint32_t i=0;i<draft.size(); ++i){
        const float* row = &logitsMxV[(size_t)i * rt.vocab_size];
        softmax_row(row, probs.data(), rt.vocab_size);
        int tok = draft[i];
        if (probs[tok] > threshold){ out.accepted++; }
        else {
        int best = 0; for (uint32_t v = 1; v < rt.vocab_size; ++v) if (row[v] > row[best]) best = static_cast<int>(v);
            out.next_token = best; return out;
        }
    }
    out.next_token = draft.back(); return out;
}

// TempKV builder (simple deterministic pseudo-KV)
static void buildQKVBatchAndAppend(RuntimeState& rt, const std::vector<int>& draft, TempKV& tmp){
    uint32_t layers = rt.kv.layers; uint32_t heads = rt.kv.heads; uint32_t D = rt.kv.D;
    tmp.init(layers, heads, (uint32_t)draft.size(), D);
    for (uint32_t l=0;l<layers;++l){ for (uint32_t h=0; h<heads; ++h){ for (uint32_t p=0;p<draft.size(); ++p){
        std::vector<float> kv(D);
        for (uint32_t d=0; d<D; ++d) kv[d] = (float)(((l+1)*(h+1)*(p+1)*(d+1)) % 97) / 97.0f;
        tmp.append(l,h,p, kv.data(), kv.data());
    }}}
}

// CPU fallback for the fused attention batch path.
static void runAttentionBatch(RuntimeState& rt, const TempKV& tmp, int M){
    g_fallback.batch_biases.assign((size_t)M, 0.0f);
    g_fallback.batch_bias = 0.0f;
    for (int i = 0; i < M; ++i){
        float total = 0.0f;
        uint32_t count = 0;
        for (uint32_t l = 0; l < tmp.layers; ++l){
            for (uint32_t h = 0; h < tmp.heads; ++h){
                uint32_t base = tmp.offset(l, h, static_cast<uint32_t>(i));
                for (uint32_t d = 0; d < tmp.D; ++d){
                    total += tmp.K[base + d] * 0.5f + tmp.V[base + d] * 0.5f;
                    ++count;
                }
            }
        }
        float row_bias = count ? (total / static_cast<float>(count)) : 0.0f;
        g_fallback.batch_biases[(size_t)i] = row_bias;
        g_fallback.batch_bias += row_bias;
    }
    if (M > 0){
        g_fallback.batch_bias /= static_cast<float>(M);
    }
}

// simple stats for adaptive M
struct SpecStats { float acceptance_rate=0.0f; int window=32; int accepted_tokens=0; int total_tokens=0; };
static void updateStats(SpecStats& s, int accepted, int total){ s.accepted_tokens += accepted; s.total_tokens += total; if (s.total_tokens >= s.window){ s.acceptance_rate = (float)s.accepted_tokens / (float)s.total_tokens; s.accepted_tokens=0; s.total_tokens=0; } }
static int adjustM(int currentM, const SpecStats& s){ if (s.acceptance_rate > 0.85f) return std::min(currentM+1, 8); if (s.acceptance_rate < 0.50f) return std::max(currentM-1, 1); return currentM; }

// control loop: collect global metrics and update knobs (simple policy)
static void controlLoop(){
    const int period_ms = 200;
    Knobs lastApplied = g_knobs;
    while (true){
        // read metrics snapshot
        float acc = g_metrics.acceptance_rate.load();
        float p95 = g_metrics.p95_ms.load();
        float gpu = g_metrics.gpu_util.load();
        float tps = g_metrics.tokens_per_sec.load();

        // simple policy from spec
        // Latency guardrail
        const float target_p95 = 120.0f;
        if (p95 > target_p95){
            int newM = std::max(1, g_knobs.M.load() - 1);
            g_knobs.M.store(newM);
            g_knobs.max_batch.store(std::max(1, g_knobs.max_batch.load() - 1));
            int cv = std::max(1, g_knobs.credits_verify.load() - 1);
            int ck = std::max(1, g_knobs.credits_kv.load() - 1);
            g_knobs.credits_verify.store(cv);
            g_knobs.credits_kv.store(ck);
            g_knobs.quant_mode.store(Knobs::INT8);
        } else {
            // throughput push
            if (gpu < 0.85f){
                g_knobs.max_batch.store(std::min(32, g_knobs.max_batch.load() + 1));
                g_knobs.M.store(std::min(8, g_knobs.M.load() + 1));
                g_knobs.credits_verify.store(std::min(8, g_knobs.credits_verify.load() + 1));
                g_knobs.credits_kv.store(std::min(8, g_knobs.credits_kv.load() + 1));
            }

            if (acc > 0.85f) g_knobs.M.store(std::min(8, g_knobs.M.load() + 1));
            else if (acc < 0.5f) g_knobs.M.store(std::max(1, g_knobs.M.load() - 1));

            if (g_metrics.link_bw_util.load() > 0.8f) g_knobs.quant_mode.store(Knobs::INT8);
            else if (gpu > 0.9f && p95 < 0.8f * target_p95) g_knobs.quant_mode.store(Knobs::INT4);
        }

        // apply credits to scheduler
        g_scheduler.init_stage("verify", g_knobs.credits_verify.load());
        g_scheduler.init_stage("kv", g_knobs.credits_kv.load());

        // simple log
        std::cout<<"[Control] metrics acc="<<acc<<" p95="<<p95<<" gpu="<<gpu<<" tps="<<tps<<" -> M="<<g_knobs.M.load()<<" batch="<<g_knobs.max_batch.load()<<" q="<<g_knobs.quant_mode.load()<<"\n"<<std::flush;

        std::this_thread::sleep_for(std::chrono::milliseconds(period_ms));
    }
}

void speculativeDecodeBatched(RuntimeState& rt, int start_token){
    int token = start_token; SpecStats stats{}; const float threshold = 0.20f;

    while (rt.tokens.size() < rt.max_tokens){
        int M = g_knobs.M.load();
        // draft prediction (deterministic fallback)
        std::vector<int> draft; draft.reserve(M);
        int cur = token;
        for (int i=0;i<M;i++){ cur = (cur + 1) % rt.vocab_size; draft.push_back(cur); }
        int m = (int)draft.size();

        // build TempKV for draft
        TempKV tmp; buildQKVBatchAndAppend(rt, draft, tmp);

        // Acquire verify credit before running expensive verify
        while (!g_scheduler.try_acquire("verify")){
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // run fused attention batch (CPU fallback)
        runAttentionBatch(rt, tmp, m);

        // run logits for M tokens (deterministic fallback)
        std::vector<float> logitsMxV; runLogitsBatch(rt, m, logitsMxV);

        // verify probabilistic prefix
        auto res = verifyPrefixProb(rt, draft, logitsMxV, threshold);

        // If accepted prefix exists, compress TempKV (simulate INT8 transfer), decompress and commit
        if (res.accepted > 0){
            // simulate compression: quantize entire tmp.K and tmp.V
            std::vector<int8_t> Kq, Vq;
            float scaleK=1.0f, scaleV=1.0f;
            quantize_int8_vec(tmp.K, Kq, scaleK);
            quantize_int8_vec(tmp.V, Vq, scaleV);

            // simulate transfer latency (tiny sleep)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

            // decompress into recovered TempKV
            std::vector<float> recK, recV;
            dequantize_int8_vec(Kq, recK, scaleK);
            dequantize_int8_vec(Vq, recV, scaleV);

            TempKV recovered; recovered.init(tmp.layers, tmp.heads, tmp.max_len, tmp.D);
            recovered.K = std::move(recK);
            recovered.V = std::move(recV);

            // Acquire kv credit before commit
            while (!g_scheduler.try_acquire("kv")){
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            // commit accepted prefix from recovered into main KV
            commit_prefix_from_temp(rt.kv, recovered, (uint32_t)res.accepted, rt.kv.current_seq);

            // release kv credit
            g_scheduler.release("kv");

            for (int i=0;i<res.accepted;++i){ rt.tokens.push_back(draft[i]); std::cout<<draft[i]<<" "<<std::flush; }
            rt.kv.current_seq += res.accepted; token = rt.tokens.back();
        }

        // Fallback for mismatch: commit single-step (use first slot)
        if (res.accepted < m){
            int fb = res.next_token;

            // compress/decompress single slot to simulate transfer and then commit
            // Build small temp of slot 0
            TempKV single; single.init(tmp.layers, tmp.heads, 1, tmp.D);
            for (uint32_t l=0;l<tmp.layers;++l){ for (uint32_t h=0; h<tmp.heads;++h){
                uint32_t off = tmp.offset(l,h,0);
                single.append(l,h,0, &tmp.K[off], &tmp.V[off]);
            }}

            std::vector<int8_t> Kq1, Vq1; float sK1=1.0f, sV1=1.0f;
            quantize_int8_vec(single.K, Kq1, sK1);
            quantize_int8_vec(single.V, Vq1, sV1);
            std::vector<float> recK1, recV1; dequantize_int8_vec(Kq1, recK1, sK1); dequantize_int8_vec(Vq1, recV1, sV1);
            TempKV recovered1; recovered1.init(single.layers, single.heads, single.max_len, single.D);
            recovered1.K = std::move(recK1); recovered1.V = std::move(recV1);

            while (!g_scheduler.try_acquire("kv")) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            commit_prefix_from_temp(rt.kv, recovered1, 1, rt.kv.current_seq);
            g_scheduler.release("kv");

            rt.tokens.push_back(fb); std::cout<<fb<<" "<<std::flush; rt.kv.current_seq += 1; token = fb;
        }

        // release verify credit
        g_scheduler.release("verify");

        // update stats and adapt M
        updateStats(stats, res.accepted, m); M = adjustM(M, stats);

        tmp.reset();
    }
    std::cout<<"\nDone speculative\n";
}

int main_debug_stream(int argc, char** argv){
    // initialize scheduler stages
    g_scheduler.init_stage("draft", 4);
    g_scheduler.init_stage("verify", 2);
    g_scheduler.init_stage("kv", 2);

    RuntimeState rt; rt.kv.init(6, 8, 512, 64); rt.max_tokens = 16; rt.kv_layers = 6; rt.kv_heads = 8; rt.kv_D = 64;
    std::vector<int> prompt = {1,2,3}; prefill(rt, prompt);

    // start control loop in background
    std::thread ctl(controlLoop);
    ctl.detach();

    speculativeDecodeBatched(rt, 0);
    return 0;
}
