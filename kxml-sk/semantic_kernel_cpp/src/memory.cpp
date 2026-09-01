#include "../include/memory.h"
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <cmath>

class MemoryManager {
public:
    MemoryManager(size_t num_buckets = 128) : buckets(num_buckets) { }

    // store a memory entry and add to KV and IVF bucket
    void ingestFact(const std::string &id, const std::string &text, const std::vector<float> &embedding,
                    const std::vector<SourceMeta> &sources, const std::vector<std::string> &tags){
        std::lock_guard<std::mutex> lk(mu);
        MemoryEntry m;
        m.id = id;
        m.text = text;
        m.embedding = embedding;
        m.sources = sources;
        m.tags = tags;
        uint64_t now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        m.created_at = now;
        m.last_accessed = now;
        apply_trust_on_ingest(m);
        memories.push_back(m);
        size_t idx = memories.size() - 1;
        // create KV entry
        KVMemoryEntry k;
        k.key = id;
        k.embedding = embedding;
        k.trust_score = m.trust_score;
        k.value = compute_saved_cost(m, avg_inference_cost_saved);
        k.recency = 1.0f; // just created
        k.tags = tags;
        kv_map[id] = k;
        // place into bucket
        size_t b = bucket_index(embedding);
        if (buckets.size()) buckets[b].push_back(idx);
    }

    // simple query returning indices of top-k memories matching embedding
    std::vector<size_t> query(const std::vector<float> &embedding, size_t k = 5, const std::vector<std::string>& needed_tags = {}){
        std::lock_guard<std::mutex> lk(mu);
        std::vector<std::pair<float,size_t>> scores;
        // pick candidate buckets
        size_t b = bucket_index(embedding);
        std::vector<size_t> candidates;
        candidates.insert(candidates.end(), buckets[b].begin(), buckets[b].end());
        // also add neighbor buckets (+/-1)
        if (b+1 < buckets.size()) candidates.insert(candidates.end(), buckets[b+1].begin(), buckets[b+1].end());
        if (b>0) candidates.insert(candidates.end(), buckets[b-1].begin(), buckets[b-1].end());

        for (size_t idx : candidates){
            const MemoryEntry &m = memories[idx];
            if (!needed_tags.empty()){
                bool ok=false;
                for (auto &t : needed_tags){
                    for (auto &mt : m.tags){ if (mt==t) { ok=true; break; } }
                    if (ok) break;
                }
                if (!ok) continue;
            }
            float sim = cosine_similarity(embedding, m.embedding);
            float recency = compute_recency(m.last_accessed);
            float score = retrieval_score(sim, m.trust_score, recency);
            scores.emplace_back(score, idx);
        }
        // sort descending
        std::sort(scores.begin(), scores.end(), [](auto &a, auto &b){ return a.first > b.first; });
        std::vector<size_t> out;
        for (size_t i=0;i<scores.size() && out.size()<k;i++) out.push_back(scores[i].second);
        return out;
    }

    // compute recency normalized 0..1 (1 is very recent)
    float compute_recency(uint64_t last_accessed){
        uint64_t now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        const uint64_t max_age = 1000ull * 60 * 60 * 24 * 30; // 30 days in ms
        uint64_t age = (now > last_accessed) ? (now - last_accessed) : 0;
        if (age >= max_age) return 0.0f;
        float r = 1.0f - static_cast<float>(age) / static_cast<float>(max_age);
        return r;
    }

    // simple cosine similarity
    static float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b){
        if (a.empty() || b.empty() || a.size() != b.size()) return 0.0f;
        double dot=0.0, na=0.0, nb=0.0;
        for (size_t i=0;i<a.size();i++){ dot += static_cast<double>(a[i])*b[i]; na += static_cast<double>(a[i])*a[i]; nb += static_cast<double>(b[i])*b[i]; }
        if (na==0.0 || nb==0.0) return 0.0f;
        return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
    }

    // configure avg cost
    void set_avg_cost(double c){ avg_inference_cost_saved = c; }

private:
    size_t bucket_index(const std::vector<float> &emb){
        if (buckets.empty() || emb.empty()) return 0;
        // sum first up to 8 dims
        double s=0.0;
        size_t n = std::min<size_t>(8, emb.size());
        for (size_t i=0;i<n;i++) s += emb[i];
        long long v = static_cast<long long>(std::llround(s * 1000000.0));
        size_t idx = static_cast<size_t>(std::llabs(v)) % buckets.size();
        return idx;
    }

    std::vector<MemoryEntry> memories;
    std::unordered_map<std::string, KVMemoryEntry> kv_map;
    std::vector<std::vector<size_t>> buckets;
    std::mutex mu;
    double avg_inference_cost_saved = 0.001; // default
};

// expose a single global manager for now
static MemoryManager g_memory_mgr(128);

// C-style API wrappers
extern "C" {
    void ingestFact_c(const char* id, const char* text, const float* embedding, size_t emb_len){
        std::vector<float> emb(embedding, embedding+emb_len);
        std::vector<SourceMeta> srcs; std::vector<std::string> tags;
        g_memory_mgr.ingestFact(id, text, emb, srcs, tags);
    }

    // query returns up to k indices; for now we return count of results
    int queryFacts_c(const float* embedding, size_t emb_len, int k){
        std::vector<float> emb(embedding, embedding+emb_len);
        auto res = g_memory_mgr.query(emb, k);
        return static_cast<int>(res.size());
    }
}
