#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>

// Source provenance metadata
struct SourceMeta {
    std::string url;
    std::string title;
    float trust_score = 0.5f; // 0..1
};

// Generic memory entry (persistent store)
struct MemoryEntry {
    std::string id;          // stable id
    std::string text;        // stored content or short summary
    std::vector<float> embedding; // vector embedding

    // retention and usage
    float importance = 1.0f;    // base importance
    uint64_t access_count = 0;  // number of times used

    // Trust & provenance
    float trust_score = 0.5f;   // aggregated trust from sources (0..1)
    float source_score = 0.5f;  // score derived from source ranking
    std::vector<SourceMeta> sources;

    // tags for agent-routing and quick filtering
    std::vector<std::string> tags;

    // timestamps (epoch ms)
    uint64_t created_at = 0;
    uint64_t last_accessed = 0;
};

// KV-style memory entry used in fast cache / retrieval
struct KVMemoryEntry {
    std::string key; // prompt or canonical key
    std::vector<float> embedding;

    float trust_score = 0.5f; // 0..1
    double value = 0.0;       // economic value (saved cost)
    float recency = 0.0f;     // normalized recency (0..1)
    std::vector<std::string> tags;
};

// Utility: compute aggregated trust from sources (simple avg, weighted later)
inline float compute_trust_from_sources(const std::vector<SourceMeta>& sources){
    if(sources.empty()) return 0.5f;
    float s = 0.0f;
    for(auto &src : sources) s += src.trust_score;
    return s / static_cast<float>(sources.size());
}

// Utility: compute economic saved cost of a memory entry
inline double compute_saved_cost(const MemoryEntry &m, double avg_inference_cost_saved){
    return static_cast<double>(m.access_count) * avg_inference_cost_saved;
}

// Retrieval scoring combining semantic similarity, trust, and recency
inline float retrieval_score(float similarity, float trust_score, float recency){
    // score = 0.5 * similarity + 0.3 * trust + 0.2 * recency
    return 0.5f * similarity + 0.3f * trust_score + 0.2f * recency;
}

// Minimal helper: update memory trust when ingesting sources
inline void apply_trust_on_ingest(MemoryEntry &m){
    float t = compute_trust_from_sources(m.sources);
    m.trust_score = t;
    m.source_score = t; // simple mapping for now
    m.importance = m.importance * m.trust_score;
}
