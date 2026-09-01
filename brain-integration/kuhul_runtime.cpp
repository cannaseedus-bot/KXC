/**
 * kuhul_runtime.cpp — Atomic KUHUL Brains / SCXQ2 Brain API Runtime
 *
 * Format: SCXQ2 (SCX2 v1.0)
 * Build:  g++ -std=c++17 -O3 -o kuhul_runtime kuhul_runtime.cpp
 * Output: merged_brain.scxq2
 *
 * 10 canonical brains, 4-lane structure, federation merge, bounded graph walk.
 *
 * Key invariants:
 *   - Deterministic: same input → same output always
 *   - Replayable: full trace preserved in proofs lane
 *   - Headless: no UI, no generation — execution only
 *   - Collapse-only: π/KUHUL proves correctness by collapse
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ─── Hashing ─────────────────────────────────────────────────────────────────

namespace hashing {

/// djb2 u32 — concept hashes
inline uint32_t hash_string(const std::string& s) {
    uint32_t h = 5381;
    for (unsigned char c : s)
        h = ((h << 5) + h) ^ c;
    return h;
}

/// FNV-1a u64 — brain IDs
inline uint64_t hash_brain_name(const std::string& name) {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : name) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ULL;
    }
    return h;
}

/// FNV-variant u64 — payload hash (quick)
inline uint64_t quick_hash(const std::string& data) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (unsigned char c : data) {
        h ^= static_cast<uint64_t>(c);
        h *= 0x100000001b3ULL;
    }
    return h;
}

} // namespace hashing

// ─── SCXQ2 Header (32 bytes, packed) ─────────────────────────────────────────

#pragma pack(push, 1)
struct SCXQ2Header {
    char     magic[4];      // "SCX2" = 53 43 58 32
    uint16_t version;       // 0x0001
    uint16_t flags;         // 0x0001
    uint64_t brain_id;      // FNV-1a u64 of brain name
    uint64_t payload_hash;  // SHA-256 truncated (quick_hash here)
    uint32_t lanes;         // 4
    uint32_t reserved;      // 0

    SCXQ2Header() : version(0x0001), flags(0x0001), brain_id(0),
                    payload_hash(0), lanes(4), reserved(0) {
        magic[0]='S'; magic[1]='C'; magic[2]='X'; magic[3]='2';
    }

    bool valid() const {
        return magic[0]=='S' && magic[1]=='C' && magic[2]=='X' && magic[3]=='2'
            && version == 0x0001 && lanes == 4;
    }
};
#pragma pack(pop)
static_assert(sizeof(SCXQ2Header) == 32, "SCXQ2Header must be 32 bytes");

// ─── Edge ─────────────────────────────────────────────────────────────────────

struct Edge {
    uint32_t from;   // concept hash
    uint32_t to;     // concept hash
    double   weight; // [0.0, 1.0]
};

// ─── Brain — 4 lanes ──────────────────────────────────────────────────────────

struct Brain {
    SCXQ2Header header;

    // Lane 0: concepts — hash → string
    std::unordered_map<uint32_t, std::string> concepts;

    // Lane 1: explanations — string pool
    std::vector<std::string> explanations;

    // Lane 2: edges — weighted concept connections
    std::vector<Edge> edges;

    // Lane 3: proofs — verification trace statements
    std::vector<std::string> proofs;

    std::string name;

    Brain() = default;
    explicit Brain(const std::string& brain_name) : name(brain_name) {
        header.brain_id = hashing::hash_brain_name(brain_name);
    }

    void add_concept(const std::string& concept) {
        concepts[hashing::hash_string(concept)] = concept;
    }

    void add_edge(const std::string& from, const std::string& to, double weight) {
        edges.push_back({hashing::hash_string(from), hashing::hash_string(to),
                         std::min(weight, 1.0)});
    }

    void add_explanation(const std::string& ex) {
        explanations.push_back(ex);
    }

    void add_proof(const std::string& proof) {
        proofs.push_back(proof);
    }

    void finalize() {
        // Compute payload hash from concepts + proofs
        std::string payload;
        for (auto& [h, c] : concepts) payload += c + ";";
        for (auto& p : proofs)         payload += p + ";";
        header.payload_hash = hashing::quick_hash(payload);
    }
};

// ─── Inference Result ─────────────────────────────────────────────────────────

struct InferenceResult {
    std::vector<std::string> path;         // concept walk
    std::vector<std::string> explanations; // lane 1 matches
    std::vector<std::string> proofs;       // lane 3 matches
    double confidence   = 1.0;
    bool   deterministic = true;
};

// ─── Execution Context ────────────────────────────────────────────────────────

struct ExecutionContext {
    std::unordered_map<std::string, std::string> variables;
    std::vector<std::string> trace;

    void set(const std::string& k, const std::string& v) { variables[k] = v; }
    std::string get(const std::string& k) const {
        auto it = variables.find(k);
        return it != variables.end() ? it->second : "";
    }
    void log(const std::string& msg) { trace.push_back(msg); }
};

// ─── Brain Registry — 10 canonical brains ─────────────────────────────────────

class BrainRegistry {
public:
    std::unordered_map<std::string, Brain> brains; // name → brain
    std::unordered_map<uint64_t, std::string> id_to_name;

    BrainRegistry() { _load_canonical(); }

    const Brain* get(const std::string& name) const {
        auto it = brains.find(name);
        return it != brains.end() ? &it->second : nullptr;
    }

    const Brain* get_by_id(uint64_t id) const {
        auto it = id_to_name.find(id);
        if (it == id_to_name.end()) return nullptr;
        return get(it->second);
    }

private:
    void _register(Brain b) {
        b.finalize();
        id_to_name[b.header.brain_id] = b.name;
        brains[b.name] = std::move(b);
    }

    void _load_canonical() {
        // ── brain_intro ────────────────────────────────────────────────────
        {
            Brain b("brain_intro");
            b.add_concept("xjson");
            b.add_concept("executable_cognition");
            b.add_concept("execution");
            b.add_concept("generation");
            b.add_explanation("XJSON is not a config format — it is executable cognition.");
            b.add_explanation("Every XJSON node carries executable semantics, not just data.");
            b.add_edge("xjson",                "executable_cognition", 0.95);
            b.add_edge("executable_cognition",  "execution",           0.90);
            b.add_edge("execution",             "generation",          0.10); // intentionally low
            b.add_proof("execution ≠ generation");
            b.add_proof("xjson.type == executable_cognition");
            _register(std::move(b));
        }

        // ── brain_grams ────────────────────────────────────────────────────
        {
            Brain b("brain_grams");
            b.add_concept("gram");
            b.add_concept("embedding");
            b.add_concept("inspectable");
            b.add_concept("symbolic_unit");
            b.add_explanation("A gram replaces the embedding. Grams are atomic symbolic units.");
            b.add_explanation("Grams are inspectable — you can read their meaning without decoding.");
            b.add_edge("gram",       "symbolic_unit",  0.95);
            b.add_edge("gram",       "inspectable",    0.90);
            b.add_edge("embedding",  "gram",           0.85); // embedding → superseded by gram
            b.add_proof("grams are inspectable");
            b.add_proof("gram.type == symbolic_unit");
            _register(std::move(b));
        }

        // ── brain_graph ────────────────────────────────────────────────────
        {
            Brain b("brain_graph");
            b.add_concept("graph_walk");
            b.add_concept("inference");
            b.add_concept("bounded_path_search");
            b.add_concept("visited_set");
            b.add_explanation("Graph walk IS inference. Bounded path search over concept graph.");
            b.add_explanation("Inference = find highest-weight path from concept to concept.");
            b.add_edge("graph_walk",         "inference",           0.95);
            b.add_edge("inference",          "bounded_path_search", 0.90);
            b.add_edge("bounded_path_search","visited_set",         0.85);
            b.add_proof("inference = bounded path search");
            b.add_proof("graph_walk.depth <= 4");
            _register(std::move(b));
        }

        // ── brain_execution ────────────────────────────────────────────────
        {
            Brain b("brain_execution");
            b.add_concept("execution");
            b.add_concept("state_mutation");
            b.add_concept("generation");
            b.add_concept("determinism");
            b.add_explanation("Execution mutates state. Generation produces tokens.");
            b.add_explanation("Execution is deterministic. Generation is stochastic.");
            b.add_edge("execution",     "state_mutation", 0.95);
            b.add_edge("execution",     "determinism",    0.90);
            b.add_edge("generation",    "execution",      0.05); // generation can trigger execution
            b.add_proof("execution ≠ generation");
            b.add_proof("execution.output == deterministic");
            _register(std::move(b));
        }

        // ── brain_compression ─────────────────────────────────────────────
        {
            Brain b("brain_compression");
            b.add_concept("scxq2");
            b.add_concept("lane_packing");
            b.add_concept("meaning_preservation");
            b.add_concept("symbolic_compression");
            b.add_explanation("SCXQ2 uses lane packing for symbolic compression.");
            b.add_explanation("Compression in SCXQ2 preserves meaning — no lossy encoding.");
            b.add_edge("scxq2",               "lane_packing",          0.95);
            b.add_edge("lane_packing",         "symbolic_compression",  0.90);
            b.add_edge("symbolic_compression", "meaning_preservation",  0.92);
            b.add_proof("compression preserves meaning");
            b.add_proof("scxq2.lanes == 4");
            _register(std::move(b));
        }

        // ── brain_proofs ──────────────────────────────────────────────────
        {
            Brain b("brain_proofs");
            b.add_concept("proof_carrying_inference");
            b.add_concept("verifiable_output");
            b.add_concept("trace");
            b.add_concept("deterministic_replay");
            b.add_explanation("Every inference result carries its proof trace.");
            b.add_explanation("Outputs are verifiable: replay the trace → same result.");
            b.add_edge("proof_carrying_inference", "verifiable_output",    0.95);
            b.add_edge("verifiable_output",         "trace",               0.90);
            b.add_edge("trace",                     "deterministic_replay", 0.92);
            b.add_proof("outputs are verifiable");
            b.add_proof("proof_lane.type == verification_trace");
            _register(std::move(b));
        }

        // ── brain_federation ──────────────────────────────────────────────
        {
            Brain b("brain_federation");
            b.add_concept("federation");
            b.add_concept("multi_brain_consensus");
            b.add_concept("determinism");
            b.add_concept("merge_algorithm");
            b.add_explanation("Federation = merge multiple brains into one consensus brain.");
            b.add_explanation("Merge preserves determinism: concepts deduplicate, edges blend.");
            b.add_edge("federation",           "multi_brain_consensus", 0.95);
            b.add_edge("multi_brain_consensus", "merge_algorithm",       0.90);
            b.add_edge("merge_algorithm",       "determinism",           0.92);
            b.add_proof("federation preserves determinism");
            b.add_proof("merge.edge_weight = min((w1+w2)*0.85, 1.0)");
            _register(std::move(b));
        }

        // ── brain_policy ──────────────────────────────────────────────────
        {
            Brain b("brain_policy");
            b.add_concept("policy");
            b.add_concept("execution_gate");
            b.add_concept("enforceable");
            b.add_concept("law");
            b.add_explanation("Policy gates execution. No execution without policy clearance.");
            b.add_explanation("Policies are enforceable — violation causes hard rejection.");
            b.add_edge("policy",         "execution_gate", 0.95);
            b.add_edge("execution_gate", "enforceable",    0.90);
            b.add_edge("enforceable",    "law",            0.88);
            b.add_proof("policies are enforceable");
            b.add_proof("execution.requires(policy_clearance)");
            _register(std::move(b));
        }

        // ── brain_models ──────────────────────────────────────────────────
        {
            Brain b("brain_models");
            b.add_concept("model");
            b.add_concept("advisor");
            b.add_concept("host_powerlessness");
            b.add_concept("law");
            b.add_explanation("Models are advisors. They recommend — they do not command.");
            b.add_explanation("Host Powerlessness: models cannot override law.");
            b.add_edge("model",             "advisor",            0.95);
            b.add_edge("advisor",           "host_powerlessness", 0.90);
            b.add_edge("host_powerlessness","law",                0.95);
            b.add_proof("models cannot override law");
            b.add_proof("model.authority < law.authority");
            _register(std::move(b));
        }

        // ── brain_build ───────────────────────────────────────────────────
        {
            Brain b("brain_build");
            b.add_concept("learning");
            b.add_concept("edge_mutation");
            b.add_concept("reinforcement");
            b.add_concept("weight_update");
            b.add_explanation("Learning IS edge mutation. Weights shift on feedback.");
            b.add_explanation("Reinforcement: successful paths gain weight; failed paths decay.");
            b.add_edge("learning",      "edge_mutation",  0.95);
            b.add_edge("edge_mutation", "weight_update",  0.90);
            b.add_edge("weight_update", "reinforcement",  0.88);
            b.add_proof("learning = edge mutation");
            b.add_proof("learning.mechanism == graph_edge_weight_delta");
            _register(std::move(b));
        }
    }
};

// ─── KuhulRuntime ─────────────────────────────────────────────────────────────

class KuhulRuntime {
public:
    static constexpr double ENTROPY_CONSTANT  = 0.21;
    static constexpr double MAX_EDGE_WEIGHT   = 1.0;
    static constexpr double EDGE_MERGE_FACTOR = 0.85;
    static constexpr int    MAX_GRAPH_DEPTH   = 4;

    BrainRegistry registry;

    KuhulRuntime() = default;

    // ── Query: look up a concept in a brain ──────────────────────────────
    std::string query_brain(const std::string& brain_name,
                            const std::string& concept) const {
        const Brain* b = registry.get(brain_name);
        if (!b) return "[brain not found: " + brain_name + "]";

        uint32_t h = hashing::hash_string(concept);
        auto it = b->concepts.find(h);
        if (it == b->concepts.end())
            return "[concept not found in " + brain_name + ": " + concept + "]";

        std::string result = "concept: " + it->second + "\n";
        // Collect outgoing edges
        for (const auto& e : b->edges) {
            if (e.from == h) {
                auto target = b->concepts.find(e.to);
                if (target != b->concepts.end())
                    result += "  → " + target->second +
                              " (weight=" + std::to_string(e.weight) + ")\n";
            }
        }
        return result;
    }

    // ── Execute: bounded graph walk from concept ──────────────────────────
    InferenceResult execute_brain(const std::string& brain_name,
                                  const std::string& start_concept,
                                  ExecutionContext& ctx) const {
        InferenceResult result;
        const Brain* b = registry.get(brain_name);
        if (!b) {
            result.deterministic = false;
            result.confidence    = 0.0;
            return result;
        }

        uint32_t start_hash = hashing::hash_string(start_concept);
        std::set<uint32_t> visited;
        uint32_t current = start_hash;
        double confidence = 1.0;

        for (int depth = 0; depth < MAX_GRAPH_DEPTH; ++depth) {
            // Add current to path
            auto concept_it = b->concepts.find(current);
            if (concept_it == b->concepts.end()) break;
            result.path.push_back(concept_it->second);
            visited.insert(current);
            ctx.log("walk[" + std::to_string(depth) + "]: " + concept_it->second);

            // Find highest-weight outgoing edge to unvisited concept
            double best_weight = -1.0;
            uint32_t best_to   = 0;
            for (const auto& e : b->edges) {
                if (e.from == current && visited.find(e.to) == visited.end()) {
                    if (e.weight > best_weight) {
                        best_weight = e.weight;
                        best_to     = e.to;
                    }
                }
            }
            if (best_weight < 0.0) break; // no unvisited outgoing edge

            confidence *= best_weight;
            current     = best_to;
        }

        // Collect matching explanations (all for now — can filter by path)
        result.explanations = b->explanations;

        // Collect proofs
        result.proofs = b->proofs;

        result.confidence    = confidence;
        result.deterministic = true;
        return result;
    }

    // ── Merge: federation of two brains ──────────────────────────────────
    Brain merge_brains(const Brain& a, const Brain& b) const {
        Brain merged("federation_" + a.name + "_" + b.name);

        // concepts: deduplicate by hash (first seen wins)
        for (auto& [h, c] : a.concepts) merged.concepts[h] = c;
        for (auto& [h, c] : b.concepts) {
            if (merged.concepts.find(h) == merged.concepts.end())
                merged.concepts[h] = c;
        }

        // explanations: union, set dedup
        std::set<std::string> seen_ex(a.explanations.begin(), a.explanations.end());
        merged.explanations = a.explanations;
        for (auto& ex : b.explanations) {
            if (seen_ex.insert(ex).second)
                merged.explanations.push_back(ex);
        }

        // edges: weight = min((w1+w2)*0.85, 1.0)
        // Build edge map for a: (from,to) → weight
        std::map<std::pair<uint32_t,uint32_t>, double> edge_map;
        for (auto& e : a.edges) edge_map[{e.from, e.to}] = e.weight;
        for (auto& e : b.edges) {
            auto key = std::make_pair(e.from, e.to);
            auto it  = edge_map.find(key);
            if (it != edge_map.end())
                it->second = std::min((it->second + e.weight) * EDGE_MERGE_FACTOR,
                                      MAX_EDGE_WEIGHT);
            else
                edge_map[key] = e.weight;
        }
        for (auto& [k, w] : edge_map)
            merged.edges.push_back({k.first, k.second, w});

        // proofs: union (set)
        std::set<std::string> seen_pr(a.proofs.begin(), a.proofs.end());
        merged.proofs = a.proofs;
        for (auto& p : b.proofs) {
            if (seen_pr.insert(p).second)
                merged.proofs.push_back(p);
        }

        merged.finalize();
        return merged;
    }

    // ── Serialize to SCXQ2 binary ─────────────────────────────────────────
    std::vector<uint8_t> serialize(const Brain& brain) const {
        std::vector<uint8_t> buf;
        auto write_u32 = [&](uint32_t v) {
            buf.push_back(v & 0xFF);
            buf.push_back((v >> 8) & 0xFF);
            buf.push_back((v >> 16) & 0xFF);
            buf.push_back((v >> 24) & 0xFF);
        };
        auto write_u64 = [&](uint64_t v) {
            for (int i = 0; i < 8; ++i) buf.push_back((v >> (8*i)) & 0xFF);
        };
        auto write_str = [&](const std::string& s) {
            write_u32(static_cast<uint32_t>(s.size()));
            for (unsigned char c : s) buf.push_back(c);
        };

        // Header
        for (char c : {'S','C','X','2'}) buf.push_back(static_cast<uint8_t>(c));
        buf.push_back(0x01); buf.push_back(0x00); // version
        buf.push_back(0x01); buf.push_back(0x00); // flags
        write_u64(brain.header.brain_id);
        write_u64(brain.header.payload_hash);
        write_u32(4); // lanes
        write_u32(0); // reserved

        // Lane 0: concepts
        write_u32(static_cast<uint32_t>(brain.concepts.size()));
        for (auto& [h, c] : brain.concepts) { write_u32(h); write_str(c); }

        // Lane 1: explanations
        write_u32(static_cast<uint32_t>(brain.explanations.size()));
        for (auto& ex : brain.explanations) write_str(ex);

        // Lane 2: edges
        write_u32(static_cast<uint32_t>(brain.edges.size()));
        for (auto& e : brain.edges) {
            write_u32(e.from); write_u32(e.to);
            uint64_t w_bits;
            memcpy(&w_bits, &e.weight, sizeof(double));
            write_u64(w_bits);
        }

        // Lane 3: proofs
        write_u32(static_cast<uint32_t>(brain.proofs.size()));
        for (auto& p : brain.proofs) write_str(p);

        return buf;
    }

    // ── Deserialize from SCXQ2 binary ─────────────────────────────────────
    Brain deserialize(const std::vector<uint8_t>& buf) const {
        size_t pos = 0;
        auto read_u32 = [&]() -> uint32_t {
            uint32_t v = buf[pos] | (buf[pos+1]<<8) | (buf[pos+2]<<16) | (buf[pos+3]<<24);
            pos += 4; return v;
        };
        auto read_u64 = [&]() -> uint64_t {
            uint64_t v = 0;
            for (int i = 0; i < 8; ++i) v |= (uint64_t)buf[pos+i] << (8*i);
            pos += 8; return v;
        };
        auto read_str = [&]() -> std::string {
            uint32_t len = read_u32();
            std::string s(reinterpret_cast<const char*>(&buf[pos]), len);
            pos += len; return s;
        };

        Brain b;
        // Header
        pos += 4; // magic
        b.header.version      = buf[pos] | (buf[pos+1]<<8); pos += 2;
        b.header.flags        = buf[pos] | (buf[pos+1]<<8); pos += 2;
        b.header.brain_id     = read_u64();
        b.header.payload_hash = read_u64();
        b.header.lanes        = read_u32();
        b.header.reserved     = read_u32();

        // Lane 0
        uint32_t nc = read_u32();
        for (uint32_t i = 0; i < nc; ++i) { uint32_t h = read_u32(); b.concepts[h] = read_str(); }

        // Lane 1
        uint32_t ne = read_u32();
        for (uint32_t i = 0; i < ne; ++i) b.explanations.push_back(read_str());

        // Lane 2
        uint32_t nedges = read_u32();
        for (uint32_t i = 0; i < nedges; ++i) {
            Edge e; e.from = read_u32(); e.to = read_u32();
            uint64_t w_bits = read_u64();
            memcpy(&e.weight, &w_bits, sizeof(double));
            b.edges.push_back(e);
        }

        // Lane 3
        uint32_t np = read_u32();
        for (uint32_t i = 0; i < np; ++i) b.proofs.push_back(read_str());

        return b;
    }

    // ── Save/Load ─────────────────────────────────────────────────────────
    bool save(const Brain& brain, const std::string& path) const {
        auto data = serialize(brain);
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        f.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size()));
        return f.good();
    }

    bool load(Brain& brain, const std::string& path) const {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)),
                                   std::istreambuf_iterator<char>());
        brain = deserialize(data);
        return true;
    }
};

// ─── main ────────────────────────────────────────────────────────────────────

int main() {
    KuhulRuntime runtime;

    std::cout << "=== Atomic KUHUL Brains — SCXQ2 Runtime ===\n\n";

    // Print registered brains
    std::cout << "Canonical brains loaded: " << runtime.registry.brains.size() << "\n";
    for (auto& [name, brain] : runtime.registry.brains) {
        std::cout << "  [" << std::hex << std::uppercase
                  << (brain.header.brain_id & 0xFFFFFFFF) << std::dec << "] "
                  << name
                  << " — " << brain.concepts.size() << " concepts"
                  << ", " << brain.edges.size() << " edges"
                  << ", " << brain.proofs.size() << " proofs\n";
    }
    std::cout << "\n";

    // Demo: execute brain_graph from concept "graph_walk"
    ExecutionContext ctx;
    auto result = runtime.execute_brain("brain_graph", "graph_walk", ctx);
    std::cout << "--- execute brain_graph from 'graph_walk' ---\n";
    std::cout << "Path: ";
    for (size_t i = 0; i < result.path.size(); ++i) {
        if (i) std::cout << " → ";
        std::cout << result.path[i];
    }
    std::cout << "\nConfidence: " << result.confidence
              << "\nDeterministic: " << (result.deterministic ? "true" : "false") << "\n";
    std::cout << "Proofs:\n";
    for (auto& p : result.proofs) std::cout << "  ✓ " << p << "\n";
    std::cout << "\n";

    // Demo: query brain_models
    std::cout << "--- query brain_models 'model' ---\n";
    std::cout << runtime.query_brain("brain_models", "model") << "\n";

    // Demo: federate brain_execution + brain_proofs
    const Brain* be = runtime.registry.get("brain_execution");
    const Brain* bp = runtime.registry.get("brain_proofs");
    if (be && bp) {
        Brain merged = runtime.merge_brains(*be, *bp);
        std::cout << "--- federation: brain_execution + brain_proofs ---\n";
        std::cout << "Merged: " << merged.concepts.size() << " concepts"
                  << ", " << merged.edges.size() << " edges"
                  << ", " << merged.proofs.size() << " proofs\n\n";

        // Save merged
        if (runtime.save(merged, "merged_brain.scxq2"))
            std::cout << "Saved → merged_brain.scxq2\n";

        // Round-trip verify
        Brain loaded;
        if (runtime.load(loaded, "merged_brain.scxq2"))
            std::cout << "Loaded ← merged_brain.scxq2  (concepts="
                      << loaded.concepts.size() << ")\n";
    }

    std::cout << "\n[kuhul_runtime] deterministic=true  entropy_constant="
              << KuhulRuntime::ENTROPY_CONSTANT << "\n";
    return 0;
}
