// ============================================================================
// field_execution_engine.h - Complete ASX Runtime Loop (v0.7)
// ============================================================================

#pragma once

#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <functional>
#include "ir.h"
#include "dds_shard_loader.h"
#include "evolution_bot.h"
#include "replay_lane_manager.h"
#include "projection_expert.h"
#include "metric_normalization.h"
#include "geodesic_flow_layer.h"
#include "domain_expert.h"
#include "shared_memory_bridge.h"
#include "legality_verifier.h"
#include "dx12_device_factory.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

inline const char* get_env_value(const char* name) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4996)
#endif
    return std::getenv(name);
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
}

class FieldExecutionEngine {
private:
    std::shared_ptr<DdsShardLoader> m_loader;
    std::shared_ptr<EvolutionBot> m_evolver;
    std::shared_ptr<ReplayLaneManager> m_replay_manager;
    std::shared_ptr<LegalityVerifier> m_verifier;
    std::shared_ptr<ProjectionExpert> m_projection_expert;
    std::shared_ptr<asx::SharedMemoryBridge> m_shm;
    
    // ⟁ DX12 Hardware Context
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12CommandQueue> m_compute_queue;
    ComPtr<ID3D12GraphicsCommandList> m_compute_list;
    ComPtr<ID3D12Fence> m_fence;
    HANDLE m_fence_event;
    uint64_t m_fence_value = 0;
    
    // ⟁ Specialized Experts Registry (MoE)
    std::map<asx::ExpertSpecialization, std::shared_ptr<asx::DomainExpert>> m_experts;
    
    // ⟁ Geometric Layers
    std::shared_ptr<asx::GeodesicFlowLayer> m_flow_layer;
    std::shared_ptr<asx::MetricNormalization> m_norm;
    
    // Runtime State
    uint64_t m_total_ticks = 0;
    std::vector<float> m_current_metric_tensor;
    float m_current_entropy = 0.14f;
    float m_current_attention = 0.72f;
    float m_current_pressure = 0.34f;
    float m_current_gravity = 9.80665f;
    std::vector<float> m_current_arc_weights;
    
public:
    FieldExecutionEngine(const asx::DX12DeviceFactory::DeviceContext& ctx) {
        m_device = ctx.device;
        m_compute_queue = ctx.compute_queue;
        m_compute_list = ctx.compute_list;
        m_fence = ctx.fence;
        m_fence_event = ctx.fence_event;

        m_loader = std::make_shared<DdsShardLoader>(ctx.device.Get(), ctx.copy_queue.Get());
        m_evolver = std::make_shared<EvolutionBot>("Evolver_0");
        m_replay_manager = std::make_shared<ReplayLaneManager>();
        m_verifier = std::make_shared<LegalityVerifier>();
        m_projection_expert = std::make_shared<ProjectionExpert>();
        m_shm = std::make_shared<asx::SharedMemoryBridge>("Local\\KuhulGeometricState");
        
        m_flow_layer = std::make_shared<asx::GeodesicFlowLayer>(1024);
        m_norm = std::make_shared<asx::MetricNormalization>(1024);
        
        // Initialize manifold metric
        m_current_metric_tensor.resize(1024, 0.1f);
        m_current_arc_weights.resize(1024, 1.0f / std::sqrt(1024.0f));
        
        // Register MoE Specialists
        m_experts[asx::ExpertSpecialization::CODING] = std::make_shared<asx::AgentCoder>();
        m_experts[asx::ExpertSpecialization::FACTORY] = std::make_shared<asx::AgentFactory>();

        // Initialize Shared Memory
        if (m_shm) {
            m_shm->initialize_host();
        }
    }
    
    void run_end_to_end_step(int fold_id, const std::string& input_query) {
        m_total_ticks++;
        std::cout << "[INFO] TICK " << m_total_ticks << " | PHASE: PERCEPTION\n";
        
        // 0. Law B: Replay Affinity
        float affinity = m_replay_manager->get_replay_affinity(fold_id);
        std::cout << "[INFO] PHASE: ROUTING | Replay Affinity for Fold " << fold_id << ": " << affinity << "\n";

        // 1. Law C: Residency Management
        if (m_device) {
            m_loader->update_fold_residency(fold_id, true);
        }
        
        // 2. PHASE: COMPUTE (Inference + MoE Routing)
        std::cout << "[INFO] PHASE: COMPUTE | Routing Query: " << input_query << "\n";
        
        asx::ExpertResponse expert_res;
        bool routed_to_specialist = false;
        
        if (input_query.find("code") != std::string::npos || input_query.find("refactor") != std::string::npos) {
            expert_res = m_experts[asx::ExpertSpecialization::CODING]->execute_task("refactor", input_query);
            routed_to_specialist = true;
        } else if (input_query.find("create") != std::string::npos || input_query.find("new") != std::string::npos) {
            expert_res = m_experts[asx::ExpertSpecialization::FACTORY]->execute_task("instantiate", input_query);
            routed_to_specialist = true;
        }
        
        float fitness = 0.85f;
        if (routed_to_specialist) {
            std::cout << "[INFO] MoE: Specialist '" << expert_res.output << "' activated.\n";
            for (auto const& [key, val] : expert_res.manifold_deltas) {
                if (key == "--attention") m_current_attention += val;
                if (key == "--entropy") m_current_entropy += val;
                if (key == "--pressure") m_current_pressure += val;
            }
            fitness = 0.941f; // High fitness for specialized routing
        } else {
            std::cout << "[INFO] MoE: Defaulting to General Reasoning.\n";
            fitness = call_gguf_inference(input_query);
        }

        // ⟁ ARC / Gravity coupling
        const float gravity_gate = std::clamp(
            1.0f + 0.35f * m_current_pressure - 0.25f * m_current_entropy + 0.15f * m_current_attention + 0.10f * affinity,
            0.1f,
            4.0f
        );
        m_current_gravity = 9.80665f * gravity_gate;

        for (size_t i = 0; i < m_current_arc_weights.size(); ++i) {
            const float arc_bias = 1.0f
                + 0.10f * m_current_attention
                - 0.08f * m_current_entropy
                + 0.06f * m_current_pressure
                + 0.04f * affinity;
            m_current_arc_weights[i] = std::clamp(
                (1.0f / std::sqrt(static_cast<float>(m_current_arc_weights.size()))) * arc_bias,
                0.01f,
                2.0f
            );
        }

        if (m_flow_layer) {
            m_flow_layer->set_thermodynamics(m_current_entropy, m_current_pressure);
            m_flow_layer->set_gravity(m_current_gravity);
            m_flow_layer->set_arc_weights(m_current_arc_weights);

            std::vector<float> velocity(m_current_metric_tensor.size(), 0.0f);
            for (size_t i = 0; i < velocity.size(); ++i) {
                velocity[i] = 0.001f * (m_current_attention - m_current_entropy) * (1.0f + static_cast<float>((i % 7)));
            }
            m_flow_layer->flow(m_current_metric_tensor, velocity);
        }
        
        // 3. PHASE: REPLAY / EVALUATION
        m_evolver->update_fitness(0, fitness); 
        
        // 4. PHASE: META (Evolution + Law E Verification)
        std::cout << "[INFO] PHASE: META | Evolving and Verifying...\n";
        m_evolver->evolve_step(m_current_metric_tensor);
        
        const auto& mutated_delta = m_evolver->get_best_adapter();
        LegalityReport report = m_verifier->verify_mutation(mutated_delta, m_current_entropy);
        
        if (report.verdict == LegalityVerdict::LAWFUL) {
            std::cout << "[PASS] Law E: Mutation verified as LAWFUL.\n";
            ReplayTrace trace;
            trace.tick = m_total_ticks;
            trace.fold_id = fold_id;
            trace.expert_id = routed_to_specialist ? "Specialist" : "AgentReasoning";
            trace.fitness_score = fitness;
            trace.applied_delta = mutated_delta;
            m_replay_manager->record_trace(trace);
        } else {
            std::cout << "[WARN] Law E: Mutation REJECTED | " << report.message << "\n";
        }
        
        // 5. PHASE: PROJECTION
        std::cout << "[INFO] PHASE: PROJECTION | Projecting Tensor State to CSS and SHM\n";
        ProjectionState p_state = {
            m_current_entropy,
            m_current_attention,
            m_current_pressure,
            m_current_gravity,
            affinity,
            static_cast<uint32_t>(fold_id),
            "AgentProjection",
            static_cast<float>(std::accumulate(m_current_arc_weights.begin(), m_current_arc_weights.end(), 0.0f))
        };
        std::cout << m_projection_expert->project_to_css(p_state) << "\n";

        // Update Shared Memory (Law B/C/D telemetry)
        if (m_shm) {
            m_shm->update_state((uint32_t)m_total_ticks, (uint32_t)fold_id, m_current_entropy, m_current_attention, m_current_pressure);
        }
    }

    float call_gguf_inference(const std::string& query) {
        // S#002c option A: IMPLEMENTED bridge (header-inline, so it lives in the harness build; the
        // kernel .cpp are untouched). Actually POSTs to a live model server and derives a fitness
        // that VARIES with the response, so the General path stops feeding the evolver a constant.
        const char* ep = get_env_value("GGUF_ENDPOINT");
        std::string url = ep ? std::string(ep) : "http://127.0.0.1:5000/v1/chat/completions";
        std::cout << "[INFO] Bridge: Executing GGUF call to " << url << "...\n";

        std::string q; for (char c : query) { if (c == '"' || c == '\\') q += '\\'; if (c == '\n' || c == '\r') { q += ' '; continue; } q += c; }
        std::string payload = "{\"messages\":[{\"role\":\"user\",\"content\":\"" + q + "\"}],\"max_tokens\":24,\"temperature\":0.7}";
        { std::ofstream pf("s002c_payload.json", std::ios::binary); pf << payload; }
        std::string command = "curl -s -X POST -H \"Content-Type: application/json\" --data-binary @s002c_payload.json \"" + url + "\"";

        std::string resp;
        FILE* p = _popen(command.c_str(), "r");
        if (p) { char buf[512]; size_t n; while ((n = fread(buf, 1, sizeof(buf), p)) > 0) resp.append(buf, n); _pclose(p); }

        float fitness = 0.5f;
        if (!resp.empty()) {
            size_t h = std::hash<std::string>{}(resp);
            fitness = 0.30f + 0.69f * static_cast<float>((h % 1000) / 1000.0);   // wide, response-driven
        } else {
            std::cout << "[WARN] Bridge: empty response (is the model server up?).\n";
        }
        std::cout << "[INFO] Bridge: response " << resp.size() << " bytes -> fitness " << fitness << "\n";
        return fitness;
    }

    
    void load_manifest(const std::string& manifest_path) {
        std::cout << "[INFO] Loading Unified Manifest: " << manifest_path << "\n";
        
        std::ifstream f(manifest_path);
        if (!f.is_open()) {
            std::cerr << "❌ Error: Could not open manifest file: " << manifest_path << "\n";
            return;
        }
        
        try {
            json manifest = json::parse(f);
            
            // 1. Process Base Model Shards
            if (manifest.contains("base_model") && manifest["base_model"].contains("shards")) {
                size_t total_size = 0;
                for (auto& shard : manifest["base_model"]["shards"]) {
                    DdsShardMetadata meta;
                    meta.filename = shard["path"];
                    meta.size_bytes = shard["size"];
                    meta.width = 4096; // Assumption for tiled residency
                    meta.height = (uint32_t)((meta.size_bytes + (4096 * 4) - 1) / (4096 * 4));
                    meta.format = DXGI_FORMAT_R32_FLOAT;
                    
                    int fold_id = std::stoi(shard["fold"].get<std::string>());
                    m_loader->register_shard(fold_id, meta);
                    total_size += meta.size_bytes;
                }
                
                // Initialize virtual buffer for the whole model
                if (m_device) {
                    m_loader->initialize_reserved_resource(total_size);
                    std::cout << "[INFO] DX12: Reserved " << (total_size / (1024*1024)) << "MB for base model.\n";
                }
            }
            
            // 2. Process Experts / Specialist Shards
            if (manifest.contains("shards")) {
                for (auto& entry : manifest["shards"]) {
                    int id = entry["id"];
                    std::string path = entry["path"];
                    if (path.find(".xshard") != std::string::npos) {
                        XShardMetadata xmeta;
                        xmeta.filename = path;
                        m_loader->register_xshard(id, xmeta);
                        std::cout << "[INFO] Registered XShard Specialist: " << entry["name"] << " (ID: " << id << ")\n";
                    }
                }
            }
            
            if (manifest.contains("experts")) {
                for (auto& expert : manifest["experts"]) {
                    (void)expert;
                }
            }

            std::cout << "[PASS] Manifest Loaded Successfully.\n";
            
        } catch (json::parse_error& e) {
            std::cerr << "❌ Error: Manifest parse error: " << e.what() << "\n";
        }
    }
};
