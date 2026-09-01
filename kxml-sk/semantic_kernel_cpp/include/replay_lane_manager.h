// ============================================================================
// replay_lane_manager.h - Temporal Cognition Persistence (ASX v0.7)
// ============================================================================

#pragma once

#include <vector>
#include <string>
#include <map>
#include <deque>
#include <fstream>
#include <chrono>
#include "evolution_bot.h" // For AdapterDelta
#include "memory_compressor.h"

struct ReplayTrace {
    uint64_t tick;
    int fold_id;
    std::string expert_id;
    float fitness_score;
    std::vector<float> manifold_coords; // Sample of state at this time
    AdapterDelta applied_delta;         // The evolution state at this time
};

class ReplayLaneManager {
private:
    // Active "Lane" of recent traces (stored in memory for Law B affinity)
    std::deque<ReplayTrace> m_active_lane;
    std::shared_ptr<MemoryCompressor> m_compressor;
    size_t m_max_lane_size = 1000;
    
    // Persistent Storage Config
    std::string m_storage_path = "replay_lanes/";
    uint32_t m_archive_interval = 100; // Archive every 100 traces
    
public:
    ReplayLaneManager() {
        m_compressor = std::make_shared<MemoryCompressor>();
    }
    
    void record_trace(const ReplayTrace& trace) {
        m_active_lane.push_back(trace);
        
        // Enforce memory limits for the active lane
        if (m_active_lane.size() > m_max_lane_size) {
            m_active_lane.pop_front();
        }
        
        // Trigger background archive if interval reached
        if (trace.tick % m_archive_interval == 0) {
            archive_to_disk();
        }
    }
    
    // ⟁ LAW B: Replay Affinity
    // Returns a bias weight based on historical fitness for a given fold
    float get_replay_affinity(int fold_id) const {
        float total_fitness = 0.0f;
        int count = 0;
        
        for (const auto& trace : m_active_lane) {
            if (trace.fold_id == fold_id) {
                total_fitness += trace.fitness_score;
                count++;
            }
        }
        
        return (count > 0) ? (total_fitness / count) : 0.5f;
    }


    // ⟁ Persistent Archival with SCXQ2 Fractal Compression
    void archive_to_disk() {
        if (m_active_lane.empty()) return;
        
        std::string filename = m_storage_path + "session_" + std::to_string(m_active_lane.back().tick) + ".replay";
        std::ofstream ofs(filename, std::ios::binary);
        
        if (!ofs) return;
        
        // 1. Write Header
        ofs.write("REPLAY", 6);
        uint32_t count = static_cast<uint32_t>(m_active_lane.size());
        ofs.write(reinterpret_cast<char*>(&count), sizeof(count));
        
        // 2. Write Compressed Traces
        for (const auto& trace : m_active_lane) {
            // Write Metadata
            ofs.write(reinterpret_cast<const char*>(&trace.tick), sizeof(trace.tick));
            ofs.write(reinterpret_cast<const char*>(&trace.fold_id), sizeof(trace.fold_id));
            ofs.write(reinterpret_cast<const char*>(&trace.fitness_score), sizeof(trace.fitness_score));
            
            // Compress Manifold Coords
            CompressedBlock c_coords = m_compressor->compress_fractal(trace.manifold_coords);
            ofs.write(reinterpret_cast<char*>(&c_coords.scale), sizeof(float));
            ofs.write(reinterpret_cast<char*>(&c_coords.offset), sizeof(float));
            uint32_t coord_size = static_cast<uint32_t>(c_coords.deltas.size());
            ofs.write(reinterpret_cast<char*>(&coord_size), sizeof(uint32_t));
            ofs.write(reinterpret_cast<char*>(c_coords.deltas.data()), coord_size);
            
            // Compress Weights
            CompressedBlock c_weights = m_compressor->compress_fractal(trace.applied_delta.weights);
            ofs.write(reinterpret_cast<char*>(&c_weights.scale), sizeof(float));
            ofs.write(reinterpret_cast<char*>(&c_weights.offset), sizeof(float));
            uint32_t weight_size = static_cast<uint32_t>(c_weights.deltas.size());
            ofs.write(reinterpret_cast<char*>(&weight_size), sizeof(uint32_t));
            ofs.write(reinterpret_cast<char*>(c_weights.deltas.data()), weight_size);
            
            std::cout << "[INFO] Replay Archived | Tick " << trace.tick 
                      << " | Compression Ratio: " << m_compressor->calculate_intelligence_ratio(c_weights) << "\n";
        }
        
        ofs.close();
    }

    
    // ⟁ Recovery Logic (Phase 5 Robustness)
    bool load_from_disk(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) return false;

        char magic[6];
        ifs.read(magic, 6);
        if (std::string(magic, 6) != "REPLAY") return false;

        uint32_t count;
        ifs.read(reinterpret_cast<char*>(&count), sizeof(count));

        for (uint32_t i = 0; i < count; ++i) {
            ReplayTrace trace;
            ifs.read(reinterpret_cast<char*>(&trace.tick), sizeof(trace.tick));
            ifs.read(reinterpret_cast<char*>(&trace.fold_id), sizeof(trace.fold_id));
            ifs.read(reinterpret_cast<char*>(&trace.fitness_score), sizeof(trace.fitness_score));

            // Decompress Coords
            CompressedBlock c_coords;
            ifs.read(reinterpret_cast<char*>(&c_coords.scale), sizeof(float));
            ifs.read(reinterpret_cast<char*>(&c_coords.offset), sizeof(float));
            uint32_t coord_size;
            ifs.read(reinterpret_cast<char*>(&coord_size), sizeof(uint32_t));
            c_coords.deltas.resize(coord_size);
            ifs.read(reinterpret_cast<char*>(c_coords.deltas.data()), coord_size);
            trace.manifold_coords = m_compressor->decompress_fractal(c_coords);

            // Decompress Weights
            CompressedBlock c_weights;
            ifs.read(reinterpret_cast<char*>(&c_weights.scale), sizeof(float));
            ifs.read(reinterpret_cast<char*>(&c_weights.offset), sizeof(float));
            uint32_t weight_size;
            ifs.read(reinterpret_cast<char*>(&weight_size), sizeof(uint32_t));
            c_weights.deltas.resize(weight_size);
            ifs.read(reinterpret_cast<char*>(c_weights.deltas.data()), weight_size);
            trace.applied_delta.weights = m_compressor->decompress_fractal(c_weights);
            trace.applied_delta.id = "restored_" + std::to_string(trace.tick);

            m_active_lane.push_back(trace);
        }
        
        std::cout << "[INFO] Replay Recovered | Traces: " << m_active_lane.size() << "\n";
        return true;
    }
    
    const std::deque<ReplayTrace>& get_active_lane() const { return m_active_lane; }
};

