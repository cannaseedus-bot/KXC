// ============================================================================
// memory_compressor.h - SCXQ2 Fractal Compression (ASX v0.7)
// ============================================================================

#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <algorithm>

struct CompressedBlock {
    float scale;
    float offset;
    std::vector<int8_t> deltas; // 8-bit quantized deltas
    size_t original_size;
};

class MemoryCompressor {
public:
    // ⟁ SCXQ2 Fractal Compression
    // Compresses a float array by finding its min/max scale and quantizing to int8 deltas.
    CompressedBlock compress_fractal(const std::vector<float>& data) {
        CompressedBlock block;
        block.original_size = data.size();
        
        if (data.empty()) return block;
        
        // 1. Find Range (Self-Similarity Base)
        auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
        float min_val = *min_it;
        float max_val = *max_it;
        
        block.offset = min_val;
        float range = max_val - min_val;
        block.scale = (range > 0) ? (range / 255.0f) : 1.0f;
        
        // 2. Quantize to 8-bit Deltas
        block.deltas.reserve(data.size());
        for (float v : data) {
            int8_t q = static_cast<int8_t>(std::round((v - min_val) / block.scale) - 128);
            block.deltas.push_back(q);
        }
        
        return block;
    }
    
    // ⟁ Decompress
    std::vector<float> decompress_fractal(const CompressedBlock& block) {
        std::vector<float> result;
        result.reserve(block.original_size);
        
        for (int8_t q : block.deltas) {
            float v = (static_cast<float>(q) + 128) * block.scale + block.offset;
            result.push_back(v);
        }
        
        return result;
    }
    
    // ⟁ Intelligence Metric
    // Intelligence = 1 - (compressed_size / original_size)
    float calculate_intelligence_ratio(const CompressedBlock& block) {
        float original_bytes = static_cast<float>(block.original_size) * static_cast<float>(sizeof(float));
        float compressed_bytes = static_cast<float>(block.deltas.size()) * static_cast<float>(sizeof(int8_t)) + static_cast<float>(sizeof(float) * 2);
        return 1.0f - (compressed_bytes / original_bytes);
    }
};
