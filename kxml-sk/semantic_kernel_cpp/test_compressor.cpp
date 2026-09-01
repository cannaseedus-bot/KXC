// ============================================================================
// test_compressor.cpp - MemoryCompressor Integrity Test (Phase 5 Robustness)
// ============================================================================

#include "include/memory_compressor.h"
#include <iostream>
#include <cassert>
#include <vector>

int main() {
    std::cout << "[TEST] Verifying MemoryCompressor Integrity...\n";
    
    MemoryCompressor compressor;
    
    // 1. Create Sample Data
    std::vector<float> original = { 0.1f, 0.5f, 0.9f, -0.3f, 1.2f, 0.0f };
    
    // 2. Compress
    CompressedBlock block = compressor.compress_fractal(original);
    std::cout << "   [TEST] Compression Ratio: " << compressor.calculate_intelligence_ratio(block) << "\n";
    
    // 3. Decompress
    std::vector<float> restored = compressor.decompress_fractal(block);
    
    // 4. Verify (within quantization error bounds)
    for (size_t i = 0; i < original.size(); ++i) {
        float diff = std::abs(original[i] - restored[i]);
        if (diff > 0.05f) { // Allow for some quantization error
            std::cerr << "❌ [FAIL] Data mismatch at index " << i << ": " << original[i] << " vs " << restored[i] << "\n";
            return 1;
        }
    }
    
    std::cout << "[PASS] MemoryCompressor integrity verified.\n";
    return 0;
}
