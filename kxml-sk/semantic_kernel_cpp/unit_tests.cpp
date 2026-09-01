// ============================================================================
// unit_tests.cpp - Core ASX-RUNTIME Component Tests (Phase 5 Robustness)
// ============================================================================

#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "include/legality_verifier.h"
#include "include/replay_lane_manager.h"
#include "include/evolution_bot.h"
#include "include/metric_normalization.h"

void test_legality_verifier() {
    std::cout << "   [TEST] LegalityVerifier...";
    LegalityVerifier verifier;
    
    // 1. Valid mutation
    AdapterDelta lawful_delta = { "shard_0", "shard_0", {0.1f}, 0.2f, 100 }; // shard_id, id, weights, fitness, gen
    LegalityReport report1 = verifier.verify_mutation(lawful_delta, 0.5f);
    assert(report1.verdict == LegalityVerdict::LAWFUL);
    
    // 2. Entropy breach
    AdapterDelta entropy_delta = { "shard_0", "shard_0", {0.1f}, 0.2f, 100 };
    LegalityReport report2 = verifier.verify_mutation(entropy_delta, 0.9f); // entropy > 0.85
    assert(report2.verdict == LegalityVerdict::UNLAWFUL);
    assert(report2.message.find("ENTROPY") != std::string::npos);
    
    // 3. Weight instability
    AdapterDelta unstable_delta = { "shard_0", "shard_0", {0.6f}, 0.1f, 100 }; // attention delta > 0.5
    LegalityReport report3 = verifier.verify_mutation(unstable_delta, 0.5f);
    assert(report3.verdict == LegalityVerdict::UNLAWFUL);
    assert(report3.message.find("INSTABILITY") != std::string::npos);
    
    std::cout << " PASSED\n";
}


void test_replay_lane_manager() {
    std::cout << "   [TEST] ReplayLaneManager...";
    ReplayLaneManager manager;
    
    ReplayTrace trace;
    trace.tick = 1001;
    trace.fold_id = 4;
    trace.fitness_score = 0.941f;
    
    manager.record_trace(trace);
    
    float affinity = manager.get_replay_affinity(4);
    assert(affinity > 0.0f);
    
    // Test fold with no history
    float null_affinity = manager.get_replay_affinity(99);
    assert(null_affinity == 0.0f);
    
    std::cout << " PASSED\n";
}

void test_metric_normalization() {
    std::cout << "   [TEST] MetricNormalization...";
    asx::MetricNormalization norm(4);
    
    // Set a non-flat metric
    std::vector<float> metric = { 2.0f, 1.0f, 0.5f, 1.0f };
    norm.set_metric(metric);
    
    std::vector<float> x = { 1.0f, 1.0f, 1.0f, 1.0f };
    norm.normalize(x);
    
    // Manifold norm should now be 1.0
    float manifold_norm = norm.compute_manifold_norm(x);
    assert(std::abs(manifold_norm - 1.0f) < 1e-4f);
    
    std::cout << " PASSED\n";
}

int main() {
    std::cout << "[ASX] STARTING CORE UNIT TESTS\n";
    std::cout << "----------------------------------\n";
    
    try {
        test_legality_verifier();
        test_replay_lane_manager();
        test_metric_normalization();
        
        std::cout << "----------------------------------\n";
        std::cout << "[PASS] All core components verified.\n";
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
