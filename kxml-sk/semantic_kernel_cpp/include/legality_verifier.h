// ============================================================================
// legality_verifier.h - MX2⟁☣ Governance (ASX v0.7)
// ============================================================================

#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include "evolution_bot.h"

enum class LegalityVerdict {
    LAWFUL,
    CAUSALITY_VIOLATION,
    BOUNDARY_BREACH,
    TENSOR_INSTABILITY,
    REJECTED
};

struct LegalityReport {
    LegalityVerdict verdict;
    std::string message;
    float entropy_drift;
};

class LegalityVerifier {
private:
    float m_max_mutation_delta = 0.5f;
    float m_max_entropy_threshold = 0.85f;
    
public:
    // ⟁ Verify Mutation (Evolution check)
    LegalityReport verify_mutation(const AdapterDelta& delta, float local_entropy) {
        LegalityReport report;
        report.verdict = LegalityVerdict::LAWFUL;
        report.entropy_drift = 0.0f;
        
        // 1. Check Entropy Boundary (Law B)
        if (local_entropy > m_max_entropy_threshold) {
            report.verdict = LegalityVerdict::BOUNDARY_BREACH;
            report.message = "Local entropy exceeds governance threshold.";
            return report;
        }
        
        // 2. Check Tensor Stability
        for (float w : delta.weights) {
            if (std::isnan(w) || std::isinf(w)) {
                report.verdict = LegalityVerdict::TENSOR_INSTABILITY;
                report.message = "Weights contain non-finite values.";
                return report;
            }
            
            // Limit the absolute delta to prevent catastrophic drift
            if (std::abs(w) > m_max_mutation_delta) {
                report.verdict = LegalityVerdict::BOUNDARY_BREACH;
                report.message = "Mutation delta exceeds stability bounds.";
                return report;
            }
        }
        
        // 3. Causality Check (Simplified)
        if (delta.generation == 0 && local_entropy < 0.01f) {
             report.verdict = LegalityVerdict::CAUSALITY_VIOLATION;
             report.message = "Static manifold cannot evolve without exploration pressure.";
             return report;
        }
        
        return report;
    }
    
    // ⟁ Verify Traversal (Geodesic check)
    bool verify_traversal(int from_fold, int to_fold, float geodesic_cost) {
        // Traversal is unlawful if cost is negative (anti-gravity) or too high
        if (geodesic_cost < 0.0f || geodesic_cost > 5.0f) return false;
        return true;
    }
};
