// ============================================================================
// metric_normalization.h - Riemannian Manifold Normalization (ASX v0.7)
// ============================================================================

#pragma once

#include <vector>
#include <cmath>
#include <iostream>

namespace asx {

class MetricNormalization {
private:
    uint32_t m_dim;
    std::vector<float> m_metric_diagonal; // Simplified diagonal metric g_ii
    
public:
    MetricNormalization(uint32_t dim);
    void set_metric(const std::vector<float>& diagonal);
    float compute_manifold_norm(const std::vector<float>& x);
    void normalize(std::vector<float>& x);
    const std::vector<float>& get_metric() const { return m_metric_diagonal; }
};


} // namespace asx
