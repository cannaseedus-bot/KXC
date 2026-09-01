// ============================================================================
// metric_normalization.cpp - Riemannian Manifold Normalization (ASX v0.7)
// ============================================================================

#include "../include/metric_normalization.h"

namespace asx {

MetricNormalization::MetricNormalization(uint32_t dim) : m_dim(dim) {
    // Initialize with identity metric (flat space)
    m_metric_diagonal.resize(dim, 1.0f);
}

void MetricNormalization::set_metric(const std::vector<float>& diagonal) {
    if (diagonal.size() == m_dim) {
        m_metric_diagonal = diagonal;
    }
}

// ⟁ Metric Norm: ||x||_g = sqrt( sum( g_ii * x_i^2 ) )
float MetricNormalization::compute_manifold_norm(const std::vector<float>& x) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < m_dim; ++i) {
        sum_sq += m_metric_diagonal[i] * x[i] * x[i];
    }
    return std::sqrt(sum_sq + 1e-6f);
}

// ⟁ Normalize on Manifold
void MetricNormalization::normalize(std::vector<float>& x) {
    float norm = compute_manifold_norm(x);
    for (float& val : x) {
        val /= norm;
    }
}

} // namespace asx
