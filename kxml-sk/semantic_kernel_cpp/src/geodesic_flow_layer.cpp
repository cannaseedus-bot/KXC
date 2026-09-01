// ============================================================================
// geodesic_flow_layer.cpp - Geodesic Manifold Traversal (ASX v0.7)
// ============================================================================

#include "../include/geodesic_flow_layer.h"

#include <algorithm>
#include <cmath>

namespace asx {

GeodesicFlowLayer::GeodesicFlowLayer(uint32_t dim) : m_dim(dim) {
    m_norm = std::make_shared<MetricNormalization>(dim);
    m_christoffel_diag.resize(dim, 0.01f);
    m_arc_weights.resize(dim, 1.0f / std::sqrt(static_cast<float>(dim)));
}

// ⟁ Geodesic Flow
// Replaces FFN. Solves d²x/dt² + Γ dx/dt dx/dt = 0
void GeodesicFlowLayer::flow(std::vector<float>& x, const std::vector<float>& velocity) {
    // 1. Compute Acceleration (Correction for manifold curvature)
    // a_k = - Γ^k_ij v^i v^j - gravity * ARC_weight_k * x_k
    const size_t limit = std::min(x.size(), velocity.size());
    std::vector<float> acceleration(limit);
    const float thermodynamic_gate = std::clamp(1.0f + 0.5f * m_pressure - 0.5f * m_entropy, 0.1f, 4.0f);
    const float gravity = std::max(0.0f, m_gravity_strength * thermodynamic_gate);
    for (size_t i = 0; i < limit; ++i) {
        const float arc_weight = m_arc_weights.empty() ? 1.0f : m_arc_weights[i % m_arc_weights.size()];
        acceleration[i] = -m_christoffel_diag[i] * velocity[i] * velocity[i]
                          - gravity * arc_weight * x[i];
    }
    
    // 2. Step Position along Geodesic
    // x_new = x + v*dt + 0.5*a*dt²
    for (size_t i = 0; i < limit; ++i) {
        x[i] += velocity[i] * m_step_size + 0.5f * acceleration[i] * m_step_size * m_step_size;
    }
    
    // 3. Project back to Manifold surface
    m_norm->normalize(x);
}

void GeodesicFlowLayer::set_connection(const std::vector<float>& christoffel) {
    if (christoffel.size() == m_dim) {
        m_christoffel_diag = christoffel;
    }
}

void GeodesicFlowLayer::set_gravity(float gravity) {
    m_gravity_strength = gravity;
}

void GeodesicFlowLayer::set_thermodynamics(float entropy, float pressure) {
    m_entropy = entropy;
    m_pressure = pressure;
}

void GeodesicFlowLayer::set_arc_weights(const std::vector<float>& arc_weights) {
    if (!arc_weights.empty()) {
        m_arc_weights = arc_weights;
    }
}

} // namespace asx
