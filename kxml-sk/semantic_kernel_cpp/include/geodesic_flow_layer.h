// ============================================================================
// geodesic_flow_layer.h - Geodesic Manifold Traversal (ASX v0.7)
// ============================================================================

#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include "metric_normalization.h"

namespace asx {

class GeodesicFlowLayer {
private:
    uint32_t m_dim;
    std::shared_ptr<MetricNormalization> m_norm;
    
    // Connection Coefficients (Γ - Christoffel Symbols)
    // Simplified: stored as a vector representing the diagonal of the connection
    std::vector<float> m_christoffel_diag;
    std::vector<float> m_arc_weights;

    float m_gravity_strength = 9.80665f;
    float m_entropy = 0.14f;
    float m_pressure = 0.34f;
    
    float m_step_size = 0.1f;
    
public:
    GeodesicFlowLayer(uint32_t dim);
    void flow(std::vector<float>& x, const std::vector<float>& velocity);
    void set_connection(const std::vector<float>& christoffel);
    void set_gravity(float gravity);
    void set_thermodynamics(float entropy, float pressure);
    void set_arc_weights(const std::vector<float>& arc_weights);
    float get_gravity() const { return m_gravity_strength; }
    const std::vector<float>& get_arc_weights() const { return m_arc_weights; }
    std::shared_ptr<MetricNormalization> get_norm() { return m_norm; }
};


} // namespace asx
