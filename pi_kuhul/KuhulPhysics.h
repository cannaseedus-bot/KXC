// KuhulPhysics.h — K'UHUL Physics Solver (C++ / DirectXMath)
//
// Automatically adjusts gravity based on loss/gradient observations.
// Integrates with KXML graph execution and training loops.
//
// Three Laws of K'UHUL Physics:
//   Law 1 (Phase Constraint):     no node executes before its phase is ready
//   Law 2 (Antigravity Permit):   antigravity nodes bypass phase check + gradient clip
//   Law 3 (Gradient Conservation):||∇|| ≤ gravity_scale × Lipschitz_constant
//
// Node taxonomy:
//   ⟁Grav⟁     gravity_scale = 1.0   constrained, phase-gated, Lipschitz-bounded
//   ⟁AntiGrav⟁ gravity_scale = 0.0   float, bypass phase gates — the [dbg] lines
//   ⟁HeavyGrav⟁gravity_scale = 2.0   extra constraints (loss head, output head)
//   ⟁NegGrav⟁  gravity_scale = -1.0  regularisation repulsion (L1 sparsity)
//
// Field equation: ∇²Φ = ρ_gravity + ρ_antigravity
// Stable when: ρ_gravity / ρ_antigravity ≥ 10  (matches AdamW weight_decay/lr ratio)
//
// Current math µMODEL status (step 700, loss=10.03):
//   approaching event horizon → solver Rule 1 triggers → tighten logit_bound 20→18
//   loss oscillating (std ≈ 0.45 × mean) → Rule 6 triggers → increase grad gravity

#pragma once
#include <DirectXMath.h>
#include <deque>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace DirectX;

// ─── K'UHUL Physics Constants ─────────────────────────────────────────────────

namespace KuhulConstants {
    constexpr float GRAVITY_BASE       = 1.0f;   // Normal gravity
    constexpr float ANTIGRAVITY_BASE   = 0.1f;   // Default float permission
    constexpr float EVENT_HORIZON      = 10.0f;  // Loss → collapse threshold
    constexpr float ESCAPE_VELOCITY    = 20.0f;  // Max safe logit magnitude
    constexpr float ORBIT_STABLE_LOW   = 0.5f;   // Target loss range low
    constexpr float ORBIT_STABLE_HIGH  = 5.0f;   // Target loss range high
    constexpr size_t WINDOW_SIZE       = 100;    // Observation history window
}

// ─── PhysicsState ─────────────────────────────────────────────────────────────

struct PhysicsState {
    // Gravity field parameters (dynamic — adjusted by solver)
    float logitGravity;      // Logit clamp bound (↓ = tighter)  default 20.0
    float gradGravity;       // Gradient clip norm (↓ = tighter)  default 1.0
    float lossGravity;       // Loss cap value     (↓ = tighter)  default 10.0
    float antigravityFloat;  // Float permission for antigravity nodes  default 0.1
    float debugFloat;        // Debug/telemetry nodes always = 1.0 (fully float)

    // Current observations
    float currentLoss;
    float currentGradNorm;
    float currentLogitMax;
    float lossVelocity;      // d(loss)/d(step)
    float lossAcceleration;  // d²(loss)/d(step)²

    // Orbital state flags
    bool inOrbit;                   // loss ∈ [ORBIT_STABLE_LOW, ORBIT_STABLE_HIGH]
    bool approachingEventHorizon;   // loss > EVENT_HORIZON * 0.8
    bool atEscapeVelocity;          // logitMax > ESCAPE_VELOCITY

    // History deques (bounded to WINDOW_SIZE)
    std::deque<float> lossHistory;
    std::deque<float> gradNormHistory;
    std::deque<float> logitMaxHistory;

    PhysicsState()
        : logitGravity(20.0f), gradGravity(1.0f), lossGravity(10.0f)
        , antigravityFloat(0.1f), debugFloat(1.0f)
        , currentLoss(0), currentGradNorm(0), currentLogitMax(0)
        , lossVelocity(0), lossAcceleration(0)
        , inOrbit(false), approachingEventHorizon(false), atEscapeVelocity(false) {}
};

// ─── KuhulPhysicsSolver ───────────────────────────────────────────────────────

class KuhulPhysicsSolver {
    PhysicsState m_state;
    size_t       m_step;
    float        m_targetLow, m_targetHigh;

    void _pushHistory(float loss, float gradNorm, float logitMax) {
        auto push = [](std::deque<float>& d, float v) {
            d.push_back(v);
            while (d.size() > KuhulConstants::WINDOW_SIZE) d.pop_front();
        };
        push(m_state.lossHistory,     loss);
        push(m_state.gradNormHistory, gradNorm);
        push(m_state.logitMaxHistory, logitMax);
    }

    void _computeDerivatives() {
        const auto& h = m_state.lossHistory;
        if (h.size() >= 2) m_state.lossVelocity = h.back() - h[h.size()-2];
        if (h.size() >= 3) m_state.lossAcceleration = m_state.lossVelocity - (h[h.size()-2] - h[h.size()-3]);
    }

    void _detectOrbitalState() {
        m_state.approachingEventHorizon = (m_state.currentLoss > KuhulConstants::EVENT_HORIZON * 0.8f);
        m_state.atEscapeVelocity        = (m_state.currentLogitMax > KuhulConstants::ESCAPE_VELOCITY);
        m_state.inOrbit                 = (m_state.currentLoss >= m_targetLow && m_state.currentLoss <= m_targetHigh);
    }

    void _adjustGravity() {
        // Rule 1: Approaching event horizon → tighten all bounds
        if (m_state.approachingEventHorizon || m_state.currentLoss > KuhulConstants::EVENT_HORIZON) {
            float excess = std::min(1.0f, (m_state.currentLoss - KuhulConstants::EVENT_HORIZON) / KuhulConstants::EVENT_HORIZON);
            float k      = 1.0f + excess * 2.0f;
            m_state.logitGravity = std::max(5.0f,  m_state.logitGravity * (1.0f - k * 0.10f));
            m_state.gradGravity  = std::max(0.1f,  m_state.gradGravity  * (1.0f - k * 0.20f));
            m_state.lossGravity  = std::max(1.0f,  m_state.lossGravity  * (1.0f - k * 0.15f));
            if (m_step % 10 == 0)
                std::cout << "[K'UHUL] Event horizon! logit_bound=" << m_state.logitGravity
                          << " grad_clip=" << m_state.gradGravity << "\n";
        }

        // Rule 2: Escape velocity → tighten logit bound
        if (m_state.atEscapeVelocity) {
            float excess = std::min(2.0f, (m_state.currentLogitMax - KuhulConstants::ESCAPE_VELOCITY) / KuhulConstants::ESCAPE_VELOCITY);
            m_state.logitGravity = std::max(5.0f, m_state.logitGravity * (1.0f - (1.0f + excess) * 0.20f));
            if (m_step % 10 == 0)
                std::cout << "[K'UHUL] Escape velocity! logit_bound=" << m_state.logitGravity << "\n";
        }

        // Rule 3: Loss accelerating upward → increase gradient gravity
        if (m_state.lossAcceleration > 0.5f && m_state.lossVelocity > 0) {
            float k = 1.0f + std::min(1.0f, m_state.lossAcceleration);
            m_state.gradGravity = std::max(0.1f, m_state.gradGravity * (1.0f - k * 0.10f));
        }

        // Rule 4: In stable orbit → relax constraints (allow more antigravity)
        if (m_state.inOrbit) {
            const float relax     = 0.01f;
            m_state.logitGravity  = std::min(30.0f,  m_state.logitGravity  * (1.0f + relax));
            m_state.gradGravity   = std::min(2.0f,   m_state.gradGravity   * (1.0f + relax));
            m_state.lossGravity   = std::min(15.0f,  m_state.lossGravity   * (1.0f + relax));
            m_state.antigravityFloat = std::min(1.0f, m_state.antigravityFloat + 0.005f);
            if (m_step % 50 == 0 && m_step > 0)
                std::cout << "[K'UHUL] In orbit. Relaxing: logit_bound=" << m_state.logitGravity
                          << " antigravity=" << m_state.antigravityFloat << "\n";
        }

        // Rule 5: Loss too low (underfitting) → relax
        if (m_state.currentLoss < m_targetLow && !m_state.approachingEventHorizon) {
            m_state.logitGravity     = std::min(40.0f, m_state.logitGravity * 1.02f);
            m_state.antigravityFloat = std::min(0.5f,  m_state.antigravityFloat + 0.01f);
        }

        // Rule 6: Gradient oscillation → increase gravity
        if (m_state.gradNormHistory.size() >= 10) {
            float mean = 0, variance = 0;
            size_t N = m_state.gradNormHistory.size();
            for (size_t i = N - 10; i < N; i++) mean += m_state.gradNormHistory[i];
            mean /= 10.0f;
            for (size_t i = N - 10; i < N; i++) { float d = m_state.gradNormHistory[i] - mean; variance += d*d; }
            variance /= 10.0f;
            if (variance > 0.5f && mean > 0.5f) {
                m_state.gradGravity = std::max(0.2f, m_state.gradGravity * 0.9f);
                if (m_step % 50 == 0)
                    std::cout << "[K'UHUL] Oscillation! grad_gravity=" << m_state.gradGravity << "\n";
            }
        }
    }

public:
    KuhulPhysicsSolver(float targetLow  = KuhulConstants::ORBIT_STABLE_LOW,
                       float targetHigh = KuhulConstants::ORBIT_STABLE_HIGH)
        : m_step(0), m_targetLow(targetLow), m_targetHigh(targetHigh) {}

    // Feed one step's observations → auto-adjusts gravity
    void Observe(float loss, float gradNorm, float logitMax) {
        ++m_step;
        m_state.currentLoss     = loss;
        m_state.currentGradNorm = gradNorm;
        m_state.currentLogitMax = logitMax;
        _pushHistory(loss, gradNorm, logitMax);
        _computeDerivatives();
        _detectOrbitalState();
        _adjustGravity();
    }

    // SIMD-accelerated application (DirectXMath)
    XMVECTOR ApplyLogitGravity(XMVECTOR logits) const {
        float b = m_state.logitGravity;
        return XMVectorClamp(logits, XMVectorReplicate(-b), XMVectorReplicate(b));
    }
    XMVECTOR ApplyGradGravity(XMVECTOR gradient) const {
        float norm = XMVectorGetX(XMVectorSqrt(XMVector4Dot(gradient, gradient)));
        if (norm > m_state.gradGravity && norm > 0.0f)
            return XMVectorMultiply(gradient, XMVectorReplicate(m_state.gradGravity / norm));
        return gradient;
    }
    float ApplyLossGravity(float loss) const { return std::min(loss, m_state.lossGravity); }

    // Getters
    float GetLogitBound()       const { return m_state.logitGravity; }
    float GetGradClipNorm()     const { return m_state.gradGravity; }
    float GetLossCap()          const { return m_state.lossGravity; }
    float GetAntigravityFloat() const { return m_state.antigravityFloat; }
    float GetDebugFloat()       const { return m_state.debugFloat; }
    bool  IsInOrbit()           const { return m_state.inOrbit; }
    bool  IsApproachingHorizon()const { return m_state.approachingEventHorizon; }

    // Field equation residual: ∇²Φ = ρ_grav + ρ_anti
    struct FieldResidual { float rho_grav, rho_anti, laplacian; bool equilibrium; };
    FieldResidual FieldEquation() const {
        float rg = m_state.logitGravity + m_state.gradGravity + m_state.lossGravity;
        float ra = m_state.antigravityFloat + m_state.debugFloat;
        float lap = rg - 10.0f * ra;
        return { rg, ra, lap, std::fabsf(lap) < ra };
    }

    void Report() const {
        std::cout << "\n[K'UHUL] Physics State (step " << m_step << ")\n"
                  << "  Gravity:     logit_bound=" << m_state.logitGravity
                  << "  grad_clip=" << m_state.gradGravity
                  << "  loss_cap=" << m_state.lossGravity << "\n"
                  << "  Antigravity: float=" << m_state.antigravityFloat
                  << "  debug=" << m_state.debugFloat << "\n"
                  << "  Loss=" << m_state.currentLoss
                  << (m_state.inOrbit ? " (IN ORBIT)" : "")
                  << (m_state.approachingEventHorizon ? " (EVENT HORIZON!)" : "") << "\n"
                  << "  GradNorm=" << m_state.currentGradNorm
                  << "  LogitMax=" << m_state.currentLogitMax << "\n"
                  << "  Velocity=" << m_state.lossVelocity
                  << "  Accel=" << m_state.lossAcceleration << "\n";
        auto r = FieldEquation();
        std::cout << "  ∇²Φ: rho_grav=" << r.rho_grav
                  << " rho_anti=" << r.rho_anti
                  << " laplacian=" << r.laplacian
                  << (r.equilibrium ? " (EQUILIBRIUM)" : "") << "\n";
    }
};

// ─── KuhulTrainer — wraps physics solver into a training loop ─────────────────

class KuhulTrainer {
    KuhulPhysicsSolver m_physics;
    size_t m_step;
    float  m_ema;       // exponential moving average of loss

public:
    KuhulTrainer(float targetLow = 0.5f, float targetHigh = 5.0f)
        : m_physics(targetLow, targetHigh), m_step(0), m_ema(0) {}

    // Call once per training step with raw observations
    void TrainStep(float rawLoss, float rawGradNorm, float rawLogitMax) {
        ++m_step;
        m_ema = (m_step == 1) ? rawLoss : (0.95f * m_ema + 0.05f * rawLoss);
        m_physics.Observe(m_ema, rawGradNorm, rawLogitMax);
        if (m_step % 100 == 0) m_physics.Report();
        if (m_physics.IsApproachingHorizon())
            std::cout << "[K'UHUL] EMERGENCY: Loss approaching event horizon at step " << m_step << "!\n";
    }

    float GetLogitBound()       const { return m_physics.GetLogitBound(); }
    float GetGradClipNorm()     const { return m_physics.GetGradClipNorm(); }
    float GetLossCap()          const { return m_physics.GetLossCap(); }
    float GetAntigravityFloat() const { return m_physics.GetAntigravityFloat(); }

    XMVECTOR ApplyLogitGravity(XMVECTOR v) const { return m_physics.ApplyLogitGravity(v); }
    XMVECTOR ApplyGradGravity(XMVECTOR v)  const { return m_physics.ApplyGradGravity(v); }
    float    ApplyLossGravity(float loss)   const { return m_physics.ApplyLossGravity(loss); }
};
