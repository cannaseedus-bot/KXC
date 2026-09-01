// ============================================================================
// projection_expert.h - CSS Tensor Projection (ASX v0.7)
// ============================================================================

#pragma once

#include <vector>
#include <string>
#include <map>
#include <sstream>
#include <iomanip>

struct ProjectionState {
    float entropy;
    float attention;
    float pressure;
    float gravity;
    float replay_affinity;
    uint32_t active_fold;
    std::string expert_id;
    float arc_weight_sum;
};

class ProjectionExpert {
public:
    // ⟁ Project to CSS
    // Generates a string of CSS variable declarations representing the tensor state
    std::string project_to_css(const ProjectionState& state) {
        std::stringstream ss;
        ss << ":root {\n";
        ss << "  --entropy: " << std::fixed << std::setprecision(4) << state.entropy << ";\n";
        ss << "  --attention: " << state.attention << ";\n";
        ss << "  --pressure: " << state.pressure << ";\n";
        ss << "  --gravity: " << state.gravity << ";\n";
        ss << "  --replay-affinity: " << state.replay_affinity << ";\n";
        ss << "  --active-fold: " << state.active_fold << ";\n";
        ss << "  --expert-id: \"" << state.expert_id << "\";\n";
        ss << "  --coherence: " << (state.attention * (1.0f - state.entropy)) << ";\n";
        ss << "  --arc-weight-sum: " << state.arc_weight_sum << ";\n";
        ss << "}\n";
        return ss.str();
    }
    
    // ⟁ Project to JSON (for API transport)
    std::map<std::string, std::string> project_to_json_vars(const ProjectionState& state) {
        std::map<std::string, std::string> vars;
        vars["--entropy"] = std::to_string(state.entropy);
        vars["--attention"] = std::to_string(state.attention);
        vars["--pressure"] = std::to_string(state.pressure);
        vars["--gravity"] = std::to_string(state.gravity);
        vars["--replay-affinity"] = std::to_string(state.replay_affinity);
        vars["--active-fold"] = std::to_string(state.active_fold);
        vars["--expert-id"] = state.expert_id;
        vars["--arc-weight-sum"] = std::to_string(state.arc_weight_sum);
        return vars;
    }
};
