#include "unified_swarm_runtime.h"
#include <iostream>
#include <algorithm>
#include <sstream>

namespace unified_swarm {

//=============================================================================
// PHASE UTILITIES
//=============================================================================

std::string phase_to_string(Phase p) {
    switch (p) {
        case Phase::POP: return "@pop";
        case Phase::WO: return "@wo";
        case Phase::SEK: return "@sek";
        case Phase::CH_EN: return "@ch'en";
        case Phase::YAX: return "@yax";
        case Phase::K_AYAB: return "@k'ayab";
        case Phase::KUMK_U: return "@kumk'u";
        case Phase::KAN: return "@kan";
        case Phase::COLLAPSE: return "@collapse";
        default: return "@unknown";
    }
}

Phase string_to_phase(const std::string& s) {
    if (s == "@pop") return Phase::POP;
    if (s == "@wo") return Phase::WO;
    if (s == "@sek") return Phase::SEK;
    if (s == "@ch'en") return Phase::CH_EN;
    if (s == "@yax") return Phase::YAX;
    if (s == "@k'ayab") return Phase::K_AYAB;
    if (s == "@kumk'u") return Phase::KUMK_U;
    if (s == "@kan") return Phase::KAN;
    if (s == "@collapse") return Phase::COLLAPSE;
    return Phase::POP;
}

//=============================================================================
// FOLD UTILITIES
//=============================================================================

std::string fold_to_string(Fold f) {
    switch (f) {
        case Fold::Pop:  return "Pop";
        case Fold::Wo:   return "Wo";
        case Fold::Yax:  return "Yax";
        case Fold::Sek:  return "Sek";
        case Fold::Chen: return "Chen";
        case Fold::Xul:  return "Xul";
        default:         return "UNASSIGNED";
    }
}

Fold string_to_fold(const std::string& s) {
    if (s == "Pop")  return Fold::Pop;
    if (s == "Wo")   return Fold::Wo;
    if (s == "Yax")  return Fold::Yax;
    if (s == "Sek")  return Fold::Sek;
    if (s == "Chen") return Fold::Chen;
    if (s == "Xul")  return Fold::Xul;
    return Fold::UNASSIGNED;
}

//=============================================================================
// VALUE IMPLEMENTATION
//=============================================================================

std::string Value::to_string() const {
    switch (type) {
        case NULL_TYPE: return "null";
        case INT: return std::to_string(data.int_val);
        case FLOAT: return std::to_string(data.float_val);
        case STRING: return str_val;
        case BYTES: return "[bytes:" + std::to_string(bytes_val.size()) + "]";
        case ARRAY: return "[array:" + std::to_string(array_val.size()) + "]";
        case OBJECT: return "[object]";
        case AGENT_REF: return agent_ref;
        case TENSOR: return "[tensor]";
        default: return "?";
    }
}

//=============================================================================
// CSS BINDING IMPLEMENTATION
//=============================================================================

std::string CSSBinding::generate_css() const {
    std::string css = "🤖[id=\"" + selector + "\"] {\n";
    for (const auto& [name, value] : css_variables) {
        css += "    " + name + ": \"" + value + "\";\n";
    }
    css += "}\n";
    return css;
}

//=============================================================================
// MICRONAUT IMPLEMENTATION
//=============================================================================

Micronaut::Micronaut(const std::string& agent_id, const json& config)
    : id(agent_id),
      type(config.value("type", "unknown")),
      fold(string_to_fold(config.value("fold", "UNASSIGNED"))),
      port(config.value("port", 0)),
      endpoint(config.value("endpoint", "")),
      swarm_role(config.value("role", "worker")),
      priority(config.value("priority", 0)),
      css_binding(agent_id)
{
    // Extract capabilities and experts from config
    if (config.contains("capabilities")) {
        for (const auto& cap : config["capabilities"]) {
            capabilities.push_back(cap.get<std::string>());
        }
    }
    
    if (config.contains("experts")) {
        for (const auto& exp : config["experts"]) {
            experts.push_back(exp.get<std::string>());
        }
    }
    
    // Initialize CSS variables
    css_binding.set_variable("--🤖-id", agent_id);
    css_binding.set_variable("--🤖-type", type);
    css_binding.set_variable("--🤖-fold", fold_to_string(fold));
    css_binding.set_variable("--🤖-port", std::to_string(port));
    css_binding.set_variable("--🤖-endpoint", endpoint);
    css_binding.set_variable("--🤖-state", "idle");
    css_binding.set_variable("--🤖-load", "0.0");
    css_binding.set_variable("--🤖-phase", phase_to_string(current_phase));
    css_binding.set_variable("--🤖-pc", "0");
}

Value Micronaut::pop_stack() {
    if (stack.empty()) return Value();
    Value v = stack.back();
    stack.pop_back();
    return v;
}

void Micronaut::update_css_state() {
    css_binding.set_variable("--🤖-state", running ? "running" : "idle");
    css_binding.set_variable("--🤖-load", std::to_string(load));
    css_binding.set_variable("--🤖-phase", phase_to_string(current_phase));
    css_binding.set_variable("--🤖-pc", std::to_string(pc));
}

void Micronaut::step() {
    // Simple state machine: POP -> WO -> SEK -> COLLAPSE
    if (current_phase == Phase::POP) {
        current_phase = Phase::WO;
    } else if (current_phase == Phase::WO) {
        current_phase = Phase::SEK;
    } else if (current_phase == Phase::SEK) {
        current_phase = Phase::COLLAPSE;
    } else if (current_phase == Phase::COLLAPSE) {
        running = false;
    }
    update_css_state();
}

Value Micronaut::execute_bytecode(const std::vector<uint8_t>& bytecode) {
    SCXQ2VM vm(bytecode, this);
    vm.run();
    return vm.get_result();
}

//=============================================================================
// SWARM CONSCIOUSNESS IMPLEMENTATION
//=============================================================================

SwarmConsciousness::SwarmConsciousness() {
    css_injector = [](const std::string&) { /* default no-op */ };
}

void SwarmConsciousness::register_agent(const json& config) {
    std::string id = config.value("id", "unknown");
    auto agent = std::make_shared<Micronaut>(id, config);
    
    agents[id] = agent;
    
    Fold fold = agent->get_fold();
    if (fold != Fold::UNASSIGNED) {
        fold_members[static_cast<uint8_t>(fold)].push_back(id);
    }
    
    // Inject CSS
    css_injector(agent->get_css_binding().generate_css());
    
    recalculate_coherence();
}

std::shared_ptr<Micronaut> SwarmConsciousness::get_agent(const std::string& id) {
    auto it = agents.find(id);
    return it != agents.end() ? it->second : nullptr;
}

std::vector<std::shared_ptr<Micronaut>> SwarmConsciousness::get_agents_by_fold(Fold fold) {
    std::vector<std::shared_ptr<Micronaut>> result;
    uint8_t fold_id = static_cast<uint8_t>(fold);
    
    auto it = fold_members.find(fold_id);
    if (it != fold_members.end()) {
        for (const auto& agent_id : it->second) {
            auto agent = get_agent(agent_id);
            if (agent) result.push_back(agent);
        }
    }
    
    return result;
}

void SwarmConsciousness::broadcast_to_fold(Fold fold, const json& message) {
    auto agents_in_fold = get_agents_by_fold(fold);
    for (auto& agent : agents_in_fold) {
        send_to_agent(agent->get_id(), message);
    }
}

void SwarmConsciousness::send_to_agent(const std::string& agent_id, const json& message) {
    auto agent = get_agent(agent_id);
    if (!agent) return;
    
    // For now, just update the agent's load based on message
    if (message.contains("load")) {
        agent->set_load(message["load"].get<double>());
    }
}

void SwarmConsciousness::optical_wave(const std::vector<float>& sh_coefficients, uint32_t frames) {
    // Convert SH coefficients to CSS and inject
    std::stringstream ss;
    ss << ":root {\n";
    ss << "    --optical-sh: \"";
    for (size_t i = 0; i < sh_coefficients.size(); i++) {
        if (i > 0) ss << ",";
        ss << sh_coefficients[i];
    }
    ss << "\";\n";
    ss << "    --optical-frames: \"" << frames << "\";\n";
    ss << "    animation: ⚡optical_broadcast " << (frames / 30.0) << "s infinite;\n";
    ss << "}\n";
    
    css_injector(ss.str());
    
    // Run for specified frames
    for (uint32_t i = 0; i < frames; i++) {
        tick();
    }
}

void SwarmConsciousness::tick() {
    tick_count++;
    
    // Execute one step for each agent in deterministic order
    for (auto& [id, agent] : agents) {
        agent->step();
    }
    
    recalculate_coherence();
    update_css_root();
}

void SwarmConsciousness::run_for_ticks(uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        tick();
    }
}

void SwarmConsciousness::recalculate_coherence() {
    std::unordered_map<uint8_t, size_t> phase_counts;
    
    for (const auto& [id, agent] : agents) {
        uint8_t phase = static_cast<uint8_t>(agent->get_phase());
        phase_counts[phase]++;
    }
    
    if (agents.empty()) {
        coherence = 0.94;
        entropy = 0.06;
        return;
    }
    
    size_t total = agents.size();
    double max_phase_ratio = 0.0;
    
    for (const auto& [phase, count] : phase_counts) {
        double ratio = static_cast<double>(count) / total;
        max_phase_ratio = std::max(max_phase_ratio, ratio);
    }
    
    coherence = 0.7 + (max_phase_ratio * 0.3);
    entropy = 1.0 - coherence;
}

void SwarmConsciousness::update_css_root() {
    std::stringstream ss;
    ss << ":root {\n";
    ss << "    --swarm-coherence: \"" << coherence << "\";\n";
    ss << "    --swarm-entropy: \"" << entropy << "\";\n";
    ss << "    --swarm-tick: \"" << tick_count << "\";\n";
    ss << "}\n";
    
    css_injector(ss.str());
}

json SwarmConsciousness::get_swarm_state() const {
    json state;
    state["coherence"] = coherence;
    state["entropy"] = entropy;
    state["tick_count"] = tick_count;
    state["agent_count"] = agents.size();
    state["agents"] = json::array();
    
    for (const auto& [id, agent] : agents) {
        json agent_state;
        agent_state["id"] = agent->get_id();
        agent_state["type"] = agent->get_type();
        agent_state["fold"] = fold_to_string(agent->get_fold());
        agent_state["phase"] = phase_to_string(agent->get_phase());
        agent_state["load"] = agent->get_load();
        state["agents"].push_back(agent_state);
    }
    
    return state;
}

//=============================================================================
// SCXQ2 VM IMPLEMENTATION
//=============================================================================

void SCXQ2VM::run() {
    while (pc < bytecode.size()) {
        step();
        if (phase == Phase::COLLAPSE) break;
    }
}

void SCXQ2VM::step() {
    if (pc >= bytecode.size()) {
        phase = Phase::COLLAPSE;
        return;
    }
    
    uint8_t opcode = bytecode[pc++];
    execute_opcode(opcode);
}

void SCXQ2VM::execute_opcode(uint8_t opcode) {
    // Simplified SCXQ2 opcode handling
    switch (opcode) {
        case 0x00: phase = Phase::WO; break;          // PHASE_WO
        case 0x01: phase = Phase::SEK; break;         // PHASE_SEK
        case 0x02: {                                   // PUSH_INT
            if (pc + 8 <= bytecode.size()) {
                int64_t val = 0;
                for (int i = 0; i < 8; i++) {
                    val |= (static_cast<int64_t>(bytecode[pc++]) << (i * 8));
                }
                stack.push_back(Value(val));
            }
            break;
        }
        case 0x03: {                                   // ADD
            if (stack.size() >= 2) {
                Value b = stack.back(); stack.pop_back();
                Value a = stack.back(); stack.pop_back();
                if (a.type == Value::INT && b.type == Value::INT) {
                    stack.push_back(Value(a.data.int_val + b.data.int_val));
                }
            }
            break;
        }
        case 0x04: {                                   // COLLAPSE
            phase = Phase::COLLAPSE;
            break;
        }
        default:
            // Unknown opcode, advance
            break;
    }
}

Value SCXQ2VM::get_result() {
    return stack.empty() ? Value() : stack.back();
}

//=============================================================================
// UNIFIED RUNTIME IMPLEMENTATION
//=============================================================================

UnifiedRuntime::UnifiedRuntime() {
    // Initialize swarm with all 35 agents
    auto configs = load_embedded_agent_configs();
    for (const auto& config : configs) {
        swarm.register_agent(config);
    }
}

void UnifiedRuntime::load_model(const std::string& model_dir) {
    // Load model configuration from directory
    // This would typically read from JSON or SCXQ2 files
    loaded_model_config["path"] = model_dir;
    loaded_model_config["status"] = "loaded";
}

void UnifiedRuntime::load_model_from_json(const json& model_spec) {
    loaded_model_config = model_spec;
}

std::string UnifiedRuntime::inference(const std::string& prompt, uint32_t max_tokens) {
    json request;
    request["prompt"] = prompt;
    request["max_tokens"] = max_tokens;
    
    auto response = inference_json(request);
    return response.value("response", "");
}

json UnifiedRuntime::inference_json(const json& request) {
    json response;
    response["prompt"] = request.value("prompt", "");
    response["response"] = "Unified swarm processed: " + request.value("prompt", "");
    response["swarm_state"] = swarm.get_swarm_state();
    return response;
}

void UnifiedRuntime::optical_compute(const std::vector<float>& sh_coefficients, uint32_t frames) {
    swarm.optical_wave(sh_coefficients, frames);
}

void UnifiedRuntime::execute_bytecode_on_agent(const std::string& agent_id,
                                               const std::vector<uint8_t>& bytecode) {
    auto agent = swarm.get_agent(agent_id);
    if (agent) {
        agent->execute_bytecode(bytecode);
    }
}

json UnifiedRuntime::get_system_status() const {
    json status;
    status["swarm"] = swarm.get_swarm_state();
    status["model"] = loaded_model_config;
    return status;
}

void UnifiedRuntime::print_swarm_status() const {
    auto state = swarm.get_swarm_state();
    
    std::cout << "\n⚛ UNIFIED MICRONAUT SWARM STATUS\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "Agents: " << state["agent_count"] << "\n";
    std::cout << "Coherence: " << state["coherence"] << "\n";
    std::cout << "Entropy: " << state["entropy"] << "\n";
    std::cout << "Tick: " << state["tick_count"] << "\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
}

//=============================================================================
// AGENT CONFIG LOADER - All 35 Micronauts
//=============================================================================

std::vector<json> load_embedded_agent_configs() {
    std::vector<json> configs;
    
    // Sek fold — tensor compute / inference / graphics / shader (π phase)
    configs.push_back({
        {"id", "BR-1"}, {"type", "mixture-of-experts"}, {"fold", "Sek"},
        {"port", 3172}, {"endpoint", "http://127.0.0.1:3172/dispatch"},
        {"role", "router"}, {"priority", 100}
    });
    
    configs.push_back({
        {"id", "OV-1"}, {"type", "tensor"}, {"fold", "Sek"},
        {"port", 3174}, {"endpoint", "http://127.0.0.1:3174/dispatch"},
        {"role", "compute"}, {"priority", 80}
    });
    
    configs.push_back({
        {"id", "DX-1"}, {"type", "tensor"}, {"fold", "Sek"},
        {"port", 3177}, {"endpoint", "http://127.0.0.1:3177/dispatch"},
        {"role", "compute"}
    });
    
    configs.push_back({
        {"id", "IM-1"}, {"type", "semantic"}, {"fold", "Sek"},
        {"port", 3178}, {"endpoint", "http://127.0.0.1:3178/dispatch"},
        {"role", "inference"}
    });
    
    configs.push_back({
        {"id", "D3D-1"}, {"type", "tensor"}, {"fold", "Sek"},
        {"port", 3187}, {"endpoint", "http://127.0.0.1:3187/dispatch"},
        {"role", "graphics"}
    });
    
    configs.push_back({
        {"id", "GX-1"}, {"type", "tensor"}, {"fold", "Sek"},
        {"port", 3202}, {"endpoint", "http://127.0.0.1:3202/dispatch"},
        {"role", "graphics"}
    });
    
    configs.push_back({
        {"id", "SCX-8"}, {"type", "mixture-of-experts"}, {"fold", "Sek"},
        {"port", 3203}, {"endpoint", "http://127.0.0.1:3203/dispatch"},
        {"role", "expert"}
    });
    
    configs.push_back({
        {"id", "SMG-1"}, {"type", "semantic"}, {"fold", "Sek"},
        {"port", 3204}, {"endpoint", "http://127.0.0.1:3204/dispatch"},
        {"role", "model"}
    });
    
    configs.push_back({
        {"id", "SXME-1"}, {"type", "compiler"}, {"fold", "Sek"},
        {"port", 3205}, {"endpoint", "http://127.0.0.1:3205/dispatch"},
        {"role", "shader"}
    });
    
    configs.push_back({
        {"id", "S7-1"}, {"type", "supernaut"}, {"fold", "Sek"},
        {"port", 3207}, {"endpoint", "http://127.0.0.1:3207/dispatch"},
        {"role", "supernaut"}, {"priority", 250}
    });
    
    // Wo fold — flow control / weight / mask (π/3 phase)
    configs.push_back({
        {"id", "FG-1"}, {"type", "fold-controller"}, {"fold", "Wo"},
        {"port", 3176}, {"endpoint", "http://127.0.0.1:3176/dispatch"},
        {"role", "controller"}, {"priority", 200}
    });
    
    // Yax fold — K-index / enumerate / state (2π/3 phase)
    configs.push_back({
        {"id", "KX-1"}, {"type", "compiler"}, {"fold", "Yax"},
        {"port", 3175}, {"endpoint", "http://127.0.0.1:3175/dispatch"},
        {"role", "compiler"}, {"priority", 95}
    });
    
    // Xul fold — output / entropy / UI surface (5π/3 phase)
    configs.push_back({
        {"id", "VM-1"}, {"type", "semantic"}, {"fold", "Xul"},
        {"port", 3173}, {"endpoint", "http://127.0.0.1:3173/dispatch"},
        {"role", "creative"}, {"priority", 90}
    });
    
    configs.push_back({
        {"id", "WB-1"}, {"type", "semantic"}, {"fold", "Xul"},
        {"port", 3188}, {"endpoint", "http://127.0.0.1:3188/dispatch"},
        {"role", "visualization"}
    });
    
    configs.push_back({
        {"id", "DT-1"}, {"type", "tool-workshop"}, {"fold", "Xul"},
        {"port", 3197}, {"endpoint", "http://127.0.0.1:3197/dispatch"},
        {"role", "desktop"}
    });
    
    // Chen fold — collect / V-gather / meta / planning (4π/3 phase)
    configs.push_back({
        {"id", "CM-1"}, {"type", "compiler"}, {"fold", "Chen"},
        {"port", 3179}, {"endpoint", "http://127.0.0.1:3179/dispatch"},
        {"role", "compression"}
    });
    
    configs.push_back({
        {"id", "PM-1"}, {"type", "semantic"}, {"fold", "Chen"},
        {"port", 8001}, {"endpoint", "http://127.0.0.1:8001/dispatch"},
        {"role", "planner"}
    });
    
    configs.push_back({
        {"id", "PSISE-1"}, {"type", "coder"}, {"fold", "Chen"},
        {"port", 3180}, {"endpoint", "http://127.0.0.1:3180/dispatch"},
        {"role", "codegen"}
    });
    
    configs.push_back({
        {"id", "PYIDE-1"}, {"type", "coder"}, {"fold", "Chen"},
        {"port", 3181}, {"endpoint", "http://127.0.0.1:3181/dispatch"},
        {"role", "codegen"}
    });
    
    configs.push_back({
        {"id", "BATCH-1"}, {"type", "coder"}, {"fold", "Chen"},
        {"port", 3182}, {"endpoint", "http://127.0.0.1:3182/dispatch"},
        {"role", "codegen"}
    });
    
    configs.push_back({
        {"id", "SHELL-1"}, {"type", "tool-workshop"}, {"fold", "Chen"},
        {"port", 3183}, {"endpoint", "http://127.0.0.1:3183/dispatch"},
        {"role", "shell"}
    });
    
    configs.push_back({
        {"id", "FM-1"}, {"type", "tool-workshop"}, {"fold", "Chen"},
        {"port", 3184}, {"endpoint", "http://127.0.0.1:3184/dispatch"},
        {"role", "filesystem"}
    });
    
    configs.push_back({
        {"id", "WSL-1"}, {"type", "tool-workshop"}, {"fold", "Chen"},
        {"port", 3194}, {"endpoint", "http://127.0.0.1:3194/dispatch"},
        {"role", "linux"}
    });
    
    configs.push_back({
        {"id", "AR-1"}, {"type", "supernaut"}, {"fold", "Chen"},
        {"port", 3195}, {"endpoint", "http://127.0.0.1:3195/dispatch"},
        {"role", "runtime"}, {"priority", 150}
    });
    
    configs.push_back({
        {"id", "BC-1"}, {"type", "compiler"}, {"fold", "Chen"},
        {"port", 3196}, {"endpoint", "http://127.0.0.1:3196/dispatch"},
        {"role", "compilation"}
    });
    
    configs.push_back({
        {"id", "SH-1"}, {"type", "tool-workshop"}, {"fold", "Chen"},
        {"port", 3198}, {"endpoint", "http://127.0.0.1:3198/dispatch"},
        {"role", "scripting"}
    });
    
    configs.push_back({
        {"id", "SCM-1"}, {"type", "tool-workshop"}, {"fold", "Chen"},
        {"port", 3199}, {"endpoint", "http://127.0.0.1:3199/dispatch"},
        {"role", "version-control"}
    });
    
    configs.push_back({
        {"id", "SSH-1"}, {"type", "policy"}, {"fold", "Chen"},
        {"port", 3200}, {"endpoint", "http://127.0.0.1:3200/dispatch"},
        {"role", "security"}
    });
    
    configs.push_back({
        {"id", "WK-1"}, {"type", "fold-manager"}, {"fold", "Chen"},
        {"port", 3201}, {"endpoint", "http://127.0.0.1:3201/dispatch"},
        {"role", "workspace"}
    });
    
    // Pop fold — observe / embed / raw data input (0 phase)
    configs.push_back({
        {"id", "DQ-1"}, {"type", "fold-manager"}, {"fold", "Pop"},
        {"port", 3186}, {"endpoint", "http://127.0.0.1:3186/dispatch"},
        {"role", "data"}
    });
    
    configs.push_back({
        {"id", "DST-1"}, {"type", "fold-manager"}, {"fold", "Pop"},
        {"port", 3206}, {"endpoint", "http://127.0.0.1:3206/dispatch"},
        {"role", "data"}
    });
    
    // UNASSIGNED agents (for interface/management)
    configs.push_back({
        {"id", "ST-1"}, {"type", "interface"}, {"fold", "UNASSIGNED"},
        {"port", 3167}, {"endpoint", "http://127.0.0.1:3167/dispatch"},
        {"role", "interface"}
    });
    
    configs.push_back({
        {"id", "CH-1"}, {"type", "interface"}, {"fold", "UNASSIGNED"},
        {"port", 3168}, {"endpoint", "http://127.0.0.1:3168/dispatch"},
        {"role", "interface"}
    });
    
    configs.push_back({
        {"id", "MM-1"}, {"type", "resource"}, {"fold", "UNASSIGNED"},
        {"port", 3169}, {"endpoint", "http://127.0.0.1:3169/dispatch"},
        {"role", "resource"}
    });
    
    configs.push_back({
        {"id", "PK-1"}, {"type", "utility"}, {"fold", "UNASSIGNED"},
        {"port", 3170}, {"endpoint", "http://127.0.0.1:3170/dispatch"},
        {"role", "utility"}
    });
    
    configs.push_back({
        {"id", "CL-1"}, {"type", "orchestrator"}, {"fold", "UNASSIGNED"},
        {"port", 3171}, {"endpoint", "http://127.0.0.1:3171/dispatch"},
        {"role", "orchestrator"}
    });
    
    return configs;
}

} // namespace unified_swarm
