#pragma once
// unified_swarm_runtime.h  --  Micronaut V0 swarm runtime
// Phases: K'UHUL glyph phases + loop-control glyphs (K'AYAB/KUMK'U) + VM control (KAN/COLLAPSE)
// Folds:  functional fold taxonomy (COMPUTE/CONTROL/STATE/UI/META/DATA) —
//         distinct from the 6-face K-CUBE phase angles (Pop/Wo/Yax/Sek/Chen/Xul)

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <cstdint>

// nlohmann/json — pulled from kxc-v1.0.0/include or beside this header.
// If not found: drop single-header json.hpp into this directory.
#if __has_include("json.hpp")
#  include "json.hpp"
#elif __has_include(<nlohmann/json.hpp>)
#  include <nlohmann/json.hpp>
#elif __has_include("../../kxml-sk/semantic_kernel_cpp/include/nlohmann/json.hpp")
#  include "../../kxml-sk/semantic_kernel_cpp/include/nlohmann/json.hpp"
#else
#  error "nlohmann/json not found — add kxml-sk/semantic_kernel_cpp/include to include path"
#endif
using json = nlohmann::json;

namespace unified_swarm {

// ============================================================
// Phase — K'UHUL execution phases
// ============================================================
// First 6 map to K-CUBE face angles (Pop=0, Wo=π/3, …, Xul=5π/3).
// K_AYAB / KUMK_U are loop-entry / loop-exit control glyphs from KLSL.
// KAN     — conditional branch.
// COLLAPSE — VM halt / fold termination.
enum class Phase : uint8_t {
    POP      = 0,   // 0        — observe / Q-read
    WO       = 1,   // π/3      — weight / mask
    SEK      = 2,   // π        — compute / QKᵀ  (note: SEK maps to index 3 in K-CUBE; kept sequential here)
    CH_EN    = 3,   // 4π/3     — collect / V-gather
    YAX      = 4,   // 2π/3     — enumerate / K-read
    K_AYAB   = 5,   // loop start glyph
    KUMK_U   = 6,   // loop end glyph
    KAN      = 7,   // conditional branch
    COLLAPSE = 8,   // halt / done
};

std::string phase_to_string(Phase p);
Phase       string_to_phase(const std::string& s);

// ============================================================
// Fold — K-CUBE phase fold (matches K-CUBE face ↔ phase mapping)
// ============================================================
// Each micronaut belongs to exactly one fold based on its geometric role.
// Phase angle: Pop=0, Wo=π/3, Yax=2π/3, Sek=π, Chen=4π/3, Xul=5π/3
enum class Fold : uint8_t {
    Pop        = 0,   // 0     — observe / embed / raw input
    Wo         = 1,   // π/3   — weight / mask / flow control
    Yax        = 2,   // 2π/3  — enumerate / K-index / state
    Sek        = 3,   // π     — compute / tensor ops / QKᵀ
    Chen       = 4,   // 4π/3  — collect / V-gather / meta
    Xul        = 5,   // 5π/3  — output / entropy / UI surface
    UNASSIGNED = 0xFF,
};

std::string fold_to_string(Fold f);
Fold        string_to_fold(const std::string& s);

// ============================================================
// Value — stack / message value type
// ============================================================
struct Value {
    enum Type : uint8_t {
        NULL_TYPE = 0,
        INT       = 1,
        FLOAT     = 2,
        STRING    = 3,
        BYTES     = 4,
        ARRAY     = 5,
        OBJECT    = 6,
        AGENT_REF = 7,
        TENSOR    = 8,
    } type = NULL_TYPE;

    union { int64_t int_val = 0; double float_val; } data;
    std::string              str_val;
    std::vector<uint8_t>     bytes_val;
    std::vector<Value>       array_val;
    std::string              agent_ref;

    Value() = default;
    explicit Value(int64_t v)          : type(INT)    { data.int_val   = v; }
    explicit Value(double  v)          : type(FLOAT)  { data.float_val = v; }
    explicit Value(const std::string& v) : type(STRING), str_val(v) {}

    std::string to_string() const;
};

// ============================================================
// CSSBinding — per-micronaut CSS custom-property binding
// ============================================================
// Exposes micronaut state to the PRIMEOS WebView layer via
// CSS variables injected into the shadow DOM (🤖[id="..."] {...}).
struct CSSBinding {
    std::string                           selector;
    std::unordered_map<std::string,std::string> css_variables;

    explicit CSSBinding(const std::string& agent_id) : selector(agent_id) {}
    void        set_variable(const std::string& name, const std::string& value)
                { css_variables[name] = value; }
    std::string generate_css() const;
};

// ============================================================
// Forward declarations
// ============================================================
class Micronaut;
class SCXQ2VM;

// ============================================================
// SCXQ2VM — minimal SCXQ2 bytecode interpreter
// ============================================================
// Executes a small subset of SCXQ2 opcodes inside a Micronaut context.
// Full SCXQ2 execution lives in scx_runtime.exe / scx2_runtime_smoke.exe.
class SCXQ2VM {
public:
    SCXQ2VM(const std::vector<uint8_t>& bytecode, Micronaut* agent)
        : bytecode(bytecode), agent(agent) {}

    void  run();
    void  step();
    Value get_result();

private:
    const std::vector<uint8_t>&  bytecode;
    Micronaut*                   agent;
    std::vector<Value>           stack;
    size_t                       pc    = 0;
    Phase                        phase = Phase::POP;

    void execute_opcode(uint8_t opcode);
};

// ============================================================
// Micronaut — single autonomous agent
// ============================================================
class Micronaut {
public:
    Micronaut(const std::string& agent_id, const json& config);

    // Identity
    const std::string& get_id()       const { return id; }
    const std::string& get_type()     const { return type; }
    Fold               get_fold()     const { return fold; }
    int                get_port()     const { return port; }
    const std::string& get_endpoint() const { return endpoint; }
    const std::string& get_role()     const { return swarm_role; }
    int                get_priority() const { return priority; }
    Phase              get_phase()    const { return current_phase; }
    double             get_load()     const { return load; }
    bool               is_running()   const { return running; }

    void set_load(double l)  { load = l; update_css_state(); }
    void set_phase(Phase p)  { current_phase = p; update_css_state(); }

    // Stack
    void  push_stack(const Value& v) { stack.push_back(v); }
    Value pop_stack();

    // CSS
    CSSBinding& get_css_binding() { return css_binding; }

    // Execution
    void  step();
    Value execute_bytecode(const std::vector<uint8_t>& bytecode);

    // Capabilities / experts
    const std::vector<std::string>& get_capabilities() const { return capabilities; }
    const std::vector<std::string>& get_experts()      const { return experts; }

private:
    std::string              id;
    std::string              type;
    Fold                     fold;
    int                      port        = 0;
    std::string              endpoint;
    std::string              swarm_role;
    int                      priority    = 0;

    Phase                    current_phase = Phase::POP;
    std::vector<Value>       stack;
    double                   load         = 0.0;
    bool                     running      = true;
    size_t                   pc           = 0;

    CSSBinding               css_binding;
    std::vector<std::string> capabilities;
    std::vector<std::string> experts;

    void update_css_state();
};

// ============================================================
// SwarmConsciousness — collective state of all agents
// ============================================================
class SwarmConsciousness {
public:
    SwarmConsciousness();

    void register_agent(const json& config);

    std::shared_ptr<Micronaut>              get_agent(const std::string& id);
    std::vector<std::shared_ptr<Micronaut>> get_agents_by_fold(Fold fold);

    void broadcast_to_fold(Fold fold, const json& message);
    void send_to_agent(const std::string& agent_id, const json& message);

    // SH-coefficient optical wave → CSS animation + tick sequence
    void optical_wave(const std::vector<float>& sh_coefficients, uint32_t frames);

    void tick();
    void run_for_ticks(uint32_t count);

    json   get_swarm_state() const;
    double get_coherence()   const { return coherence; }
    double get_entropy()     const { return entropy; }

    // CSS injection callback — set by the host (PRIMEOS WebView bridge)
    std::function<void(const std::string&)> css_injector;

private:
    std::unordered_map<std::string, std::shared_ptr<Micronaut>> agents;
    std::unordered_map<uint8_t, std::vector<std::string>>       fold_members;

    uint64_t tick_count = 0;
    double   coherence  = 0.94;
    double   entropy    = 0.06;

    void recalculate_coherence();
    void update_css_root();
};

// ============================================================
// UnifiedRuntime — top-level host
// ============================================================
class UnifiedRuntime {
public:
    UnifiedRuntime();

    void load_model(const std::string& model_dir);
    void load_model_from_json(const json& model_spec);

    std::string inference(const std::string& prompt, uint32_t max_tokens = 256);
    json        inference_json(const json& request);

    void optical_compute(const std::vector<float>& sh_coefficients, uint32_t frames);
    void execute_bytecode_on_agent(const std::string& agent_id,
                                   const std::vector<uint8_t>& bytecode);

    json get_system_status() const;
    void print_swarm_status() const;

    SwarmConsciousness& get_swarm() { return swarm; }

private:
    SwarmConsciousness swarm;
    json               loaded_model_config;
};

// ============================================================
// Agent config loader — all 35 hardcoded micronauts
// ============================================================
std::vector<json> load_embedded_agent_configs();

} // namespace unified_swarm
