// ir_types.h - Complete K'UHUL IR Type System v0.1
// Frozen Reference Implementation for Layer 6 Compiler
// Date: 2026-07-12

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <memory>
#include <optional>
#include <span>
#include <concepts>
#include <queue>
#include <iostream>

namespace KuhulIR {

// ===== Forward Declarations =====
struct IRNode;
struct IRGraph;
struct IRFunction;
struct IRBlock;
struct IREdge;
struct IRProgram;
struct DMLOperatorDesc;

// ===== Value Types =====
enum class ValueType : uint8_t {
    Int,
    Float,
    Bool,
    String,
    List,
    Map,
    Tensor,
    NodeRef,
    CallRef,
    Void,
    Error
};

// ===== Base Value Variant =====
using Value = std::variant<
    int64_t,                      // Int
    double,                       // Float
    bool,                         // Bool
    std::string,                  // String
    std::vector<Value>,          // List
    std::unordered_map<std::string, Value>, // Map
    IRNode*,                      // NodeRef
    IRFunction*,                  // CallRef
    std::monostate                // Void/Error
>;

// ===== Tensor Type =====
struct TensorType {
    std::vector<size_t> shape;    // Dimensions
    ValueType dtype;              // Element type
    size_t size;                  // Total elements
    bool is_dynamic;              // Dynamic shape?
    std::optional<size_t> stride; // Memory stride
    
    // Convenience methods
    size_t total_elements() const {
        size_t total = 1;
        for (auto dim : shape) total *= dim;
        return total;
    }
    
    size_t bytes() const {
        size_t elem_size = (dtype == ValueType::Float) ? 4 : 8;
        return total_elements() * elem_size;
    }
};

// ===== Metadata =====
struct SourceLocation {
    std::string file;
    size_t line;
    size_t column;
    std::string function;
    
    std::string to_string() const {
        return file + ":" + std::to_string(line) + ":" + std::to_string(column);
    }
};

// ===== IR Node (Core computation unit) =====
struct IRNode {
    // Identity
    std::string id;                              // Glyph name (e.g., "@conv")
    std::string op_code;                         // Canonical op (e.g., "conv2d")
    size_t graph_id;                             // Unique node ID
    SourceLocation source_loc;                   // Debug info
    
    // Data
    std::unordered_map<std::string, Value> attrs; // Modifiers/constants
    std::vector<IRNode*> inputs;                 // Input edges
    std::vector<IRNode*> outputs;                // Output edges (reverse)
    ValueType output_type;                       // Node output type
    TensorType output_tensor_shape;              // Output tensor shape
    
    // Flags
    bool is_constant = false;                    // Constant folding
    bool is_control_flow = false;                // If/loop
    bool is_side_effecting = false;              // Has side effects
    bool is_pure = true;                         // No side effects
    bool is_fused = false;                       // Part of fusion group
    
    // Execution
    mutable void* runtime_data = nullptr;        // Backend-specific data
    mutable size_t runtime_data_size = 0;
    
    // Methods
    bool has_attribute(const std::string& key) const {
        return attrs.find(key) != attrs.end();
    }
    
    template<typename T>
    std::optional<T> get_attribute(const std::string& key) const {
        auto it = attrs.find(key);
        if (it == attrs.end()) return std::nullopt;
        if (auto* val = std::get_if<T>(&it->second)) {
            return *val;
        }
        return std::nullopt;
    }
    
    void set_attribute(const std::string& key, const Value& value) {
        attrs[key] = value;
    }
    
    size_t input_count() const { return inputs.size(); }
    size_t output_count() const { return outputs.size(); }
};

// ===== Edge Types =====
enum class EdgeType : uint8_t {
    Forward,    // → standard dataflow
    Downward,   // ↧ subgraph input
    Upward,     // ↥ subgraph output
    Loop,       // ↺ iteration
    Bind,       // ⟜ parameter binding
    Control,    // ⚡ control dependency
    Data,       // 📊 data dependency
    Tensor      // 🧮 tensor flow
};

// ===== IR Edge =====
struct IREdge {
    IRNode* source = nullptr;
    IRNode* target = nullptr;
    EdgeType type = EdgeType::Forward;
    size_t source_output_index = 0;
    size_t target_input_index = 0;
    std::optional<size_t> weight = std::nullopt;  // Edge weight for optimization
    
    // Metadata
    std::string name;
    SourceLocation source_loc;
};

// ===== IR Block (Graph scope) =====
struct IRBlock {
    std::string name;
    IRNode* entry_node = nullptr;                 // Block root
    std::vector<std::unique_ptr<IRNode>> nodes;   // Owned nodes
    std::vector<IREdge> edges;                    // Internal edges
    std::vector<IRBlock*> subblocks;              // Nested blocks
    IRBlock* parent = nullptr;                    // Parent block
    
    // Execution order
    std::vector<IRNode*> execution_order;         // Topological order
    
    // Methods
    void add_node(std::unique_ptr<IRNode> node) {
        node->graph_id = nodes.size();
        nodes.push_back(std::move(node));
    }
    
    IRNode* get_node(size_t id) const {
        if (id < nodes.size()) return nodes[id].get();
        return nullptr;
    }
    
    void add_edge(const IREdge& edge) {
        edges.push_back(edge);
        if (edge.source && edge.target) {
            edge.source->outputs.push_back(edge.target);
            edge.target->inputs.push_back(edge.source);
        }
    }
    
    size_t node_count() const { return nodes.size(); }
};

// ===== Function Definition =====
struct IRFunction {
    std::string name;
    std::vector<std::string> params;              // Parameter names
    std::vector<ValueType> param_types;           // Parameter types
    IRBlock body;                                 // Function body
    IRNode* return_node = nullptr;                // Return node
    
    // Attributes
    std::unordered_map<std::string, Value> attrs;
    bool is_exported = false;
    bool is_inline = false;
    
    // Methods
    void add_parameter(const std::string& name, ValueType type) {
        params.push_back(name);
        param_types.push_back(type);
    }
};

// ===== IR Graph (Represents full computation) =====
struct IRGraph {
    std::string name;
    std::vector<IRNode*> nodes;                   // All nodes (non-owning)
    std::vector<IREdge> edges;                    // All edges
    IRNode* entry = nullptr;                      // Entry node
    IRNode* output = nullptr;                     // Output node
    
    size_t node_count() const { return nodes.size(); }
    size_t edge_count() const { return edges.size(); }
};

// ===== Program (Complete compilation unit) =====
struct IRProgram {
    std::string name;
    std::string version = "0.1.0";
    
    std::vector<IRFunction> functions;
    std::vector<IRBlock> top_level_blocks;
    std::vector<IREdge> top_level_edges;
    
    // Global state
    std::unordered_map<std::string, Value> globals;
    std::unordered_map<std::string, ValueType> global_types;
    
    // Methods
    IRFunction* get_function(const std::string& name) {
        for (auto& func : functions) {
            if (func.name == name) return &func;
        }
        return nullptr;
    }
    
    void add_function(IRFunction&& func) {
        functions.push_back(std::move(func));
    }
};

// ===== DirectML Operator Extensions =====
struct DMLOperatorDesc {
    enum class Type {
        Conv2D,
        MatMul,
        ReLU,
        Softmax,
        Add,
        Multiply,
        Constant,
        If,
        Loop,
        BatchNormalization,
        Pooling,
        Concat,
        Split,
        Reshape,
        Transpose,
        ReduceSum,
        ReduceMean,
        ElementWise,
        Gemm,
        Activation,
        Custom
    } type;
    
    // Common fields
    std::string name;
    std::vector<size_t> input_tensor_shapes;
    std::vector<size_t> output_tensor_shapes;
    std::unordered_map<std::string, Value> parameters;
    
    // Specific operator fields
    struct Conv2DParams {
        std::vector<size_t> strides;
        std::vector<size_t> dilations;
        std::vector<size_t> pads;
        size_t groups = 1;
        bool use_bias = false;
    };
    
    struct MatMulParams {
        bool transpose_a = false;
        bool transpose_b = false;
        double alpha = 1.0;
    };
    
    struct PoolingParams {
        std::vector<size_t> kernel_size;
        std::vector<size_t> strides;
        std::vector<size_t> pads;
        enum class Mode { Max, Average } mode;
    };
    
    struct ActivationParams {
        enum class Type { ReLU, Sigmoid, Tanh, LeakyReLU, ELU, GELU } type;
        double alpha = 0.0;
    };
    
    using SpecificParams = std::variant<
        Conv2DParams,
        MatMulParams,
        PoolingParams,
        ActivationParams,
        std::monostate
    >;
    
    SpecificParams specific_params;
    
    bool has_specific_params() const {
        return !std::holds_alternative<std::monostate>(specific_params);
    }
    
    template<typename T>
    const T* get_specific_params() const {
        return std::get_if<T>(&specific_params);
    }
};

// ===== Optimization Metadata =====
struct OptimizationInfo {
    bool constant_folded = false;
    bool dead_code_eliminated = false;
    bool fused = false;
    std::string fusion_group;
    size_t estimated_flops = 0;
    size_t memory_usage = 0;
    
    double pi_phase = 0.0;
    double reward = 0.0;
    double entropy = 0.0;
};

// ===== Execution Context =====
struct ExecutionContext {
    void* device_context = nullptr;
    void* command_queue = nullptr;
    std::unordered_map<IRNode*, void*> node_resources;
    
    size_t total_ops = 0;
    double total_time_ms = 0.0;
    std::vector<double> stage_times;
};

} // namespace KuhulIR
