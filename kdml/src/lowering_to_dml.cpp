// lowering_to_dml.cpp - DirectML Lowering Implementation
// Frozen Reference Implementation for Layer 6 Compiler
// Date: 2026-07-12

#include "kuhul/lowering_to_dml.h"
#include "kuhul/ir_types.h"
#include <algorithm>
#include <iostream>

namespace KuhulIR {

DirectMLLowering::DirectMLLowering() {}

DirectMLLowering::~DirectMLLowering() {
    // Clean up allocated resources
    for (auto* resource : m_allocated_resources) {
        if (resource) {
            // resource->Release();  // Uncomment when using actual COM objects
        }
    }
    m_allocated_resources.clear();
}

void DirectMLLowering::Initialize(ID3D12Device* d3dDevice, 
                                   IDMLDevice* dmlDevice, 
                                   ID3D12CommandQueue* cmdQueue) {
    m_d3dDevice = d3dDevice;
    m_dmlDevice = dmlDevice;
    m_cmdQueue = cmdQueue;
}

IDMLCompiledOperator* DirectMLLowering::LowerNodeToDML(IRNode* node) {
    if (!node) return nullptr;
    
    DMLOperatorDesc desc = BuildDMLDesc(node);
    return CreateDMLOperator(desc);
}

IDMLCompiledOperator* DirectMLLowering::LowerGraph(IRGraph* graph) {
    if (!graph) return nullptr;
    
    // 1. Validate graph
    if (!ValidateGraph(graph)) {
        LogError("Graph validation failed");
        return nullptr;
    }
    LogInfo("✓ Graph validation passed");
    
    // 2. Topological sort
    std::vector<IRNode*> sorted_nodes = TopologicalSort(graph);
    if (sorted_nodes.empty()) {
        LogError("Topological sort failed or empty graph");
        return nullptr;
    }
    LogInfo("✓ Topological sort: " + std::to_string(sorted_nodes.size()) + " nodes");
    
    // 3. Lower each node
    std::unordered_map<IRNode*, IDMLCompiledOperator*> node_to_dml;
    std::unordered_map<IRNode*, ID3D12Resource*> node_to_resource;
    
    for (auto* node : sorted_nodes) {
        node_to_dml[node] = LowerNodeToDML(node);
        if (!node_to_dml[node]) {
            LogError("Failed to lower node: " + node->id);
            return nullptr;
        }
        
        // Allocate resources for outputs
        node_to_resource[node] = AllocateResource(node->output_tensor_shape);
        if (!node_to_resource[node]) {
            LogError("Failed to allocate resource for node: " + node->id);
            return nullptr;
        }
    }
    LogInfo("✓ All nodes lowered and resources allocated");
    
    // 4. Build execution plan
    return BuildExecutionPlan(sorted_nodes, node_to_dml, node_to_resource);
}

DMLOperatorDesc DirectMLLowering::BuildDMLDesc(IRNode* node) {
    DMLOperatorDesc desc;
    desc.name = node->id;
    desc.input_tensor_shapes = GetInputShapes(node);
    desc.output_tensor_shapes = GetOutputShapes(node);
    
    // Map op_code to DML type
    if (node->op_code == "conv2d") {
        desc.type = DMLOperatorDesc::Type::Conv2D;
        auto& params = desc.specific_params.emplace<DMLOperatorDesc::Conv2DParams>();
        
        if (auto strides = node->get_attribute<std::vector<Value>>("strides")) {
            params.strides = ExtractSizeVector(*strides);
        }
        if (auto dilations = node->get_attribute<std::vector<Value>>("dilations")) {
            params.dilations = ExtractSizeVector(*dilations);
        }
        if (auto groups = node->get_attribute<int64_t>("groups")) {
            params.groups = *groups;
        }
        if (auto use_bias = node->get_attribute<bool>("use_bias")) {
            params.use_bias = *use_bias;
        }
        
    } else if (node->op_code == "matmul") {
        desc.type = DMLOperatorDesc::Type::MatMul;
        auto& params = desc.specific_params.emplace<DMLOperatorDesc::MatMulParams>();
        
        if (auto transpose_a = node->get_attribute<bool>("transpose_a")) {
            params.transpose_a = *transpose_a;
        }
        if (auto transpose_b = node->get_attribute<bool>("transpose_b")) {
            params.transpose_b = *transpose_b;
        }
        if (auto alpha = node->get_attribute<double>("alpha")) {
            params.alpha = *alpha;
        }
        
    } else if (node->op_code == "relu") {
        desc.type = DMLOperatorDesc::Type::ReLU;
        
    } else if (node->op_code == "softmax") {
        desc.type = DMLOperatorDesc::Type::Softmax;
        
    } else if (node->op_code == "add") {
        desc.type = DMLOperatorDesc::Type::Add;
        
    } else if (node->op_code == "multiply") {
        desc.type = DMLOperatorDesc::Type::Multiply;
        
    } else if (node->op_code == "constant") {
        desc.type = DMLOperatorDesc::Type::Constant;
        if (auto value = node->get_attribute<Value>("value")) {
            desc.parameters["value"] = *value;
        }
        
    } else if (node->op_code == "pooling") {
        desc.type = DMLOperatorDesc::Type::Pooling;
        auto& params = desc.specific_params.emplace<DMLOperatorDesc::PoolingParams>();
        
        if (auto kernel = node->get_attribute<std::vector<Value>>("kernel_size")) {
            params.kernel_size = ExtractSizeVector(*kernel);
        }
        if (auto strides = node->get_attribute<std::vector<Value>>("strides")) {
            params.strides = ExtractSizeVector(*strides);
        }
        if (auto mode = node->get_attribute<std::string>("mode")) {
            params.mode = (*mode == "max") ? 
                DMLOperatorDesc::PoolingParams::Mode::Max : 
                DMLOperatorDesc::PoolingParams::Mode::Average;
        }
        
    } else if (node->op_code == "activation") {
        desc.type = DMLOperatorDesc::Type::Activation;
        auto& params = desc.specific_params.emplace<DMLOperatorDesc::ActivationParams>();
        
        if (auto type = node->get_attribute<std::string>("type")) {
            if (*type == "relu") params.type = DMLOperatorDesc::ActivationParams::Type::ReLU;
            else if (*type == "sigmoid") params.type = DMLOperatorDesc::ActivationParams::Type::Sigmoid;
            else if (*type == "tanh") params.type = DMLOperatorDesc::ActivationParams::Type::Tanh;
            else if (*type == "leaky_relu") params.type = DMLOperatorDesc::ActivationParams::Type::LeakyReLU;
            else if (*type == "elu") params.type = DMLOperatorDesc::ActivationParams::Type::ELU;
            else if (*type == "gelu") params.type = DMLOperatorDesc::ActivationParams::Type::GELU;
        }
        if (auto alpha = node->get_attribute<double>("alpha")) {
            params.alpha = *alpha;
        }
    }
    
    return desc;
}

IDMLCompiledOperator* DirectMLLowering::CreateDMLOperator(const DMLOperatorDesc& desc) {
    // Stub implementation - full implementation requires DirectML SDK
    LogInfo("Creating DML operator: " + desc.name + " (type: " + std::to_string((int)desc.type) + ")");
    return nullptr; // Would return compiled operator in full implementation
}

bool DirectMLLowering::ValidateGraph(IRGraph* graph) {
    if (!graph || graph->nodes.empty()) {
        LogError("Graph is null or empty");
        return false;
    }
    
    // Check for cycles
    if (HasCycle(graph)) {
        LogError("Graph contains cycles");
        return false;
    }
    
    // Check tensor shapes are valid
    for (auto* node : graph->nodes) {
        if (!ValidateTensorShapes(node)) {
            return false;
        }
    }
    
    return true;
}

bool DirectMLLowering::HasCycle(IRGraph* graph) {
    std::unordered_map<IRNode*, bool> visited;
    std::unordered_map<IRNode*, bool> recursion_stack;
    
    for (auto* node : graph->nodes) {
        if (!visited[node]) {
            if (HasCycleDFS(node, visited, recursion_stack)) {
                return true;
            }
        }
    }
    return false;
}

bool DirectMLLowering::HasCycleDFS(IRNode* node,
                                   std::unordered_map<IRNode*, bool>& visited,
                                   std::unordered_map<IRNode*, bool>& recursion_stack) {
    visited[node] = true;
    recursion_stack[node] = true;
    
    for (auto* input : node->inputs) {
        if (!visited[input]) {
            if (HasCycleDFS(input, visited, recursion_stack)) {
                return true;
            }
        } else if (recursion_stack[input]) {
            return true;
        }
    }
    
    recursion_stack[node] = false;
    return false;
}

bool DirectMLLowering::ValidateTensorShapes(IRNode* node) {
    if (!node) return false;
    
    if (node->output_tensor_shape.shape.empty()) {
        LogError("Node " + node->id + " has empty output shape");
        return false;
    }
    
    for (auto dim : node->output_tensor_shape.shape) {
        if (dim == 0) {
            LogError("Node " + node->id + " has zero dimension");
            return false;
        }
    }
    
    return true;
}

std::vector<IRNode*> DirectMLLowering::TopologicalSort(IRGraph* graph) {
    std::unordered_map<IRNode*, size_t> in_degree;
    std::queue<IRNode*> queue;
    std::vector<IRNode*> sorted;
    
    // Calculate in-degrees
    for (auto* node : graph->nodes) {
        in_degree[node] = node->inputs.size();
        if (in_degree[node] == 0) {
            queue.push(node);
        }
    }
    
    // Kahn's algorithm
    while (!queue.empty()) {
        auto* node = queue.front();
        queue.pop();
        sorted.push_back(node);
        
        for (auto* output : node->outputs) {
            in_degree[output]--;
            if (in_degree[output] == 0) {
                queue.push(output);
            }
        }
    }
    
    return sorted;
}

ID3D12Resource* DirectMLLowering::AllocateResource(const TensorType& shape) {
    // Stub - would allocate D3D12 buffer in full implementation
    LogInfo("Allocating resource: " + std::to_string(shape.bytes()) + " bytes");
    m_allocated_resources.push_back(nullptr);
    return nullptr;
}

void DirectMLLowering::FreeResource(ID3D12Resource* resource) {
    if (resource) {
        auto it = std::find(m_allocated_resources.begin(), m_allocated_resources.end(), resource);
        if (it != m_allocated_resources.end()) {
            m_allocated_resources.erase(it);
        }
        // resource->Release();
    }
}

IDMLCompiledOperator* DirectMLLowering::BuildExecutionPlan(
    const std::vector<IRNode*>& sorted_nodes,
    const std::unordered_map<IRNode*, IDMLCompiledOperator*>& node_to_dml,
    const std::unordered_map<IRNode*, ID3D12Resource*>& node_to_resource) {
    
    LogInfo("Building execution plan for " + std::to_string(sorted_nodes.size()) + " nodes");
    
    // Stub - would create command list and binding tables
    return nullptr;
}

std::vector<size_t> DirectMLLowering::GetInputShapes(IRNode* node) {
    std::vector<size_t> shapes;
    for (auto* input : node->inputs) {
        if (input) {
            shapes.push_back(input->output_tensor_shape.size);
        }
    }
    return shapes;
}

std::vector<size_t> DirectMLLowering::GetOutputShapes(IRNode* node) {
    std::vector<size_t> shapes;
    if (node->output_tensor_shape.size > 0) {
        shapes.push_back(node->output_tensor_shape.size);
    }
    return shapes;
}

std::vector<size_t> DirectMLLowering::ExtractSizeVector(const std::vector<Value>& values) {
    std::vector<size_t> result;
    for (const auto& val : values) {
        if (auto* int_val = std::get_if<int64_t>(&val)) {
            result.push_back(static_cast<size_t>(*int_val));
        }
    }
    return result;
}

int DirectMLLowering::MapToDMLType(DMLOperatorDesc::Type type) {
    switch (type) {
        case DMLOperatorDesc::Type::Conv2D: return 1;
        case DMLOperatorDesc::Type::MatMul: return 2;
        case DMLOperatorDesc::Type::ReLU: return 3;
        case DMLOperatorDesc::Type::Softmax: return 4;
        case DMLOperatorDesc::Type::Add: return 5;
        case DMLOperatorDesc::Type::Multiply: return 6;
        case DMLOperatorDesc::Type::Pooling: return 7;
        default: return 0;
    }
}

void DirectMLLowering::LogError(const std::string& message) {
    std::cerr << "[ERROR] " << message << std::endl;
}

void DirectMLLowering::LogInfo(const std::string& message) {
    std::cout << "[INFO] " << message << std::endl;
}

std::unique_ptr<DirectMLLowering> CreateDirectMLLowering(
    ID3D12Device* d3dDevice,
    IDMLDevice* dmlDevice,
    ID3D12CommandQueue* cmdQueue) {
    
    auto lowering = std::make_unique<DirectMLLowering>();
    lowering->Initialize(d3dDevice, dmlDevice, cmdQueue);
    return lowering;
}

} // namespace KuhulIR
