// test_ir_types.cpp - Basic IR Type System Tests
// Frozen Reference Implementation for Layer 6 Compiler
// Date: 2026-07-12

#include <iostream>
#include <cassert>
#include "kuhul/ir_types.h"

using namespace KuhulIR;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "K'UHUL IR Type System - Basic Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Test 1: Value variant
    std::cout << "Test 1: Value variant types" << std::endl;
    Value v_int = int64_t(42);
    Value v_float = 3.14159;
    Value v_bool = true;
    Value v_string = std::string("hello");
    Value v_list = std::vector<Value>{int64_t(1), int64_t(2), int64_t(3)};
    
    assert(std::holds_alternative<int64_t>(v_int));
    assert(std::holds_alternative<double>(v_float));
    assert(std::holds_alternative<bool>(v_bool));
    assert(std::holds_alternative<std::string>(v_string));
    assert(std::holds_alternative<std::vector<Value>>(v_list));
    
    std::cout << "  ✓ All value types created successfully" << std::endl;
    std::cout << std::endl;
    
    // Test 2: Tensor type
    std::cout << "Test 2: Tensor type and shape" << std::endl;
    TensorType tensor;
    tensor.shape = {1, 32, 112, 112};
    tensor.dtype = ValueType::Float;
    tensor.size = 1 * 32 * 112 * 112;
    tensor.is_dynamic = false;
    
    assert(tensor.total_elements() == 402432);
    assert(tensor.bytes() == 402432 * 4);
    
    std::cout << "  ✓ Tensor shape: (1, 32, 112, 112)" << std::endl;
    std::cout << "  ✓ Total elements: " << tensor.total_elements() << std::endl;
    std::cout << "  ✓ Memory size: " << tensor.bytes() / (1024*1024) << "MB" << std::endl;
    std::cout << std::endl;
    
    // Test 3: IR Node creation
    std::cout << "Test 3: IR Node creation" << std::endl;
    auto node = std::make_unique<IRNode>();
    node->id = "@conv";
    node->op_code = "conv2d";
    node->graph_id = 0;
    node->source_loc.file = "example.kuhul";
    node->source_loc.line = 10;
    node->output_type = ValueType::Float;
    node->output_tensor_shape = tensor;
    node->is_pure = true;
    
    node->set_attribute("strides", std::vector<Value>{int64_t(2), int64_t(2)});
    node->set_attribute("groups", int64_t(1));
    
    assert(node->has_attribute("strides"));
    assert(node->has_attribute("groups"));
    assert(!node->has_attribute("unknown"));
    
    auto groups = node->get_attribute<int64_t>("groups");
    assert(groups.has_value());
    assert(groups.value() == 1);
    
    std::cout << "  ✓ Node created: " << node->id << " (" << node->op_code << ")" << std::endl;
    std::cout << "  ✓ Attributes: strides, groups" << std::endl;
    std::cout << "  ✓ Output shape: (1, 32, 112, 112)" << std::endl;
    std::cout << std::endl;
    
    // Test 4: IR Block
    std::cout << "Test 4: IR Block and graph structure" << std::endl;
    IRBlock block;
    block.name = "conv_block";
    
    // Create first node
    auto node1 = std::make_unique<IRNode>();
    node1->id = "@input";
    node1->op_code = "input";
    node1->output_tensor_shape.shape = {1, 3, 224, 224};
    node1->output_tensor_shape.size = 1 * 3 * 224 * 224;
    
    auto* node1_ptr = node1.get();
    block.add_node(std::move(node1));
    
    // Create second node
    auto node2 = std::make_unique<IRNode>();
    node2->id = "@conv";
    node2->op_code = "conv2d";
    node2->output_tensor_shape.shape = {1, 32, 112, 112};
    node2->output_tensor_shape.size = 1 * 32 * 112 * 112;
    
    auto* node2_ptr = node2.get();
    block.add_node(std::move(node2));
    
    // Create edge
    IREdge edge;
    edge.source = node1_ptr;
    edge.target = node2_ptr;
    edge.type = EdgeType::Forward;
    edge.name = "input_to_conv";
    
    // Note: Can't call add_edge yet since it modifies the raw pointers
    // This would need nodes to be stored first
    
    assert(block.node_count() == 2);
    assert(block.get_node(0) == node1_ptr);
    assert(block.get_node(1) == node2_ptr);
    
    std::cout << "  ✓ Block created: " << block.name << std::endl;
    std::cout << "  ✓ Nodes in block: " << block.node_count() << std::endl;
    std::cout << "  ✓ Node 0: " << block.get_node(0)->id << std::endl;
    std::cout << "  ✓ Node 1: " << block.get_node(1)->id << std::endl;
    std::cout << std::endl;
    
    // Test 5: IR Function
    std::cout << "Test 5: IR Function" << std::endl;
    IRFunction func;
    func.name = "my_conv";
    func.add_parameter("input", ValueType::Float);
    func.add_parameter("weights", ValueType::Float);
    func.add_parameter("bias", ValueType::Float);
    
    assert(func.params.size() == 3);
    assert(func.param_types[0] == ValueType::Float);
    assert(func.params[0] == "input");
    
    std::cout << "  ✓ Function: " << func.name << std::endl;
    std::cout << "  ✓ Parameters: ";
    for (const auto& param : func.params) {
        std::cout << param << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    
    // Test 6: IR Program
    std::cout << "Test 6: IR Program" << std::endl;
    IRProgram program;
    program.name = "conv_inference";
    program.version = "0.1.0";
    program.add_function(std::move(func));
    
    assert(program.functions.size() == 1);
    assert(program.functions[0].name == "my_conv");
    
    auto* fn = program.get_function("my_conv");
    assert(fn != nullptr);
    assert(fn->name == "my_conv");
    
    std::cout << "  ✓ Program: " << program.name << std::endl;
    std::cout << "  ✓ Version: " << program.version << std::endl;
    std::cout << "  ✓ Functions: " << program.functions.size() << std::endl;
    std::cout << "  ✓ Found function: " << fn->name << std::endl;
    std::cout << std::endl;
    
    // Test 7: DML Operator Descriptor
    std::cout << "Test 7: DML Operator Descriptor" << std::endl;
    DMLOperatorDesc desc;
    desc.type = DMLOperatorDesc::Type::Conv2D;
    desc.name = "conv_op";
    desc.input_tensor_shapes = {1, 3, 224, 224};
    
    auto& conv_params = desc.specific_params.emplace<DMLOperatorDesc::Conv2DParams>();
    conv_params.strides = {2, 2};
    conv_params.groups = 1;
    conv_params.use_bias = true;
    
    assert(desc.has_specific_params());
    auto* params = desc.get_specific_params<DMLOperatorDesc::Conv2DParams>();
    assert(params != nullptr);
    assert(params->groups == 1);
    
    std::cout << "  ✓ DML operator: " << desc.name << " (Conv2D)" << std::endl;
    std::cout << "  ✓ Specific params created" << std::endl;
    std::cout << "  ✓ Groups: " << params->groups << std::endl;
    std::cout << "  ✓ Use bias: " << (params->use_bias ? "true" : "false") << std::endl;
    std::cout << std::endl;
    
    // ===== Test Summary =====
    std::cout << "========================================" << std::endl;
    std::cout << "All tests passed! ✅" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
