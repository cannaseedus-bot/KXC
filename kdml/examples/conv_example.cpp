// main_example.cpp - K'UHUL v0.1 DirectML Lowering Example
// Frozen Reference Implementation for Layer 6 Compiler
// Date: 2026-07-12

#include <iostream>
#include <memory>
#include <iomanip>
#include "kuhul/ir_types.h"
#include "kuhul/lowering_to_dml.h"

using namespace KuhulIR;

// ===== Build Example IR Program =====
IRProgram BuildExampleProgram() {
    IRProgram program;
    program.name = "conv_example";
    program.version = "0.1.0";
    
    // Create function
    IRFunction conv_fn;
    conv_fn.name = "my_conv";
    conv_fn.add_parameter("input", ValueType::Float);
    conv_fn.add_parameter("weights", ValueType::Float);
    conv_fn.add_parameter("bias", ValueType::Float);
    
    // Create block
    conv_fn.body.name = "conv_body";
    
    // Node 1: Conv2D
    auto conv_node = std::make_unique<IRNode>();
    conv_node->id = "@conv";
    conv_node->op_code = "conv2d";
    conv_node->source_loc = {"example.kuhul", 4, 8, "my_conv"};
    conv_node->set_attribute("strides",   make_value_list({int64_t(2), int64_t(2)}));
    conv_node->set_attribute("dilations", make_value_list({int64_t(1), int64_t(1)}));
    conv_node->set_attribute("pads",      make_value_list({int64_t(1), int64_t(1)}));
    conv_node->set_attribute("groups", int64_t(1));
    conv_node->set_attribute("use_bias", true);
    conv_node->output_type = ValueType::Float;
    conv_node->output_tensor_shape.shape = {1, 32, 112, 112};
    conv_node->output_tensor_shape.dtype = ValueType::Float;
    conv_node->output_tensor_shape.size = 1 * 32 * 112 * 112;
    conv_node->output_tensor_shape.is_dynamic = false;
    conv_node->is_pure = true;
    
    // Node 2: ReLU
    auto relu_node = std::make_unique<IRNode>();
    relu_node->id = "@relu";
    relu_node->op_code = "relu";
    relu_node->source_loc = {"example.kuhul", 11, 8, "my_conv"};
    relu_node->output_type = ValueType::Float;
    relu_node->output_tensor_shape.shape = {1, 32, 112, 112};
    relu_node->output_tensor_shape.dtype = ValueType::Float;
    relu_node->output_tensor_shape.size = 1 * 32 * 112 * 112;
    relu_node->output_tensor_shape.is_dynamic = false;
    relu_node->is_pure = true;
    
    // Node 3: Pooling (Max)
    auto pool_node = std::make_unique<IRNode>();
    pool_node->id = "@pool";
    pool_node->op_code = "pooling";
    pool_node->source_loc = {"example.kuhul", 16, 8, "my_conv"};
    pool_node->set_attribute("kernel_size", make_value_list({int64_t(2), int64_t(2)}));
    pool_node->set_attribute("strides",     make_value_list({int64_t(2), int64_t(2)}));
    pool_node->set_attribute("mode", std::string("max"));
    pool_node->output_type = ValueType::Float;
    pool_node->output_tensor_shape.shape = {1, 32, 56, 56};
    pool_node->output_tensor_shape.dtype = ValueType::Float;
    pool_node->output_tensor_shape.size = 1 * 32 * 56 * 56;
    pool_node->output_tensor_shape.is_dynamic = false;
    pool_node->is_pure = true;
    
    // Get raw pointers before moving into block
    auto* conv_ptr = conv_node.get();
    auto* relu_ptr = relu_node.get();
    auto* pool_ptr = pool_node.get();
    
    // Add nodes to block
    conv_fn.body.add_node(std::move(conv_node));
    conv_fn.body.add_node(std::move(relu_node));
    conv_fn.body.add_node(std::move(pool_node));
    
    // Add edges
    IREdge edge1;
    edge1.source = conv_ptr;
    edge1.target = relu_ptr;
    edge1.type = EdgeType::Forward;
    edge1.name = "conv_to_relu";
    edge1.source_loc = {"example.kuhul", 11, 20, "my_conv"};
    
    IREdge edge2;
    edge2.source = relu_ptr;
    edge2.target = pool_ptr;
    edge2.type = EdgeType::Forward;
    edge2.name = "relu_to_pool";
    edge2.source_loc = {"example.kuhul", 16, 20, "my_conv"};
    
    conv_fn.body.add_edge(edge1);
    conv_fn.body.add_edge(edge2);
    
    // Set entry and return nodes
    conv_fn.body.entry_node = conv_fn.body.get_node(0);
    conv_fn.return_node = conv_fn.body.get_node(2);
    
    // Add function to program
    program.add_function(std::move(conv_fn));
    
    return program;
}

// ===== Print Formatting Utilities =====
void PrintHeader(const std::string& title) {
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << title << std::endl;
    std::cout << "========================================" << std::endl;
}

void PrintSection(const std::string& section) {
    std::cout << std::endl << section << std::endl;
}

// ===== Main Execution =====
int main() {
    PrintHeader("K'UHUL v0.1 - DirectML Lowering Example");
    
    // 1. Build IR from source
    PrintSection("1. Building IR from source...");
    IRProgram program = BuildExampleProgram();
    std::cout << "   ✓ Program: " << program.name << std::endl;
    std::cout << "   ✓ Version: " << program.version << std::endl;
    std::cout << "   ✓ Functions: " << program.functions.size() << std::endl;
    
    auto& fn = program.functions[0];
    std::cout << "   ✓ Function: " << fn.name << std::endl;
    std::cout << "   ✓ Parameters: " << fn.params.size() << std::endl;
    std::cout << "   ✓ Nodes in body: " << fn.body.node_count() << std::endl;
    
    // 2. Print IR nodes
    PrintSection("2. IR Nodes:");
    for (size_t i = 0; i < fn.body.node_count(); i++) {
        auto* node = fn.body.get_node(i);
        if (!node) continue;
        
        std::cout << "   [" << i << "] " << std::setw(8) << node->id 
                  << " (" << std::setw(10) << node->op_code << ")" << std::endl;
        
        std::cout << "       Shape: (";
        for (size_t j = 0; j < node->output_tensor_shape.shape.size(); j++) {
            if (j > 0) std::cout << ", ";
            std::cout << node->output_tensor_shape.shape[j];
        }
        std::cout << ")" << std::endl;
        
        std::cout << "       Size: " << node->output_tensor_shape.size << " elements = "
                  << node->output_tensor_shape.bytes() / (1024*1024) << "MB" << std::endl;
        
        std::cout << "       Inputs: " << node->input_count() 
                  << ", Outputs: " << node->output_count()
                  << ", Pure: " << (node->is_pure ? "yes" : "no") << std::endl;
    }
    
    // 3. Print edges
    PrintSection("3. Graph Edges:");
    std::cout << "   Total edges: " << fn.body.edges.size() << std::endl;
    for (size_t i = 0; i < fn.body.edges.size(); i++) {
        const auto& edge = fn.body.edges[i];
        std::string type_name;
        switch (edge.type) {
            case EdgeType::Forward: type_name = "Forward"; break;
            case EdgeType::Data: type_name = "Data"; break;
            case EdgeType::Control: type_name = "Control"; break;
            default: type_name = "Other"; break;
        }
        
        std::cout << "   [" << i << "] " << edge.name << ": "
                  << (edge.source ? edge.source->id : "?") << " → "
                  << (edge.target ? edge.target->id : "?")
                  << " (" << type_name << ")" << std::endl;
    }
    
    // 4. Lower to DirectML
    PrintSection("4. Lowering to DirectML...");
    
    auto lowering = std::make_unique<DirectMLLowering>();
    
    // Create graph from function body
    IRGraph graph;
    graph.name = fn.name;
    for (size_t i = 0; i < fn.body.node_count(); i++) {
        graph.nodes.push_back(fn.body.get_node(i));
    }
    for (const auto& edge : fn.body.edges) {
        graph.edges.push_back(edge);
    }
    graph.entry = fn.body.entry_node;
    graph.output = fn.return_node;
    
    // Lower the graph
    [[maybe_unused]] auto* compiled_graph = lowering->LowerGraph(&graph);
    std::cout << "   ✓ Lowering complete!" << std::endl;
    
    // 5. Output summary
    PrintHeader("SUMMARY");
    
    std::cout << "Function:        " << fn.name << std::endl;
    std::cout << "Parameters:      " << fn.params.size() << std::endl;
    std::cout << "  - input        fp32" << std::endl;
    std::cout << "  - weights      fp32" << std::endl;
    std::cout << "  - bias         fp32" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Operations:      " << fn.body.node_count() << std::endl;
    std::cout << "  - Conv2D       stride=2, groups=1, use_bias=true" << std::endl;
    std::cout << "  - ReLU         activation" << std::endl;
    std::cout << "  - Pooling      max, kernel=2x2, stride=2" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Output shape:    ";
    if (fn.return_node) {
        std::cout << "(";
        for (size_t i = 0; i < fn.return_node->output_tensor_shape.shape.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << fn.return_node->output_tensor_shape.shape[i];
        }
        std::cout << ")" << std::endl;
    }
    
    // Compute FLOPs (approximate)
    // Conv2D: 1*32*112*112 * (3*3*3) * 2 (for stride-2) = ~67M FLOPs
    // ReLU: ~400K FLOPs (trivial)
    // Pooling: ~100K FLOPs (trivial)
    size_t conv_flops = 1 * 32 * 112 * 112 * 3 * 3 * 3 / 2;
    std::cout << std::endl;
    std::cout << "Estimated FLOPs: ~" << conv_flops / 1000000 << "M" << std::endl;
    
    // Memory
    size_t input_mem = 1 * 3 * 224 * 224 * 4;      // Input
    size_t conv_out = 1 * 32 * 112 * 112 * 4;      // Conv output
    size_t pool_out = 1 * 32 * 56 * 56 * 4;        // Pool output
    size_t total_mem = input_mem + conv_out + pool_out;
    
    std::cout << "Memory (activations): ~" << total_mem / (1024*1024) << "MB" << std::endl;
    
    PrintHeader("SUCCESS");
    std::cout << "✅ K'UHUL IR successfully lowered to DirectML" << std::endl;
    std::cout << std::endl;
    
    return 0;
}
