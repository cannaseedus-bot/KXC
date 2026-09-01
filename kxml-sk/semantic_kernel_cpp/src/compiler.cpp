#include "../include/compiler.h"
#include <stdexcept>

static void lowerNode(ExecutionNode &in, CompiledProgram &out){
    // support 'add' and 'matmul' and 'mul'
    if (in.op == "add"){
        IRNode ir;
        ir.phase = in.phase;
        ir.opcode = OpCode::ADD_I64;
        // operand a
        if (in.state.count("a")){
            Value v = in.state.at("a");
            if (std::holds_alternative<int64_t>(v)){
                ir.a_idx = static_cast<uint32_t>(out.buffers.i64.size());
                out.buffers.i64.push_back(std::get<int64_t>(v));
            } else if (std::holds_alternative<std::string>(v)){
                ir.a_idx = out.symbols.resolve_i64(std::get<std::string>(v));
            } else {
                ir.a_idx = out.symbols.resolve_i64("zero");
            }
        } else {
            ir.a_idx = out.symbols.resolve_i64("zero");
        }
        // operand b
        if (in.state.count("b")){
            Value v = in.state.at("b");
            if (std::holds_alternative<int64_t>(v)){
                ir.b_idx = static_cast<uint32_t>(out.buffers.i64.size());
                out.buffers.i64.push_back(std::get<int64_t>(v));
            } else if (std::holds_alternative<std::string>(v)){
                ir.b_idx = out.symbols.resolve_i64(std::get<std::string>(v));
            } else {
                ir.b_idx = out.symbols.resolve_i64("zero");
            }
        } else {
            ir.b_idx = out.symbols.resolve_i64("zero");
        }
        // out target
        if (in.state.count("out") && std::holds_alternative<std::string>(in.state.at("out"))){
            ir.out_idx = out.symbols.resolve_i64(std::get<std::string>(in.state.at("out")));
        } else {
            // default target: node id + "_res"
            std::string name = in.op + "_" + std::to_string(out.nodes.size());
            ir.out_idx = out.symbols.resolve_i64(name);
        }
        out.nodes.push_back(ir);
    } else if (in.op == "mul"){
        IRNode ir;
        ir.phase = in.phase;
        ir.opcode = OpCode::MUL_I64;
        if (in.state.count("a") && std::holds_alternative<int64_t>(in.state.at("a"))) {
            ir.a_idx = static_cast<uint32_t>(out.buffers.i64.size()); out.buffers.i64.push_back(std::get<int64_t>(in.state.at("a")));
        } else if (in.state.count("a") && std::holds_alternative<std::string>(in.state.at("a"))) ir.a_idx = out.symbols.resolve_i64(std::get<std::string>(in.state.at("a")));
        if (in.state.count("b") && std::holds_alternative<int64_t>(in.state.at("b"))) {
            ir.b_idx = static_cast<uint32_t>(out.buffers.i64.size()); out.buffers.i64.push_back(std::get<int64_t>(in.state.at("b")));
        } else if (in.state.count("b") && std::holds_alternative<std::string>(in.state.at("b"))) ir.b_idx = out.symbols.resolve_i64(std::get<std::string>(in.state.at("b")));
        if (in.state.count("out") && std::holds_alternative<std::string>(in.state.at("out"))) ir.out_idx = out.symbols.resolve_i64(std::get<std::string>(in.state.at("out")));
        else ir.out_idx = out.symbols.resolve_i64(in.op + "_" + std::to_string(out.nodes.size()));
        out.nodes.push_back(ir);
    } else if (in.op == "matmul"){
        // expecting state: a (name), b (name), out (name), M,N,K ints
        IRNode ir;
        ir.phase = in.phase;
        ir.opcode = OpCode::MUL_I64;
        // parse dims
        int64_t M = 0, N = 0, K = 0;
        if (in.state.count("M") && std::holds_alternative<int64_t>(in.state.at("M"))) M = std::get<int64_t>(in.state.at("M"));
        if (in.state.count("N") && std::holds_alternative<int64_t>(in.state.at("N"))) N = std::get<int64_t>(in.state.at("N"));
        if (in.state.count("K") && std::holds_alternative<int64_t>(in.state.at("K"))) K = std::get<int64_t>(in.state.at("K"));
        // map buffers to f32 symbols
        if (in.state.count("a") && std::holds_alternative<std::string>(in.state.at("a"))) ir.a_idx = out.symbols.resolve_f32(std::get<std::string>(in.state.at("a")));
        if (in.state.count("b") && std::holds_alternative<std::string>(in.state.at("b"))) ir.b_idx = out.symbols.resolve_f32(std::get<std::string>(in.state.at("b")));
        if (in.state.count("out") && std::holds_alternative<std::string>(in.state.at("out"))) ir.out_idx = out.symbols.resolve_f32(std::get<std::string>(in.state.at("out")));
        out.nodes.push_back(ir);
    } else if (in.op == "bimodal_attention"){
        IRNode ir;
        ir.phase = in.phase;
        ir.opcode = OpCode::BIMODAL_ATTENTION;
        if (in.state.count("a")) ir.a_idx = out.symbols.resolve_f32(std::get<std::string>(in.state.at("a")));
        if (in.state.count("b")) ir.b_idx = out.symbols.resolve_f32(std::get<std::string>(in.state.at("b")));
        if (in.state.count("out")) ir.out_idx = out.symbols.resolve_f32(std::get<std::string>(in.state.at("out")));
        out.nodes.push_back(ir);
    } else if (in.op == "geodesic_flow"){
        IRNode ir;
        ir.phase = in.phase;
        ir.opcode = OpCode::GEODESIC_FLOW;
        if (in.state.count("coords")) ir.a_idx = out.symbols.resolve_f32(std::get<std::string>(in.state.at("coords")));
        if (in.state.count("velocity")) ir.b_idx = out.symbols.resolve_f32(std::get<std::string>(in.state.at("velocity")));
        if (in.state.count("out")) ir.out_idx = out.symbols.resolve_f32(std::get<std::string>(in.state.at("out")));
        out.nodes.push_back(ir);
    }


    for (auto &ch : in.children) lowerNode(const_cast<ExecutionNode&>(ch), out);
}

bool compileProgram(ExecutionNode &root, CompiledProgram &out){
    out.nodes.clear(); out.buffers.i64.clear(); out.buffers.f64.clear(); out.symbols = SymbolTable(&out.buffers);
    try{
        lowerNode(root, out);
    } catch (const std::exception &){
        return false;
    }
    return true;
}
