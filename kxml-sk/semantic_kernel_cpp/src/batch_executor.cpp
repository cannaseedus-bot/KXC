#include "../include/executor.h"
#include "../include/ir.h"
#include <unordered_map>
#include <unordered_set>

// Build IRNodes from ExecutionNode list for a single phase. We append numeric operands
// into Buffers (SoA). For correctness first implementation uses gather/scatter: each
// node's a,b,out are placed into buffers and IRNode stores indices.

bool build_ir_from_execution(ExecutionNode &root, std::vector<IRNode> &out_ir, Buffers &buf){
    // flatten nodes of all phases - but caller will restrict to phase nodes
    // Not used in this simple API; stub returns false
    (void)root; (void)out_ir; (void)buf; return false;
}

void build_batches_for_phase(std::vector<IRNode> &ir, XCFEPhase phase, BatchMap &out_batches){
    out_batches.clear();
    for (auto &node : ir){
        if (node.phase != phase) continue;
        auto &batch = out_batches[node.opcode];
        batch.opcode = node.opcode;
        batch.nodes.push_back(const_cast<IRNode*>(&node));
    }
}

// Helper to lower a list of ExecutionNode* (all in same phase) to IRNodes and buffers
static void lower_phase_nodes_to_ir(const std::vector<ExecutionNode*> &nodes, XCFEPhase phase, std::vector<IRNode> &out_ir, Buffers &buf, std::vector<ExecutionNode*> &out_node_map){
    out_ir.clear(); out_node_map.clear();
    // deterministic: nodes already in order
    for (auto *n : nodes){
        if (n->op=="add"){
            IRNode ir;
            ir.phase = phase;
            ir.opcode = OpCode::ADD_I64;
            // a
            int64_t aval = 0; int64_t bval = 0;
            if (n->state.count("a") && std::holds_alternative<int64_t>(n->state.at("a"))) aval = std::get<int64_t>(n->state.at("a"));
            if (n->state.count("b") && std::holds_alternative<int64_t>(n->state.at("b"))) bval = std::get<int64_t>(n->state.at("b"));
            ir.a_idx = static_cast<uint32_t>(buf.i64.size()); buf.i64.push_back(aval);
            ir.b_idx = static_cast<uint32_t>(buf.i64.size()); buf.i64.push_back(bval);
            ir.out_idx = static_cast<uint32_t>(buf.i64.size()); buf.i64.push_back(0);
            out_ir.push_back(ir);
            out_node_map.push_back(n);
        } else if (n->op == "bimodal_attention") {
            IRNode ir;
            ir.phase = phase;
            ir.opcode = OpCode::BIMODAL_ATTENTION;
            float aval = 0.0f, bval = 0.0f;
            // Simplified: we'll use f32 buffers for these geometric ops
            // In a real system, these would be large tensor references
            ir.a_idx = static_cast<uint32_t>(buf.f32.size()); buf.f32.push_back(aval);
            ir.b_idx = static_cast<uint32_t>(buf.f32.size()); buf.f32.push_back(bval);
            ir.out_idx = static_cast<uint32_t>(buf.f32.size()); buf.f32.push_back(0);
            out_ir.push_back(ir);
            out_node_map.push_back(n);
        } else if (n->op == "geodesic_flow") {
            IRNode ir;
            ir.phase = phase;
            ir.opcode = OpCode::GEODESIC_FLOW;
            float aval = 0.0f, bval = 0.0f;
            ir.a_idx = static_cast<uint32_t>(buf.f32.size()); buf.f32.push_back(aval);
            ir.b_idx = static_cast<uint32_t>(buf.f32.size()); buf.f32.push_back(bval);
            ir.out_idx = static_cast<uint32_t>(buf.f32.size()); buf.f32.push_back(0);
            out_ir.push_back(ir);
            out_node_map.push_back(n);
        }
    }
}


void execute_phases_with_batches(ExecutionNode &root, OpRegistry &reg, size_t threads){
    BatchExecutor bex;
    ExecutionContext ctx;
    std::vector<XCFEPhase> phases = {XCFEPhase::Pop, XCFEPhase::Wo, XCFEPhase::Sek, XCFEPhase::Chen, XCFEPhase::Xul};
    for (auto phase: phases){
        std::vector<ExecutionNode*> nodes; collectPhaseNodes(root, phase, nodes);
        if (nodes.empty()) continue;
        // Lower to IR
        std::vector<IRNode> ir; Buffers buf; std::vector<ExecutionNode*> node_map;
        lower_phase_nodes_to_ir(nodes, phase, ir, buf, node_map);
        if (ir.empty()){
            // no SIMD-applicable nodes in this phase; run scalar ops via registry
            for (auto n : nodes) if (!n->op.empty()) reg.execute(n->op, *n, ctx);
            // merge phaseBuffer into global as run_execution would
            for (auto &p: ctx.phaseBuffer) ctx.global[p.first] = p.second;
            ctx.phaseBuffer.clear();
            continue;
        }
        // build batches
        BatchMap batches; build_batches_for_phase(ir, phase, batches);
        // execute batches
        bex.executePhase(batches, buf);
        // scatter results back into execution nodes' state and phaseBuffer
        for (size_t i=0;i<ir.size();++i){
            auto &node = ir[i];
            auto *exec = node_map[i];
            if (node.opcode==OpCode::ADD_I64){
                int64_t res = buf.i64[node.out_idx];
                exec->state["result"] = res;
                ctx.phaseBuffer["result"] = res;
            } else if (node.opcode == OpCode::BIMODAL_ATTENTION || node.opcode == OpCode::GEODESIC_FLOW) {
                float res = buf.f32[node.out_idx];
                exec->state["tensor_result"] = res;
                ctx.phaseBuffer["manifold_coherence"] = res;
            }
        }

        // run scalar/non-vector nodes in this phase
        // build set of vectorized nodes
        std::unordered_set<ExecutionNode*> vectorized;
        for (auto ptr : node_map) vectorized.insert(ptr);
        for (auto n : nodes){ if (!vectorized.count(n) && !n->op.empty()) reg.execute(n->op, *n, ctx); }
        // merge phaseBuffer into global
        for (auto &p: ctx.phaseBuffer) ctx.global[p.first] = p.second;
        ctx.phaseBuffer.clear();
    }
}
