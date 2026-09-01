#include "../include/planner.h"
#include "../include/executor.h"
#include <algorithm>
#include <thread>
#include <unordered_map>

ExecutionPlan planProgram(const CompiledProgram &prog){
    // First perform a simple fusion pass over linear IR to detect ADD->MUL chains
    // and produce a temporary node list where matched pairs are replaced by FUSED nodes.
    std::vector<IRNode> linear;
    linear.reserve(prog.nodes.size());
    for (auto &n : prog.nodes) linear.push_back(n);

    std::vector<IRNode> lowered;
    lowered.reserve(linear.size());

    for (size_t i = 0; i < linear.size(); ++i){
        auto &n = linear[i];
        // pattern: n = ADD, next = MUL, and next.a_idx == n.out_idx
        if (i + 1 < linear.size()){
            auto &n2 = linear[i+1];
            if (n.opcode == OpCode::ADD_I64 && n2.opcode == OpCode::MUL_I64 && n2.a_idx == n.out_idx){
                IRNode fused;
                fused.phase = n.phase;
                fused.opcode = OpCode::FUSED_ADD_MUL_I64;
                fused.a_idx = n.a_idx;
                fused.b_idx = n.b_idx;
                fused.c_idx = n2.b_idx;
                fused.out_idx = n2.out_idx;
                lowered.push_back(fused);
                ++i; // skip next
                continue;
            }
        }
        // otherwise copy
        lowered.push_back(n);
    }

    // now group lowered nodes into phase/op batches
    ExecutionPlan plan;
    plan.phases.resize(5); // Pop, Wo, Sek, Chen, Xul
    for (auto &node : lowered){
        size_t p = static_cast<size_t>(node.phase);
        if (p >= plan.phases.size()) p = static_cast<size_t>(XCFEPhase::Unknown);
        auto &phase = plan.phases[p];
        bool found = false;
        for (auto &batch : phase.batches){
            if (batch.opcode == node.opcode){
                batch.nodes.push_back(const_cast<IRNode*>(&node));
                found = true; break;
            }
        }
        if (!found){
            LaneBatch b; b.opcode = node.opcode; b.nodes.push_back(const_cast<IRNode*>(&node));
            phase.batches.push_back(std::move(b));
        }
    }
    return plan;
}

void executePlan(ExecutionPlan &plan, Buffers &buf, size_t threads){
    ThreadPool pool(threads);

    for (auto &phase : plan.phases){
        // enqueue each batch as a separate task for parallelism
        for (auto &batch : phase.batches){
            pool.taskStarted();
            pool.enqueue([&b = batch, &buf, &pool]() mutable {
                switch (b.opcode){
                    case OpCode::ADD_I64:
                        executeAddLane(const_cast<LaneBatch&>(b), const_cast<Buffers&>(buf));
                        break;
                    case OpCode::FUSED_ADD_MUL_I64:
                        // CPU fallback: for each node, compute (a + b) * c
                        for (auto *n : b.nodes){
                            int64_t a = buf.i64[n->a_idx];
                            int64_t bval = buf.i64[n->b_idx];
                            int64_t c = buf.i64[n->c_idx];
                            int64_t tmp = a + bval;
                            buf.i64[n->out_idx] = tmp * c;
                        }
                        break;
                    default:
                        // scalar fallback for ADD if necessary
                        for (auto *n : b.nodes){
                            if (n->opcode == OpCode::ADD_I64){
                                buf.i64[n->out_idx] = buf.i64[n->a_idx] + buf.i64[n->b_idx];
                            } else if (n->opcode == OpCode::MUL_I64){
                                buf.i64[n->out_idx] = buf.i64[n->a_idx] * buf.i64[n->b_idx];
                            }
                        }
                        break;
                }
                pool.taskFinished();
            });
        }
        // wait for phase completion
        pool.wait();
    }
}
