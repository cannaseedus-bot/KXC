#pragma once

#include <vector>
#include <cstdint>
#include <unordered_map>

// Forward declarations from executor.h to avoid circular include
enum class XCFEPhase : uint8_t;
struct ExecutionNode;

enum class OpCode : uint16_t {
    NOP = 0,
    ADD_I64 = 1,
    MUL_I64 = 2,
    FUSED_ADD_MUL_I64 = 3,
    BIMODAL_ATTENTION = 4,
    GEODESIC_FLOW = 5,
    // extend as needed
};

struct IRNode {
    XCFEPhase phase = static_cast<XCFEPhase>(0);
    OpCode opcode = OpCode::NOP;
    uint32_t a_idx = 0;
    uint32_t b_idx = 0;
    uint32_t c_idx = 0; // optional, used for fused nodes
    uint32_t out_idx = 0;
};

struct Buffers {
    std::vector<int64_t> i64;
    std::vector<double> f64;
    std::vector<float> f32; // for matmul/FP32 tensors
};

struct LaneBatch {
    OpCode opcode = OpCode::NOP;
    std::vector<IRNode*> nodes;
};

using BatchMap = std::unordered_map<OpCode, LaneBatch>;

// build IRNodes from ExecutionNode tree (simple mapping for add)
bool build_ir_from_execution(ExecutionNode &root, std::vector<IRNode> &out_ir, Buffers &buf);

// build batches per phase
void build_batches_for_phase(std::vector<IRNode> &ir, XCFEPhase phase, BatchMap &out_batches);
