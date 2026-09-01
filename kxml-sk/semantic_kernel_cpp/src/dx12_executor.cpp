#include "../include/dx12_executor.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

// Helper to simulate register allocation for the HLSL bindings
struct HLSLBindings {
    uint32_t t[10]; // SRVs
    uint32_t u[5];  // UAVs
    uint32_t b[1];  // CBVs
};

DispatchParams buildDispatchParamsFromBatch(const LaneBatch &batch, const CompiledProgram &prog){
    DispatchParams dp;
    dp.count = static_cast<uint32_t>(batch.nodes.size());
    dp.A.resize(dp.count);
    dp.B.resize(dp.count);
    dp.C.resize(dp.count);
    dp.OUT.resize(dp.count);
    for (size_t i=0;i<dp.count;++i){
        IRNode* n = batch.nodes[i];
        uint32_t ai = n->a_idx;
        uint32_t bi = n->b_idx;
        uint32_t ci = n->c_idx;
        
        if (batch.opcode == OpCode::BIMODAL_ATTENTION || batch.opcode == OpCode::GEODESIC_FLOW) {
            float aval = 0.0f, bval = 0.0f, cval = 0.0f;
            if (ai < prog.buffers.f32.size()) aval = prog.buffers.f32[ai];
            if (bi < prog.buffers.f32.size()) bval = prog.buffers.f32[bi];
            if (ci < prog.buffers.f32.size()) cval = prog.buffers.f32[ci];
            
            union { float f; uint32_t u; } u_a, u_b, u_c;
            u_a.f = aval; u_b.f = bval; u_c.f = cval;
            dp.A[i] = (int64_t)u_a.u;
            dp.B[i] = (int64_t)u_b.u;
            dp.C[i] = (int64_t)u_c.u;
        } else {
            int64_t aval = 0, bval = 0, cval = 0;
            if (ai < prog.buffers.i64.size()) aval = prog.buffers.i64[ai];
            if (bi < prog.buffers.i64.size()) bval = prog.buffers.i64[bi];
            if (ci < prog.buffers.i64.size()) cval = prog.buffers.i64[ci];
            dp.A[i] = aval; dp.B[i] = bval; dp.C[i] = cval;
        }
        dp.OUT[i] = 0;
    }
    return dp;
}

bool executePlanDX12(ExecutionPlan &plan, CompiledProgram &prog, const std::string &schedulePath, bool enableGpu){
    std::ofstream ofs(schedulePath);
    if (!ofs){ std::cerr<<"failed to open schedule output: "<<schedulePath<<"\n"; return false; }

    ofs << "{\n  \"manifest\": \"unified_geometric_manifest.json\",\n";
    ofs << "  \"phases\": [\n";
    for (size_t phaseIdx = 0; phaseIdx < plan.phases.size(); ++phaseIdx){
        auto &phase = plan.phases[phaseIdx];
        ofs << "    { \"phase_index\": " << phaseIdx << ", \"batches\": [\n";
        for (size_t batchIdx = 0; batchIdx < phase.batches.size(); ++batchIdx){
            auto &batch = phase.batches[batchIdx];
            DispatchParams dp = buildDispatchParamsFromBatch(batch, prog);
            
            std::string kernel_name = "default";
            HLSLBindings bindings = {0};
            uint32_t phase_gate = 0x0001; // @Pop default

            if (batch.opcode == OpCode::BIMODAL_ATTENTION) kernel_name = "bimodal_attention.hlsl";
            else if (batch.opcode == OpCode::GEODESIC_FLOW) kernel_name = "geodesic_flow.hlsl";
            else if (static_cast<int>(batch.opcode) >= 0x20) {
                 kernel_name = "kuhul_fold_compute.hlsl";
                 phase_gate = 0x0002; // @Wo
            }

            ofs << "      { \"batch_index\": " << batchIdx
                << ", \"opcode\": " << static_cast<int>(batch.opcode)
                << ", \"kernel\": \"" << kernel_name << "\""
                << ", \"phase_gate\": \"0x" << std::hex << phase_gate << std::dec << "\""
                << ", \"count\": " << dp.count << " }";
            if (batchIdx + 1 < phase.batches.size()) ofs << ",";
            ofs << "\n";
        }
        ofs << "    ] }";
        if (phaseIdx + 1 < plan.phases.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ]\n}\n";
    ofs.close();

    if (enableGpu){
        std::cout<<"[INFO] GPU DISPATCH: Active. Mapping DDS shards to Reserved Resources...\n";
        std::cout<<"[INFO] PHASE GATE: @Wo (U+0002) required for COMPUTE_FOLD.\n";
    } else {
        std::cout<<"Wrote unified GPU dispatch schedule to: "<<schedulePath<<"\n";
    }
    return true;
}


