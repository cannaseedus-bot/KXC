#pragma once

#include <string>
#include "planner.h"
#include "compiler.h"

// Execute the given plan on DX12 if available, otherwise produce a GPU dispatch schedule
// schedulePath: output JSON schedule describing dispatches (for external DX12 runner)
// If enableGpu=true and build enabled with DX12, an attempt to run GPU dispatch will be made (best-effort).
bool executePlanDX12(ExecutionPlan &plan, CompiledProgram &prog, const std::string &schedulePath, bool enableGpu=false);

// Helper: build dispatch params for a batch (packs per-batch arrays)
struct DispatchParams {
    uint32_t count;
    std::vector<int64_t> A;
    std::vector<int64_t> B;
    std::vector<int64_t> C; // for fused kernels
    std::vector<int64_t> OUT; // zero-initialized
};

DispatchParams buildDispatchParamsFromBatch(const LaneBatch &batch, const CompiledProgram &prog);
