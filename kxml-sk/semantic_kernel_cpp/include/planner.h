#pragma once

#include "compiler.h"
#include "ir.h"
#include <vector>

struct PlannedPhase {
    std::vector<LaneBatch> batches;
};

struct ExecutionPlan {
    std::vector<PlannedPhase> phases; // index by XCFEPhase (0..4)
};

// Create ExecutionPlan from CompiledProgram (groups by phase and opcode)
ExecutionPlan planProgram(const CompiledProgram &prog);

// Execute a prepared plan using Buffers from compiled program
void executePlan(ExecutionPlan &plan, Buffers &buf, size_t threads);
