#pragma once
#include "kxc.h"

// Lower a KernelDesc (MATRIX) → KernelIR (SCXQ2).
// Reads registry JSON from ctx.registryDir to perform classification.
bool lower(const KernelDesc& desc, const CompileCtx& ctx, KernelIR& out, std::string& err);
