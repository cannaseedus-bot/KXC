#pragma once
#include "kxc.h"

// Validates IR classification against registry JSON files.
// Sets ir.registryMatched and ir.lawful accordingly.
bool validate_manifest(KernelIR& ir, const CompileCtx& ctx, std::string& err);
