#pragma once
#include "kxc.h"
#include <vector>
#include <string>

struct ArtifactEntry {
    std::string name;
    std::string cid;  // "fnv1a64:..."
};

bool emit_manifest(const KernelIR& ir, const std::string& smcaPath, std::string& err);
bool emit_compile_report(const KernelIR& ir, const CompileCtx& ctx,
                         const std::vector<ArtifactEntry>& artifacts,
                         const std::string& reportPath, const std::string& idbPath,
                         std::string& err);
