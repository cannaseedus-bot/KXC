#include "validate_manifest.h"
#include <fstream>
#include <sstream>

static std::string slurp_vm(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};
    std::ostringstream ss; ss << f.rdbuf(); return ss.str();
}

bool validate_manifest(KernelIR& ir, const CompileCtx& ctx, std::string& err) {
    // kernel-classes.json must contain the kernelClass
    std::string classJson = slurp_vm(ctx.registryDir + "/kernel-classes.json");
    if (classJson.empty()) {
        // registry not found — mark unmatched but don't fail
        ir.registryMatched = false;
        return true;
    }
    ir.registryMatched = (classJson.find(ir.kernelClass) != std::string::npos);

    // kernel-extras.json: check binding tier requirement
    std::string extrasJson = slurp_vm(ctx.registryDir + "/kernel-extras.json");
    // If binding tier < 1 and class requires it, mark not lawful
    for (const auto& req : ir.requires_) {
        if (req.find("bindingTier") != std::string::npos && ir.bindingTier < 1) {
            ir.lawful = false;
        }
    }
    return true;
}
