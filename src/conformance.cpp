#include "conformance.h"

bool conformance_check(const KernelIR& ir, std::string& err) {
    // forbids: if "waveOps" is forbidden, caps must agree
    for (const auto& f : ir.forbids) {
        if (f == "waveOps" && ir.waveOps) {
            err = "conformance: kernel '" + ir.desc.name + "' forbids waveOps but caps has waveOps=true";
            return false;
        }
    }
    // requires: bindingTier >= 1
    for (const auto& r : ir.requires_) {
        if (r.find("bindingTier") != std::string::npos && ir.bindingTier < 1) {
            err = "conformance: bindingTier requirement not met";
            return false;
        }
    }
    return true;
}
