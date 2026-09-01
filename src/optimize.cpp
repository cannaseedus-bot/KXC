#include "optimize.h"
#include <algorithm>

bool optimize(KernelIR& ir, std::string& err) {
    // caps legality: waveOps forbidden on this stack
    if (ir.waveOps) {
        err = "kernel '" + ir.desc.name + "': waveOps not available on this caps profile";
        return false;
    }
    // thread total must be <= 1024 (D3D11 cs_5_0 / D3D12 limit)
    uint64_t total = (uint64_t)ir.desc.threads[0]
                   * ir.desc.threads[1]
                   * ir.desc.threads[2];
    if (total > 1024) {
        err = "thread group size " + std::to_string(total) + " exceeds 1024";
        return false;
    }
    // forbids check
    for (const auto& f : ir.forbids) {
        if (f == "waveOps" && ir.waveOps) {
            err = "kernel '" + ir.desc.name + "' forbids waveOps";
            return false;
        }
    }
    return true;
}
