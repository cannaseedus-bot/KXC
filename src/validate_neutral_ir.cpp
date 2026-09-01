#include "validate_neutral_ir.h"

bool validate_neutral_ir(const KernelIR& ir, std::string& err) {
    if (ir.desc.name.empty()) { err = "IR: kernel name is empty"; return false; }
    for (int i = 0; i < 3; ++i)
        if (ir.desc.threads[i] == 0) { err = "IR: thread dim[" + std::to_string(i) + "] is zero"; return false; }
    if (ir.kernelClass.empty()) { err = "IR: kernelClass not set"; return false; }
    return true;
}
