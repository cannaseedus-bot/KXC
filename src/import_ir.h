#pragma once
#include "kxc.h"

// Import a KernelIR from a previously emitted .ir.json file.
bool import_ir(const std::string& path, KernelIR& out, std::string& err);
