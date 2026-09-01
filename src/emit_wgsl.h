#pragma once
#include "kxc.h"
bool emit_wgsl(const KernelIR& ir, const std::string& outPath, std::string& err);
