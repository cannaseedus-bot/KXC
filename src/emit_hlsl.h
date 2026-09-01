#pragma once
#include "kxc.h"
bool emit_hlsl(const KernelIR& ir, const std::string& outPath, std::string& err);
