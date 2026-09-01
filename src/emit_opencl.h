#pragma once
#include "kxc.h"
#include <string>
bool emit_opencl(const KernelIR& ir, const std::string& outPath, std::string& err);
