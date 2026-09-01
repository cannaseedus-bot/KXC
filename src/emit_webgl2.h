#pragma once
#include "kxc.h"
#include <string>
bool emit_webgl2(const KernelIR& ir, const std::string& outPath, std::string& err);
