#pragma once
#include "kxc.h"
#include <string>
bool emit_dml(const KernelIR& ir, const std::string& outPath, std::string& err);
