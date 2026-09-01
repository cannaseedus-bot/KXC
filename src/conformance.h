#pragma once
#include "kxc.h"

// SMCA conformance check: verify requires/forbids against caps profile.
bool conformance_check(const KernelIR& ir, std::string& err);
