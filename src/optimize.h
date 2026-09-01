#pragma once
#include "kxc.h"

// SCXQ7: legality + caps-aware optimization pass.
// Mutates ir in place. Returns false on a fatal constraint violation.
bool optimize(KernelIR& ir, std::string& err);
