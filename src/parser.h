#pragma once
#include "kxc.h"
#include <string>

// Parse a .kuhul kernel descriptor file → KernelDesc.
// Returns false and writes an error to `err` on failure.
bool parse_kuhul(const std::string& path, KernelDesc& out, std::string& err);
