#pragma once

#include <string>

namespace asx {

std::string sha256_hex(const std::string& input);
std::string sha256_file_hex(const std::string& path);

}  // namespace asx
