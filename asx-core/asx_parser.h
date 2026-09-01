#pragma once

#include <string>

#include "asx_value.h"

namespace asx {

Value parse_document(const std::string& source);
Value parse_file(const std::string& path);

}  // namespace asx
