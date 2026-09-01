#pragma once

#include <cstddef>
#include <cstdint>

namespace asx {

std::uint32_t crc32(const std::uint8_t* data, std::size_t length);

}  // namespace asx

