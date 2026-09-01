#include "crc32.h"

namespace asx {

std::uint32_t crc32(const std::uint8_t* data, std::size_t length) {
    std::uint32_t c = 0xffffffffu;
    for (std::size_t i = 0; i < length; ++i) {
        c ^= static_cast<std::uint32_t>(data[i]);
        for (int j = 0; j < 8; ++j) {
            c = (c >> 1) ^ (0xedb88320u & static_cast<std::uint32_t>(-(static_cast<std::int32_t>(c & 1u))));
        }
    }
    return (c ^ 0xffffffffu);
}

}  // namespace asx

