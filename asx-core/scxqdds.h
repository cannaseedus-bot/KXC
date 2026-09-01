#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "asx_value.h"

namespace asx {

constexpr const char* DEFAULT_SCXQDDS_ARTIFACT_PATH = "C:\\public_html\\MX2LM\\codex\\AS-XCFE\\artifacts\\training\\model.scxqdds";

struct SCXQDDSChunk {
    std::uint32_t id = 0;
    std::uint8_t type = 0;
    std::uint32_t offset = 0;
    std::uint32_t length = 0;
    std::uint32_t crc_expected = 0;
    std::uint32_t crc_computed = 0;
    bool crc_ok = false;
};

struct SCXQDDSDecodeResult {
    std::uint8_t version = 0;
    std::uint8_t flags = 0;
    std::vector<SCXQDDSChunk> chunks;
    std::vector<std::uint8_t> reconstructed;
    std::uint32_t file_crc_expected = 0;
    std::uint32_t file_crc_computed = 0;
    bool file_crc_ok = false;
    bool ok = false;
    std::string summary_hash;
    std::string error;  // present only on hard error
};

SCXQDDSDecodeResult decode_scxqdds(const std::vector<std::uint8_t>& bytes);
Value scxqdds_result_to_json(const SCXQDDSDecodeResult& result);

}  // namespace asx

