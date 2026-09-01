#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "asx_canonical.h"
#include "crc32.h"
#include "scxqdds.h"

namespace {

std::vector<std::uint8_t> encode_varint_u32(std::uint32_t value) {
    std::vector<std::uint8_t> out;
    std::uint32_t n = value;
    while (n >= 0x80u) {
        out.push_back(static_cast<std::uint8_t>((n & 0x7fu) | 0x80u));
        n >>= 7;
    }
    out.push_back(static_cast<std::uint8_t>(n));
    return out;
}

void append(std::vector<std::uint8_t>& dst, const std::vector<std::uint8_t>& src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

void append_u32_le(std::vector<std::uint8_t>& dst, std::uint32_t value) {
    dst.push_back(static_cast<std::uint8_t>(value & 0xffu));
    dst.push_back(static_cast<std::uint8_t>((value >> 8) & 0xffu));
    dst.push_back(static_cast<std::uint8_t>((value >> 16) & 0xffu));
    dst.push_back(static_cast<std::uint8_t>((value >> 24) & 0xffu));
}

std::vector<std::uint8_t> build_sample_sqds() {
    const std::vector<std::uint8_t> p0 = {'h', 'e', 'l', 'l', 'o'};
    const std::vector<std::uint8_t> p1 = {'w', 'o', 'r', 'l', 'd'};

    const std::uint32_t crc0 = asx::crc32(p0.data(), p0.size());
    const std::uint32_t crc1 = asx::crc32(p1.data(), p1.size());

    std::vector<std::uint8_t> header;
    header.push_back('S');
    header.push_back('Q');
    header.push_back('D');
    header.push_back('S');
    header.push_back(0x01);  // version
    header.push_back(0x00);  // flags

    append(header, encode_varint_u32(2));  // chunk count

    // chunk 0: id=0, type=0x03, offset=0, length=5, crc=crc0
    append(header, encode_varint_u32(0));
    header.push_back(0x03);
    append(header, encode_varint_u32(0));
    append(header, encode_varint_u32(static_cast<std::uint32_t>(p0.size())));
    append_u32_le(header, crc0);

    // chunk 1: id=1, type=0x02, offset=5, length=5, crc=crc1
    append(header, encode_varint_u32(1));
    header.push_back(0x02);
    append(header, encode_varint_u32(static_cast<std::uint32_t>(p0.size())));
    append(header, encode_varint_u32(static_cast<std::uint32_t>(p1.size())));
    append_u32_le(header, crc1);

    // payloads (concatenated)
    std::vector<std::uint8_t> bytes = header;
    append(bytes, p0);
    append(bytes, p1);

    // trailing file CRC is CRC32 over all bytes so far
    const std::uint32_t file_crc = asx::crc32(bytes.data(), bytes.size());
    append_u32_le(bytes, file_crc);
    return bytes;
}

}  // namespace

int main() {
    try {
        const auto sample = build_sample_sqds();
        const auto a = asx::decode_scxqdds(sample);
        const auto b = asx::decode_scxqdds(sample);

        const asx::Value ja = asx::scxqdds_result_to_json(a);
        const asx::Value jb = asx::scxqdds_result_to_json(b);
        const std::string sa = asx::canonical_json(ja);
        const std::string sb = asx::canonical_json(jb);

        if (!a.error.empty() || !b.error.empty()) {
            std::cerr << sa << "\n";
            return 2;
        }
        if (!a.ok || !b.ok) {
            std::cerr << sa << "\n";
            return 1;
        }
        if (sa != sb || a.summary_hash != b.summary_hash) {
            std::cerr << "{\"@kind\":\"scxqdds.selftest.result.v1\",\"@ok\":false,\"@error\":\"nondeterministic output\"}\n";
            return 1;
        }

        // Emit a deterministic JSON payload (canonical, no timestamps).
        std::cout << sa << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "{\"@kind\":\"scxqdds.selftest.result.v1\",\"@ok\":false,\"@error\":\"" << e.what() << "\"}\n";
        return 2;
    }
}

