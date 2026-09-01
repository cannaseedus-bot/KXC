#include "scxqdds.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>

#include "asx_canonical.h"
#include "crc32.h"
#include "sha256.h"

namespace asx {

namespace {

std::uint32_t read_u32_le(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    if (offset + 4 > bytes.size()) throw std::runtime_error("unexpected EOF while reading u32");
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

std::uint32_t decode_varint_u32(const std::vector<std::uint8_t>& bytes, std::size_t& offset) {
    std::uint32_t result = 0;
    int shift = 0;
    while (offset < bytes.size()) {
        const std::uint8_t b = bytes[offset++];
        result |= static_cast<std::uint32_t>(b & 0x7fu) << shift;
        if ((b & 0x80u) == 0u) return result;
        shift += 7;
        if (shift > 28) throw std::runtime_error("varint too large");
    }
    throw std::runtime_error("unexpected EOF while decoding varint");
}

std::string u32_to_string(std::uint32_t value) {
    std::ostringstream out;
    out << value;
    return out.str();
}

std::string u8_to_hex(std::uint8_t value) {
    static const char* hex = "0123456789abcdef";
    std::string out;
    out.push_back(hex[(value >> 4) & 0x0f]);
    out.push_back(hex[value & 0x0f]);
    return out;
}

std::string build_summary_hash(const SCXQDDSDecodeResult& result) {
    std::vector<std::string> seeds;
    seeds.reserve(result.chunks.size() + 2);
    for (const auto& c : result.chunks) {
        std::ostringstream line;
        line << "CHUNK|" << c.id << "|" << static_cast<unsigned>(c.type) << "|" << c.offset << "|" << c.length << "|"
             << (c.crc_ok ? "1" : "0");
        seeds.push_back(line.str());
    }
    {
        std::ostringstream line;
        line << "FILECRC|" << (result.file_crc_ok ? "1" : "0");
        seeds.push_back(line.str());
    }
    {
        std::ostringstream line;
        line << "LEN|" << result.reconstructed.size();
        seeds.push_back(line.str());
    }
    std::sort(seeds.begin(), seeds.end());
    std::string seed_text;
    for (const auto& s : seeds) seed_text += s + "\n";
    return "sha256:" + sha256_hex(seed_text);
}

}  // namespace

SCXQDDSDecodeResult decode_scxqdds(const std::vector<std::uint8_t>& bytes) {
    SCXQDDSDecodeResult out;
    try {
        if (bytes.size() < 4 + 1 + 1 + 1 + 4) throw std::runtime_error("file too small for SQDS container");
        if (!(bytes[0] == 'S' && bytes[1] == 'Q' && bytes[2] == 'D' && bytes[3] == 'S')) {
            throw std::runtime_error("invalid magic (expected SQDS)");
        }

        std::size_t offset = 4;
        out.version = bytes[offset++];
        out.flags = bytes[offset++];
        const std::uint32_t chunk_count = decode_varint_u32(bytes, offset);

        struct Header {
            std::uint32_t id;
            std::uint8_t type;
            std::uint32_t off;
            std::uint32_t len;
            std::uint32_t crc;
        };

        std::vector<Header> headers;
        headers.reserve(chunk_count);
        for (std::uint32_t i = 0; i < chunk_count; ++i) {
            const std::uint32_t id = decode_varint_u32(bytes, offset);
            if (offset >= bytes.size()) throw std::runtime_error("unexpected EOF while reading chunk type");
            const std::uint8_t type = bytes[offset++];
            const std::uint32_t off = decode_varint_u32(bytes, offset);
            const std::uint32_t len = decode_varint_u32(bytes, offset);
            const std::uint32_t crc = read_u32_le(bytes, offset);
            offset += 4;
            headers.push_back({id, type, off, len, crc});
        }

        const std::size_t payload_start = offset;
        const std::size_t payload_end = bytes.size() - 4;
        if (payload_end < payload_start) throw std::runtime_error("invalid SQDS payload bounds");

        std::size_t reconstructed_length = 0;
        for (const auto& h : headers) {
            const std::size_t end = static_cast<std::size_t>(h.off) + static_cast<std::size_t>(h.len);
            if (end > reconstructed_length) reconstructed_length = end;
        }
        out.reconstructed.assign(reconstructed_length, 0);

        std::size_t cursor = payload_start;
        bool ok = true;
        out.chunks.reserve(headers.size());
        std::vector<std::uint8_t> write_marks(out.reconstructed.size(), 0);

        for (const auto& h : headers) {
            if (cursor + static_cast<std::size_t>(h.len) > payload_end) {
                throw std::runtime_error("unexpected EOF while reading chunk payload");
            }
            const std::uint8_t* payload_ptr = bytes.data() + cursor;
            const std::size_t payload_len = h.len;
            cursor += payload_len;

            const std::uint32_t computed = crc32(payload_ptr, payload_len);
            const bool crc_ok = (computed == h.crc);
            if (!crc_ok) ok = false;

            if (static_cast<std::size_t>(h.off) + payload_len > out.reconstructed.size()) {
                throw std::runtime_error("chunk copy exceeds reconstructed buffer");
            }
            bool overlaps = false;
            for (std::size_t i = 0; i < payload_len; ++i) {
                const std::size_t idx = static_cast<std::size_t>(h.off) + i;
                if (write_marks[idx] != 0) overlaps = true;
                write_marks[idx] = 1;
            }
            if (overlaps) ok = false;
            std::copy(payload_ptr, payload_ptr + payload_len, out.reconstructed.begin() + h.off);

            SCXQDDSChunk chunk;
            chunk.id = h.id;
            chunk.type = h.type;
            chunk.offset = h.off;
            chunk.length = h.len;
            chunk.crc_expected = h.crc;
            chunk.crc_computed = computed;
            chunk.crc_ok = crc_ok;
            out.chunks.push_back(chunk);
        }

        if (cursor != payload_end) {
            // Extra bytes indicate malformed container. Treat as validation failure, but not a hard error.
            ok = false;
        }

        out.file_crc_expected = read_u32_le(bytes, bytes.size() - 4);
        out.file_crc_computed = crc32(bytes.data(), bytes.size() - 4);
        out.file_crc_ok = (out.file_crc_expected == out.file_crc_computed);
        if (!out.file_crc_ok) ok = false;

        out.ok = ok;
        out.summary_hash = build_summary_hash(out);
        return out;
    } catch (const std::exception& e) {
        out.ok = false;
        out.error = e.what();
        out.summary_hash = "";
        return out;
    }
}

Value scxqdds_result_to_json(const SCXQDDSDecodeResult& result) {
    Value root = Value::make_object();
    root.object_value["@kind"] = Value::make_string("scxqdds.decode.result.v1");
    root.object_value["@ok"] = Value::make_bool(result.error.empty());
    if (!result.error.empty()) {
        root.object_value["@error"] = Value::make_string(result.error);
        return root;
    }

    root.object_value["@valid"] = Value::make_bool(result.ok);
    root.object_value["version"] = Value::make_number(result.version);
    root.object_value["flags"] = Value::make_number(result.flags);
    root.object_value["chunk_count"] = Value::make_number(static_cast<double>(result.chunks.size()));

    Value chunks = Value::make_array();
    for (const auto& c : result.chunks) {
        Value item = Value::make_object();
        item.object_value["id"] = Value::make_number(c.id);
        item.object_value["type"] = Value::make_string("0x" + u8_to_hex(c.type));
        item.object_value["offset"] = Value::make_number(c.offset);
        item.object_value["length"] = Value::make_number(c.length);
        item.object_value["crc_expected"] = Value::make_string(u32_to_string(c.crc_expected));
        item.object_value["crc_computed"] = Value::make_string(u32_to_string(c.crc_computed));
        item.object_value["crc_ok"] = Value::make_bool(c.crc_ok);
        chunks.array_value.push_back(item);
    }
    root.object_value["chunks"] = chunks;

    root.object_value["reconstructed_length"] = Value::make_number(static_cast<double>(result.reconstructed.size()));
    root.object_value["file_crc_expected"] = Value::make_string(u32_to_string(result.file_crc_expected));
    root.object_value["file_crc_computed"] = Value::make_string(u32_to_string(result.file_crc_computed));
    root.object_value["file_crc_ok"] = Value::make_bool(result.file_crc_ok);
    root.object_value["@summary_hash"] = Value::make_string(result.summary_hash);
    return root;
}

}  // namespace asx
