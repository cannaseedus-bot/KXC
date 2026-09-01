#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sha256.h"
#include "../xvm-d3d12/src/scxq2_format.h"

namespace fs = std::filesystem;

namespace {

std::string escape_json(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (char c : input) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

std::string to_lower_ascii(std::string value) {
    for (char& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return value;
}

fs::path safe_resolve_under(const fs::path& base_dir, const std::string& relative_path) {
    const fs::path base = fs::weakly_canonical(base_dir);
    const fs::path candidate = fs::weakly_canonical(base / fs::path(relative_path));

    const std::string base_str = to_lower_ascii(base.lexically_normal().string());
    const std::string cand_str = to_lower_ascii(candidate.lexically_normal().string());
    if (cand_str.rfind(base_str, 0) != 0) throw std::runtime_error("path escapes vectors directory: " + relative_path);
    if (cand_str.size() > base_str.size()) {
        const char boundary = cand_str[base_str.size()];
        if (boundary != '\\' && boundary != '/') throw std::runtime_error("path escapes vectors directory: " + relative_path);
    }
    return candidate;
}

std::string read_text_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open file: " + p.string());
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return s;
}

int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

std::vector<std::uint8_t> parse_hex_bytes(const std::string& text) {
    std::string compact;
    compact.reserve(text.size());
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) continue;
        compact.push_back(c);
    }
    if (compact.size() % 2 != 0) throw std::runtime_error("hex text must have even length");
    std::vector<std::uint8_t> out;
    out.reserve(compact.size() / 2);
    for (std::size_t i = 0; i < compact.size(); i += 2) {
        const int hi = hex_value(compact[i]);
        const int lo = hex_value(compact[i + 1]);
        if (hi < 0 || lo < 0) throw std::runtime_error("invalid hex character");
        out.push_back(static_cast<std::uint8_t>((hi << 4) | lo));
    }
    return out;
}

struct VectorCase {
    std::string id;
    std::string file;
    std::string expect = "PASS";  // PASS | ERROR
};

std::vector<VectorCase> load_cases(const std::string& vector_path) {
    std::ifstream in(vector_path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open vectors file");

    std::vector<VectorCase> cases;
    std::string line;
    VectorCase current;
    bool in_cases = false;
    while (std::getline(in, line)) {
        if (line.find("cases") != std::string::npos) {
            in_cases = true;
            continue;
        }
        if (!in_cases) continue;
        const auto trim = [](std::string s) {
            auto start = s.find_first_not_of(" \t");
            auto end = s.find_last_not_of(" \t\r\n");
            if (start == std::string::npos || end == std::string::npos) return std::string();
            return s.substr(start, end - start + 1);
        };
        line = trim(line);
        if (line.rfind("@id:", 0) == 0) {
            if (!current.id.empty()) cases.push_back(current);
            current = {};
            current.id = trim(line.substr(4));
            if (!current.id.empty() && current.id[0] == '"') {
                current.id = current.id.substr(1, current.id.size() - 2);
            }
        } else if (line.rfind("file:", 0) == 0) {
            current.file = trim(line.substr(5));
            if (!current.file.empty() && current.file[0] == '"') {
                current.file = current.file.substr(1, current.file.size() - 2);
            }
        } else if (line.rfind("expect:", 0) == 0) {
            current.expect = trim(line.substr(7));
            if (!current.expect.empty() && current.expect[0] == '"') {
                current.expect = current.expect.substr(1, current.expect.size() - 2);
            }
        }
    }
    if (!current.id.empty()) cases.push_back(current);
    if (cases.empty()) throw std::runtime_error("no cases found in vectors file");
    return cases;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: scxq2_vector_runner <vectors.asx>\n";
        return 2;
    }

    try {
        const fs::path vector_path = fs::path(argv[1]);
        const fs::path vector_dir = vector_path.has_parent_path() ? vector_path.parent_path() : fs::path(".");

        const auto cases = load_cases(vector_path.string());

        int pass_count = 0;
        int mismatch_count = 0;
        std::vector<std::string> detail_hash_parts;

        std::cout << "{\n";
        std::cout << "  \"@kind\": \"scxq2.vector.summary.v1\",\n";
        std::cout << "  \"@id\": \"scxq2.vectors\",\n";
        std::cout << "  \"@cases\": [\n";

        for (std::size_t i = 0; i < cases.size(); ++i) {
            const auto& entry = cases[i];
            const fs::path resolved = safe_resolve_under(vector_dir, entry.file);
            const std::string hex = read_text_file(resolved);
            const auto bytes = parse_hex_bytes(hex);

            SCXQ2Parsed parsed;
            std::string error;
            const bool ok = scxq2_parse_and_validate(bytes, parsed, error);
            const std::string actual = ok ? "PASS" : "ERROR";
            const bool matched = (actual == entry.expect);
            if (ok) ++pass_count;
            if (!matched) ++mismatch_count;

            const std::string seed = entry.id + "|" + resolved.generic_string() + "|" + actual + "|" + entry.expect +
                                     "|" + std::to_string(parsed.const_count) + "|" + std::to_string(parsed.instr_count) +
                                     "|" + std::to_string(parsed.crc_expected) + "|" + std::to_string(parsed.crc_computed);
            detail_hash_parts.push_back(seed);

            std::cout << "    {\"@id\":\"" << escape_json(entry.id)
                      << "\",\"@file\":\"" << escape_json(entry.file)
                      << "\",\"@actual\":\"" << actual
                      << "\",\"@expected\":\"" << entry.expect
                      << "\",\"@matched\":" << (matched ? "true" : "false")
                      << ",\"@crc\":{\"expected\":" << parsed.crc_expected << ",\"computed\":" << parsed.crc_computed << "}";
            if (!ok) std::cout << ",\"@error\":\"" << escape_json(error) << "\"";
            std::cout << "}";
            if (i + 1 != cases.size()) std::cout << ",";
            std::cout << "\n";
        }

        std::sort(detail_hash_parts.begin(), detail_hash_parts.end());
        std::string seed;
        for (const auto& part : detail_hash_parts) seed += part + "\n";
        const std::string summary_hash = "sha256:" + asx::sha256_hex(seed);
        const std::string verdict = mismatch_count == 0 ? "PASS" : "FAIL";

        std::cout << "  ],\n";
        std::cout << "  \"@counts\": {\"pass\": " << pass_count << ", \"fail\": " << (static_cast<int>(cases.size()) - pass_count)
                  << ", \"mismatch\": " << mismatch_count << "},\n";
        std::cout << "  \"@verdict\": \"" << verdict << "\",\n";
        std::cout << "  \"@summary_hash\": \"" << summary_hash << "\"\n";
        std::cout << "}\n";

        return mismatch_count == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "{\n";
        std::cerr << "  \"@kind\": \"scxq2.vector.summary.v1\",\n";
        std::cerr << "  \"@ok\": false,\n";
        std::cerr << "  \"@error\": \"" << escape_json(e.what()) << "\"\n";
        std::cerr << "}\n";
        return 2;
    }
}
