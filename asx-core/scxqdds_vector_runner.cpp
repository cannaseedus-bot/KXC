#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "asx_canonical.h"
#include "asx_parser.h"
#include "asx_value.h"
#include "scxqdds.h"
#include "sha256.h"

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
    std::string expect = "FAIL";  // PASS | FAIL | ERROR
};

std::vector<VectorCase> load_cases(const asx::Value& root) {
    const asx::Value* cases = asx::get_path(root, "cases");
    if (cases == nullptr || !cases->is_array()) throw std::runtime_error("expected cases array");
    std::vector<VectorCase> out;
    for (const auto& item : cases->array_value) {
        if (!item.is_object()) throw std::runtime_error("case entry must be object");
        VectorCase c;
        c.id = asx::require_string(item, "@id");
        c.file = asx::require_string(item, "file");
        const asx::Value* ev = asx::get_path(item, "expect");
        if (ev != nullptr && ev->is_string()) {
            c.expect = ev->string_value;
        } else {
            const asx::Value* legacy = asx::get_path(item, "expect_valid");
            if (legacy == nullptr || legacy->type != asx::ValueType::Bool) {
                throw std::runtime_error("expect must be string or expect_valid must be boolean");
            }
            c.expect = legacy->bool_value ? "PASS" : "FAIL";
        }
        if (c.expect != "PASS" && c.expect != "FAIL" && c.expect != "ERROR") {
            throw std::runtime_error("expect must be PASS, FAIL, or ERROR");
        }
        out.push_back(std::move(c));
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "usage: scxqdds_vector_runner <vectors.asx>\n";
        return 2;
    }

    try {
        const char* target = (argc == 2) ? argv[1] : "C:\\public_html\\MX2LM\\codex\\AS-XCFE\\artifacts\\training\\scxqdds_vectors.asx";
        const fs::path vector_path = fs::path(target);
        const fs::path vector_dir = vector_path.has_parent_path() ? vector_path.parent_path() : fs::path(".");

        const asx::Value document = asx::parse_file(vector_path.string());
        const asx::Value* root = asx::get_child(document, "scxqdds.vectors");
        if (root == nullptr || !root->is_object()) throw std::runtime_error("missing @@scxqdds.vectors block");

        const std::string vector_id = asx::require_string(*root, "@id");
        const auto cases = load_cases(*root);

        int pass_count = 0;
        int fail_count = 0;
        int mismatch_count = 0;
        std::vector<std::string> detail_hash_parts;

        std::cout << "{\n";
        std::cout << "  \"@kind\": \"scxqdds.vector.summary.v1\",\n";
        std::cout << "  \"@ok\": true,\n";
        std::cout << "  \"@id\": \"" << escape_json(vector_id) << "\",\n";
        std::cout << "  \"@cases\": [\n";

        for (std::size_t i = 0; i < cases.size(); ++i) {
            const auto& entry = cases[i];
            const fs::path resolved = safe_resolve_under(vector_dir, entry.file);
            const std::string hex = read_text_file(resolved);
            const auto bytes = parse_hex_bytes(hex);
            const auto decoded = asx::decode_scxqdds(bytes);

            const bool hard_error = !decoded.error.empty();
            const bool actual_valid = (!hard_error) && decoded.ok;
            const std::string actual = hard_error ? "ERROR" : (actual_valid ? "PASS" : "FAIL");
            const bool matched = (actual == entry.expect);

            if (hard_error) {
                ++fail_count;
                if (!matched) ++mismatch_count;
            } else if (actual_valid) {
                ++pass_count;
                if (!matched) ++mismatch_count;
            } else {
                ++fail_count;
                if (!matched) ++mismatch_count;
            }

            const std::string expected = entry.expect;

            detail_hash_parts.push_back(entry.id + "|" + resolved.generic_string() + "|" + actual + "|" + expected + "|" + decoded.summary_hash);

            std::cout << "    {\"@id\":\"" << escape_json(entry.id)
                      << "\",\"@file\":\"" << escape_json(entry.file)
                      << "\",\"@actual\":\"" << actual
                      << "\",\"@expected\":\"" << expected
                      << "\",\"@matched\":" << (matched ? "true" : "false")
                      << ",\"@summary_hash\":\"" << escape_json(decoded.summary_hash) << "\"";
            if (hard_error) {
                std::cout << ",\"@error\":\"" << escape_json(decoded.error) << "\"";
            }
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
        std::cout << "  \"@counts\": {\"pass\": " << pass_count << ", \"fail\": " << fail_count << ", \"mismatch\": " << mismatch_count << "},\n";
        std::cout << "  \"@verdict\": \"" << verdict << "\",\n";
        std::cout << "  \"@summary_hash\": \"" << summary_hash << "\"\n";
        std::cout << "}\n";

        return mismatch_count == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "{\n";
        std::cerr << "  \"@kind\": \"scxqdds.vector.summary.v1\",\n";
        std::cerr << "  \"@ok\": false,\n";
        std::cerr << "  \"@error\": \"" << escape_json(e.what()) << "\"\n";
        std::cerr << "}\n";
        return 2;
    }
}
