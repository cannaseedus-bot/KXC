#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "asx_canonical.h"
#include "asx_parser.h"
#include "asx_value.h"
#include "sha256.h"

namespace fs = std::filesystem;

namespace {

struct Finding {
    std::string code;
    std::string detail;
};

struct Report {
    bool ok = true;
    std::vector<Finding> findings;
    std::string envelope_hash;
};

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("unable to read file: " + path.string());
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void add_finding(Report& report, const std::string& code, const std::string& detail) {
    report.ok = false;
    report.findings.push_back({code, detail});
}

std::set<std::string> collect_string_set(const asx::Value& root, const std::string& path) {
    std::set<std::string> out;
    const asx::Value* node = asx::get_path(root, path);
    if (node == nullptr) return out;
    if (!node->is_array()) throw std::runtime_error("expected array at path: " + path);
    for (const auto& item : node->array_value) {
        if (!item.is_string()) throw std::runtime_error("expected string array entry at path: " + path);
        out.insert(item.string_value);
    }
    return out;
}

std::string strip_comments_and_strings(const std::string& cpp) {
    std::string out;
    out.reserve(cpp.size());
    bool in_string = false;
    bool in_line_comment = false;
    bool in_block_comment = false;

    for (std::size_t i = 0; i < cpp.size(); ++i) {
        const char c = cpp[i];
        const char next = (i + 1 < cpp.size()) ? cpp[i + 1] : '\0';

        if (in_line_comment) {
            if (c == '\n') {
                in_line_comment = false;
                out.push_back(c);
            }
            continue;
        }

        if (in_block_comment) {
            if (c == '*' && next == '/') {
                in_block_comment = false;
                ++i;
            }
            continue;
        }

        if (in_string) {
            if (c == '\\' && next != '\0') {
                ++i;
                continue;
            }
            if (c == '"') {
                in_string = false;
            }
            continue;
        }

        if (c == '/' && next == '/') {
            in_line_comment = true;
            ++i;
            continue;
        }
        if (c == '/' && next == '*') {
            in_block_comment = true;
            ++i;
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }

        out.push_back(c);
    }

    return out;
}

bool contains_symbol(const std::string& text, const std::regex& pattern) {
    return std::regex_search(text, pattern);
}

std::set<std::string> observe_effects_cpp(const std::string& cpp) {
    std::set<std::string> out;
    const std::string cleaned = strip_comments_and_strings(cpp);
    const std::regex random_pattern(R"(\bstd::random_device\b|\bstd::mt19937\b)");
    const std::regex time_pattern(R"(\bstd::chrono::system_clock\b|\bstd::time\s*\()");
    const std::regex fs_read_pattern(R"(\bstd::ifstream\b|\bstd::fstream\b)");
    const std::regex fs_write_pattern(R"(\bstd::ofstream\b)");
    const std::regex net_pattern(R"(\bsocket\s*\(|\bconnect\s*\()");

    if (contains_symbol(cleaned, random_pattern)) out.insert("random.read");
    if (contains_symbol(cleaned, time_pattern)) out.insert("time.read");
    if (contains_symbol(cleaned, fs_read_pattern)) out.insert("fs.read");
    if (contains_symbol(cleaned, fs_write_pattern)) out.insert("fs.write");
    if (contains_symbol(cleaned, net_pattern)) out.insert("net.open");
    return out;
}

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

Report verify_module(const fs::path& envelope_path) {
    Report report;
    const asx::Value document = asx::parse_file(envelope_path.string());
    report.envelope_hash = asx::sha256_hex(asx::canonical_json(document));

    const asx::Value* module = asx::get_child(document, "module.envelope");
    if (module == nullptr || !module->is_object()) {
        throw std::runtime_error("missing @@module.envelope block");
    }

    const std::string& target_relative = asx::require_string(*module, "@module.target.path");
    const std::string& declared_hash = asx::require_string(*module, "@module.target.hash.value");
    const std::string& hash_algo = asx::require_string(*module, "@module.target.hash.algo");
    if (hash_algo != "sha256") throw std::runtime_error("unsupported hash algorithm: " + hash_algo);

    const fs::path target_path = fs::weakly_canonical(envelope_path.parent_path() / target_relative);
    const std::string actual_hash = asx::sha256_file_hex(target_path.string());
    if (declared_hash != actual_hash) {
        add_finding(report, "HASH_MISMATCH", "declared=" + declared_hash + " actual=" + actual_hash);
    }

    const std::set<std::string> forbidden = collect_string_set(*module, "@capabilities.forbids");
    const std::set<std::string> fs_declared = collect_string_set(*module, "@effects.fs");
    const std::set<std::string> net_declared = collect_string_set(*module, "@effects.net");
    const std::set<std::string> time_declared = collect_string_set(*module, "@effects.time");
    const std::set<std::string> random_declared = collect_string_set(*module, "@effects.random");

    const std::string cpp = read_text(target_path);
    const std::set<std::string> observed = observe_effects_cpp(cpp);

    for (const std::string& effect : observed) {
        if (forbidden.count(effect) > 0) add_finding(report, "FORBIDDEN_AUTHORITY", effect);
        if (effect == "fs.read" && fs_declared.count("read") == 0) add_finding(report, "UNDECLARED_EFFECT", effect);
        if (effect == "fs.write" && fs_declared.count("write") == 0) add_finding(report, "UNDECLARED_EFFECT", effect);
        if (effect == "net.open" && net_declared.count("open") == 0) add_finding(report, "UNDECLARED_EFFECT", effect);
        if (effect == "time.read" && time_declared.count("read") == 0) add_finding(report, "UNDECLARED_EFFECT", effect);
        if (effect == "random.read" && random_declared.count("read") == 0) add_finding(report, "UNDECLARED_EFFECT", effect);
    }

    return report;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: asx_verifier <module.asx>\n";
        return 2;
    }

    try {
        const Report report = verify_module(argv[1]);
        std::cout << "{\n";
        std::cout << "  \"@kind\": \"asx.verifier.result.v1\",\n";
        std::cout << "  \"@ok\": " << (report.ok ? "true" : "false") << ",\n";
        std::cout << "  \"@envelope_hash\": \"" << report.envelope_hash << "\",\n";
        std::cout << "  \"@findings\": [\n";
        for (std::size_t i = 0; i < report.findings.size(); ++i) {
            const auto& finding = report.findings[i];
            std::cout << "    {\"code\":\"" << escape_json(finding.code) << "\",\"detail\":\"" << escape_json(finding.detail) << "\"}";
            if (i + 1 != report.findings.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "  ]\n";
        std::cout << "}\n";
        return report.ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "{\n";
        std::cerr << "  \"@kind\": \"asx.verifier.result.v1\",\n";
        std::cerr << "  \"@ok\": false,\n";
        std::cerr << "  \"@error\": \"" << escape_json(error.what()) << "\"\n";
        std::cerr << "}\n";
        return 1;
    }
}
