#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "asx_canonical.h"
#include "scxqdds.h"

namespace fs = std::filesystem;

static std::vector<std::uint8_t> read_file_bytes(const fs::path& p) {
    std::ifstream file(p, std::ios::binary);
    if (!file) throw std::runtime_error("failed to open file");
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "usage: scxqdds_decode <file.sqds>\n";
        return 2;
    }

    try {
        const char* target = (argc == 2) ? argv[1] : asx::DEFAULT_SCXQDDS_ARTIFACT_PATH;
        const fs::path path = fs::path(target);
        const auto bytes = read_file_bytes(path);
        const asx::SCXQDDSDecodeResult decoded = asx::decode_scxqdds(bytes);

        const asx::Value json = asx::scxqdds_result_to_json(decoded);
        std::cout << asx::canonical_json(json) << "\n";

        if (!decoded.error.empty()) return 2;
        return decoded.ok ? 0 : 1;
    } catch (const std::exception& e) {
        asx::Value root = asx::Value::make_object();
        root.object_value["@kind"] = asx::Value::make_string("scxqdds.decode.result.v1");
        root.object_value["@ok"] = asx::Value::make_bool(false);
        root.object_value["@error"] = asx::Value::make_string(e.what());
        std::cerr << asx::canonical_json(root) << "\n";
        return 2;
    }
}

