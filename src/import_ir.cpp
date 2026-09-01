#include "import_ir.h"
#include <fstream>
#include <sstream>

static std::string json_str_ir(const std::string& json, const std::string& key) {
    std::string pat = "\"" + key + "\"";
    auto pos = json.find(pat);
    if (pos == std::string::npos) return {};
    pos = json.find(':', pos + pat.size());
    if (pos == std::string::npos) return {};
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return {};
    auto end = json.find('"', pos + 1);
    return end == std::string::npos ? std::string{} : json.substr(pos + 1, end - pos - 1);
}

bool import_ir(const std::string& path, KernelIR& out, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open IR file: " + path; return false; }
    std::ostringstream ss; ss << f.rdbuf();
    std::string json = ss.str();

    out.desc.name    = json_str_ir(json, "kernel");
    out.kernelClass  = json_str_ir(json, "kernelClass");
    out.collapseClass = json_str_ir(json, "collapseClass");
    if (out.desc.name.empty() || out.kernelClass.empty()) {
        err = "IR file missing required fields"; return false;
    }
    out.registryMatched = (json.find("\"registryMatched\": true") != std::string::npos);
    out.lawful          = (json.find("\"lawful\": true") != std::string::npos);
    return true;
}
