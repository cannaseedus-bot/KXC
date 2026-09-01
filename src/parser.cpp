#include "parser.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

// ── helpers ───────────────────────────────────────────────────────────────────
static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static bool parse_bool(const std::string& v) {
    return v == "true" || v == "1" || v == "yes";
}

// Tokenise a bracket expression: "[Token arg0 arg1 ...]" → vector of strings
static std::vector<std::string> tokenise_bracket(const std::string& line) {
    std::string inner = line;
    // strip outer [ ]
    auto lb = inner.find('[');
    auto rb = inner.rfind(']');
    if (lb == std::string::npos || rb == std::string::npos) return {};
    inner = inner.substr(lb + 1, rb - lb - 1);
    std::vector<std::string> toks;
    std::istringstream ss(inner);
    std::string tok;
    while (ss >> tok) toks.push_back(tok);
    return toks;
}

// ── public ────────────────────────────────────────────────────────────────────
bool parse_kuhul(const std::string& path, KernelDesc& out, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open: " + path; return false; }

    bool in_block = false;
    std::string line;
    int lineno = 0;

    while (std::getline(f, line)) {
        ++lineno;
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        auto toks = tokenise_bracket(line);
        if (toks.empty()) continue;

        const std::string& kw = toks[0];

        if (kw == "Pop") {
            if (toks.size() < 2) { err = "line " + std::to_string(lineno) + ": [Pop] missing name"; return false; }
            if (in_block) { err = "line " + std::to_string(lineno) + ": nested [Pop] not allowed"; return false; }
            out = KernelDesc{};
            out.name = toks[1];
            in_block = true;

        } else if (kw == "Muwan") {
            if (!in_block) { err = "line " + std::to_string(lineno) + ": [Muwan] outside block"; return false; }
            if (toks.size() >= 5 && toks[1] == "dispatch") {
                out.threads[0] = static_cast<uint32_t>(std::stoul(toks[2]));
                out.threads[1] = static_cast<uint32_t>(std::stoul(toks[3]));
                out.threads[2] = static_cast<uint32_t>(std::stoul(toks[4]));
            }

        } else if (kw == "Sek") {
            if (!in_block) { err = "line " + std::to_string(lineno) + ": [Sek] outside block"; return false; }
            if (toks.size() < 3) { err = "line " + std::to_string(lineno) + ": [Sek] needs property and value"; return false; }
            const std::string& prop = toks[1];
            const std::string& val  = toks[2];
            if      (prop == "needsDecompress")    out.needsDecompress    = parse_bool(val);
            else if (prop == "needsSoftmax")        out.needsSoftmax       = parse_bool(val);
            else if (prop == "needsMatMul")         out.needsMatMul        = parse_bool(val);
            else if (prop == "kvInt4")              out.kvInt4             = parse_bool(val);
            else if (prop == "needsMoERoute")       out.needsMoERoute      = parse_bool(val);
            else if (prop == "needsMoEExpertFFN")   out.needsMoEExpertFFN  = parse_bool(val);
            else if (prop == "needsMoECombine")     out.needsMoECombine    = parse_bool(val);
            else if (prop == "needsPhaseMatch")     out.needsPhaseMatch    = parse_bool(val);
            // unknown Sek properties are silently accepted

        } else if (kw == "Yax") {
            // supported syntax, no action needed at parse level
        } else if (kw == "Xul") {
            if (!in_block) { err = "line " + std::to_string(lineno) + ": [Xul] without [Pop]"; return false; }
            in_block = false;
            // single-kernel files: return after first complete block
            return true;

        } else {
            err = "line " + std::to_string(lineno) + ": unsupported syntax '" + kw + "'";
            return false;
        }
    }

    if (in_block) { err = "unexpected EOF: missing [Xul]"; return false; }
    err = "no kernel found in file";
    return false;
}
