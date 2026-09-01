/**
 * klsl_compiler.cpp
 * KLSL → HLSL / XVM compiler.
 *
 * Two-pass line-oriented approach:
 *   Pass 1 — collect shader metadata, buffer declarations, entry point name
 *   Pass 2 — emit HLSL (or XVM bytecode)
 */

#include "klsl_compiler.h"
#include "klsl_opcodes.h"
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <cassert>

namespace klsl {

// ─────────────────────────────────────────────────────────────────────────────
// Glyph constants (UTF-8 byte strings)
// ⟁ = U+27C1 → E2 9F 81
// ' = ASCII 0x27
// ─────────────────────────────────────────────────────────────────────────────

static const char G_SHADER_PFX[] = "\xe2\x9f\x81 shader ";     // ⟁ shader
static const char G_XUL[]        = "\xe2\x9f\x81Xul\xe2\x9f\x81";  // ⟁Xul⟁
static const char G_WO[]         = "\xe2\x9f\x81Wo\xe2\x9f\x81 ";  // ⟁Wo⟁ (trailing space)
static const char G_SEK[]        = "\xe2\x9f\x81Sek\xe2\x9f\x81 "; // ⟁Sek⟁
static const char G_CHEN[]       = "\xe2\x9f\x81" "Ch'en\xe2\x9f\x81 "; // ⟁Ch'en⟁
static const char G_YAX[]        = "\xe2\x9f\x81Yax\xe2\x9f\x81 ";   // ⟁Yax⟁
static const char G_KAYAB[]      = "\xe2\x9f\x81K'ayab'\xe2\x9f\x81 "; // ⟁K'ayab'⟁
static const char G_KUMKU[]      = "\xe2\x9f\x81Kumk'u\xe2\x9f\x81";  // ⟁Kumk'u⟁
static const char POP_PFX[]      = "[Pop ";
static const char XUL_KW[]       = "[Xul]";

// ─────────────────────────────────────────────────────────────────────────────
// String utilities
// ─────────────────────────────────────────────────────────────────────────────

static bool sw(const std::string& s, const char* pfx) {
    size_t n = strlen(pfx);
    return s.size() >= n && s.compare(0, n, pfx) == 0;
}

static std::string trimL(std::string s) {
    size_t i = 0;
    while (i < s.size() && (unsigned char)s[i] <= ' ') ++i;
    return s.substr(i);
}
static std::string trimR(std::string s) {
    while (!s.empty() && (unsigned char)s.back() <= ' ') s.pop_back();
    return s;
}
static std::string trim(const std::string& s) { return trimR(trimL(s)); }

// Text after prefix (prefix already matched), left-trimmed
static std::string after(const std::string& s, const char* pfx) {
    return trimL(s.substr(strlen(pfx)));
}

// Find matching closing paren for s[openIdx]=='('
static size_t matchParen(const std::string& s, size_t openIdx) {
    if (openIdx >= s.size() || s[openIdx] != '(') return std::string::npos;
    int d = 1;
    for (size_t i = openIdx + 1; i < s.size(); ++i) {
        if (s[i] == '(') ++d;
        else if (s[i] == ')') { if (--d == 0) return i; }
    }
    return std::string::npos;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal data types
// ─────────────────────────────────────────────────────────────────────────────

enum class BufKind { Structured, RWStructured, Constant, ByteAddress };

static bool isBufKw(const std::string& tok) {
    return tok == "StructuredBuffer"   || tok == "RWStructuredBuffer" ||
           tok == "ConstantBuffer"     || tok == "ByteAddressBuffer";
}
static BufKind bufKindOf(const std::string& t) {
    if (t == "StructuredBuffer")   return BufKind::Structured;
    if (t == "RWStructuredBuffer") return BufKind::RWStructured;
    if (t == "ConstantBuffer")     return BufKind::Constant;
    return BufKind::ByteAddress;
}
static const char* bufKindStr(BufKind k) {
    switch (k) {
        case BufKind::Structured:   return "StructuredBuffer";
        case BufKind::RWStructured: return "RWStructuredBuffer";
        case BufKind::Constant:     return "ConstantBuffer";
        default:                    return "ByteAddressBuffer";
    }
}

struct BufDecl {
    BufKind     kind;
    std::string elemType;
    std::string name;
    std::string reg;   // "t0", "u1", "b0", etc.
};

struct ShaderMeta {
    std::string name;
    std::string stage  = "compute";
    int tx = 64, ty = 1, tz = 1;
};

struct Compiler {
    std::vector<std::string> lines;
    size_t       li    = 0;
    std::string  fname;
    CompileOptions opts;

    ShaderMeta           meta;
    std::vector<BufDecl> buffers;
    std::string          entryName;

    std::ostringstream hlsl;
    int depth = 0;           // brace nesting depth

    bool        ok      = true;
    std::string errMsg;
    int         errLine = 0;

    void error(const std::string& msg) {
        if (!ok) return;
        ok      = false;
        errMsg  = fname + ":" + std::to_string((int)li + 1) + ": " + msg;
        errLine = (int)li + 1;
    }
    std::string ind() const { return std::string((size_t)depth * 4, ' '); }
};

// ─────────────────────────────────────────────────────────────────────────────
// Buffer declaration parser
// payload: "StructuredBuffer<float> myBuf : register(t0)"
// ─────────────────────────────────────────────────────────────────────────────

static bool parseBufDecl(const std::string& payload, BufDecl& out) {
    size_t lt = payload.find('<');
    if (lt == std::string::npos) return false;
    std::string kw = trim(payload.substr(0, lt));
    if (!isBufKw(kw)) return false;
    out.kind = bufKindOf(kw);

    size_t gt = payload.find('>', lt + 1);
    if (gt == std::string::npos) return false;
    out.elemType = trim(payload.substr(lt + 1, gt - lt - 1));

    std::string rest = trim(payload.substr(gt + 1));
    size_t colon = rest.find(':');
    if (colon == std::string::npos) { out.name = trim(rest); return true; }
    out.name = trim(rest.substr(0, colon));

    std::string rp = trim(rest.substr(colon + 1));
    // register(tN)
    const char rPfx[] = "register(";
    if (sw(rp, rPfx)) {
        size_t rEnd = rp.find(')', strlen(rPfx));
        out.reg = trim(rp.substr(strlen(rPfx), rEnd == std::string::npos
                                                 ? std::string::npos
                                                 : rEnd - strlen(rPfx)));
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// For-loop head parser
// payload: "uint k in 0 .. cb.in_dim"  →  "for (uint k = 0; k < cb.in_dim; k++)"
// ─────────────────────────────────────────────────────────────────────────────

static std::string parseLoop(const std::string& payload) {
    size_t inPos = payload.find(" in ");
    if (inPos == std::string::npos)
        return "for (/* KLSL: " + payload + " */)";

    std::string varPart   = trim(payload.substr(0, inPos));
    std::string rangePart = trim(payload.substr(inPos + 4));

    size_t dd = rangePart.find("..");
    if (dd == std::string::npos)
        return "for (/* KLSL: " + payload + " */)";

    std::string start = trimR(rangePart.substr(0, dd));
    std::string end_  = trimL(rangePart.substr(dd + 2));

    // varPart: "uint k" or just "k"
    std::istringstream vs(varPart);
    std::string t1, t2;
    vs >> t1;
    if (vs >> t2) {
        return "for (" + t1 + " " + t2 + " = " + start + "; " +
               t2 + " < " + end_ + "; ++" + t2 + ")";
    } else {
        return "for (uint " + t1 + " = " + start + "; " +
               t1 + " < " + end_ + "; ++" + t1 + ")";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SV_ semantic parameter builder
// ─────────────────────────────────────────────────────────────────────────────

static std::string buildEntryParams(const std::vector<std::string>& body) {
    bool dtid = false, gtid = false, gid = false;
    for (auto& l : body) {
        if (l.find("SV_DispatchThreadID") != std::string::npos) dtid = true;
        if (l.find("SV_GroupThreadID")    != std::string::npos) gtid = true;
        if (l.find("SV_GroupID")          != std::string::npos) gid  = true;
    }
    std::string p;
    auto sep = [&]() -> const char* { return p.empty() ? "" : ",\n                "; };
    if (dtid) p += std::string(sep()) + "uint3 SV_DispatchThreadID : SV_DispatchThreadID";
    if (gtid) p += std::string(sep()) + "uint3 SV_GroupThreadID    : SV_GroupThreadID";
    if (gid)  p += std::string(sep()) + "uint3 SV_GroupID          : SV_GroupID";
    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// Statement translators
// ─────────────────────────────────────────────────────────────────────────────

// ⟁Wo⟁ payload inside a function  →  HLSL statement
static std::string xlateWo(const std::string& p) {
    std::string s = trimR(p);
    if (!s.empty() && s.back() == ';') s.pop_back();
    // Slice notation a[x..y] → leave with comment
    if (s.find("..") != std::string::npos)
        return "/* KLSL slice (expand manually): " + s + " */";
    return s + ";";
}

// ⟁Sek⟁ payload inside a function  →  HLSL statement
// Returns {hlsl_text, opens_block}
static std::pair<std::string,bool> xlateSek(const std::string& p) {
    std::string s = trim(p);

    // return [expr]
    if (sw(s, "return")) {
        std::string rest = trim(s.substr(6));
        return {rest.empty() ? "return;" : "return " + rest + ";", false};
    }

    // if (cond) [body]
    if (sw(s, "if ") || sw(s, "if(")) {
        size_t op = s.find('(');
        if (op != std::string::npos) {
            size_t cp = matchParen(s, op);
            if (cp != std::string::npos) {
                std::string cond = s.substr(0, cp + 1);   // "if (cond)"
                std::string body = trim(s.substr(cp + 1));
                if (!body.empty()) {
                    // Inline body: "if (cond) return;" etc.
                    auto [bhlsl, bopen] = xlateSek(body);
                    return {cond + " " + bhlsl, bopen};
                }
                // No inline body — caller opens block
                return {cond + " {", true};
            }
        }
        return {s + " {", true};
    }

    // for (...)
    if (sw(s, "for ") || sw(s, "for(")) {
        return {s + " {", true};
    }

    // while (...)
    if (sw(s, "while ") || sw(s, "while(")) {
        return {s + " {", true};
    }

    // bare expression / assignment
    if (!s.empty() && s.back() == ';') s.pop_back();
    return {s + ";", false};
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1: collect metadata + buffers + entry point name
// ─────────────────────────────────────────────────────────────────────────────

static void pass1(Compiler& c) {
    bool inShader = false, inFunc = false;
    for (c.li = 0; c.li < c.lines.size(); ++c.li) {
        std::string ln = trim(c.lines[c.li]);
        if (ln.empty() || sw(ln, "//") || sw(ln, "/*")) continue;

        if (sw(ln, G_XUL)) { inShader = false; continue; }

        if (sw(ln, G_SHADER_PFX) && !inShader) {
            c.meta.name = trim(ln.substr(strlen(G_SHADER_PFX)));
            inShader = true; continue;
        }
        if (!inShader) continue;

        // Metadata / buffer declarations (outside functions)
        if (!inFunc && sw(ln, G_WO)) {
            std::string pay = after(ln, G_WO);

            if (sw(pay, "stage ")) {
                std::string st = after(pay, "stage ");
                if (!st.empty() && st.front() == '"') st = st.substr(1);
                if (!st.empty() && st.back()  == '"') st.pop_back();
                c.meta.stage = st;
                continue;
            }

            if (sw(pay, "threads ")) {
                std::string th = after(pay, "threads ");
                size_t lb = th.find('['), rb = th.find(']');
                if (lb != std::string::npos && rb != std::string::npos) {
                    std::string inner = th.substr(lb + 1, rb - lb - 1);
                    std::istringstream ss(inner);
                    char comma;
                    ss >> c.meta.tx >> comma >> c.meta.ty >> comma >> c.meta.tz;
                }
                continue;
            }

            BufDecl bd;
            if (parseBufDecl(pay, bd)) { c.buffers.push_back(bd); continue; }
            // Other top-level Wo: ignore (custom metadata)
            continue;
        }

        // Function boundaries
        if (sw(ln, POP_PFX)) {
            std::string name = trim(ln.substr(strlen(POP_PFX)));
            if (!name.empty() && name.back() == ']') name.pop_back();
            if (c.entryName.empty()) c.entryName = name;
            inFunc = true; continue;
        }
        if (sw(ln, XUL_KW)) { inFunc = false; continue; }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Collect entry-point body lines (for SV_ param detection)
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<std::string> collectEntryBody(const Compiler& c) {
    std::vector<std::string> body;
    bool scanning = false;
    for (auto& raw : c.lines) {
        std::string ln = trim(raw);
        if (sw(ln, POP_PFX)) {
            std::string name = trim(ln.substr(strlen(POP_PFX)));
            if (!name.empty() && name.back() == ']') name.pop_back();
            scanning = (name == c.entryName);
            continue;
        }
        if (sw(ln, XUL_KW)) { scanning = false; continue; }
        if (scanning) body.push_back(ln);
    }
    return body;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2: emit HLSL
// ─────────────────────────────────────────────────────────────────────────────

static void pass2(Compiler& c) {
    auto& out = c.hlsl;
    out << "// Generated by KLSL compiler v0.1\n";
    out << "// Shader: " << c.meta.name << "\n\n";

    // Emit buffer declarations
    for (auto& bd : c.buffers) {
        out << bufKindStr(bd.kind) << "<" << bd.elemType << "> " << bd.name;
        if (!bd.reg.empty()) out << " : register(" << bd.reg << ")";
        out << ";\n";
    }
    if (!c.buffers.empty()) out << "\n";

    // Detect SV_ params from entry body
    auto entryBody   = collectEntryBody(c);
    std::string params = buildEntryParams(entryBody);

    // Override entry if requested
    std::string entryName = c.opts.entryOverride.empty()
                                ? c.entryName
                                : c.opts.entryOverride;

    bool inShader = false, inFunc = false;
    std::string currentFunc;

    for (c.li = 0; c.li < c.lines.size(); ++c.li) {
        std::string raw = c.lines[c.li];
        std::string ln  = trim(raw);

        // Blank / comment pass-through
        if (ln.empty()) { out << "\n"; continue; }
        if (sw(ln, "//")) { out << c.ind() << ln << "\n"; continue; }
        if (sw(ln, "/*")) { out << c.ind() << ln << "\n"; continue; }

        // ⟁Xul⟁  (end shader)
        if (sw(ln, G_XUL)) { inShader = false; continue; }

        // ⟁ shader <name>
        if (sw(ln, G_SHADER_PFX) && !inShader) { inShader = true; continue; }
        if (!inShader) continue;

        // Top-level Wo / Sek / Ch'en / Yax — already handled in pass1 output
        if (!inFunc && (sw(ln, G_WO) || sw(ln, G_SEK) ||
                        sw(ln, G_CHEN) || sw(ln, G_YAX)))
            continue;

        // [Pop <name>]
        if (sw(ln, POP_PFX)) {
            std::string name = trim(ln.substr(strlen(POP_PFX)));
            if (!name.empty() && name.back() == ']') name.pop_back();
            currentFunc = name;
            inFunc      = true;

            bool isEntry = (name == entryName) && (c.meta.stage == "compute");
            if (isEntry) {
                out << "[numthreads("
                    << c.meta.tx << ", " << c.meta.ty << ", " << c.meta.tz
                    << ")]\n";
                out << "void " << name << "(" << params << ")\n{\n";
            } else {
                out << "void " << name << "()\n{\n";
            }
            c.depth = 1;
            continue;
        }

        // [Xul]
        if (sw(ln, XUL_KW)) {
            if (inFunc) {
                while (c.depth > 1) {
                    --c.depth;
                    out << std::string((size_t)c.depth * 4, ' ') << "}\n";
                }
                out << "}\n\n";
                c.depth = 0;
                inFunc  = false;
            }
            continue;
        }

        if (!inFunc) continue;

        // ─── Inside function ────────────────────────────────────────────

        // ⟁Wo⟁ — variable declaration / assignment
        if (sw(ln, G_WO)) {
            out << c.ind() << xlateWo(after(ln, G_WO)) << "\n";
            continue;
        }

        // ⟁Sek⟁ — statement / control flow
        if (sw(ln, G_SEK)) {
            auto [hlslLine, opens] = xlateSek(after(ln, G_SEK));
            out << c.ind() << hlslLine << "\n";
            if (opens) ++c.depth;
            continue;
        }

        // ⟁Ch'en⟁ — assignment target (context-dependent; emit as comment)
        if (sw(ln, G_CHEN)) {
            out << c.ind() << "// [→ " << after(ln, G_CHEN) << "]\n";
            continue;
        }

        // ⟁Yax⟁ — load variable (emit as comment)
        if (sw(ln, G_YAX)) {
            out << c.ind() << "// [← " << after(ln, G_YAX) << "]\n";
            continue;
        }

        // ⟁K'ayab'⟁ — for loop start
        if (sw(ln, G_KAYAB)) {
            out << c.ind() << parseLoop(after(ln, G_KAYAB)) << " {\n";
            ++c.depth;
            continue;
        }

        // ⟁Kumk'u⟁ — for loop end / close brace
        if (sw(ln, G_KUMKU)) {
            if (c.depth > 1) {
                --c.depth;
                out << c.ind() << "}\n";
            }
            continue;
        }

        // Plain HLSL line (embedded directly)
        out << c.ind() << ln << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

// XVM bytecode generation (pass3)
static void pass3_xvm(const Compiler& c, CompileResult& res) {
    std::vector<uint8_t> pool;   // const pool bytes
    std::vector<uint8_t> ibuf;   // instruction bytes
    uint32_t const_count = 0;
    uint32_t instr_count = 0;

    // Helper: push varint to buffer
    auto push_varint = [](std::vector<uint8_t>& buf, uint32_t v) {
        do {
            uint8_t b = static_cast<uint8_t>(v & 0x7fu);
            v >>= 7;
            if (v) b |= 0x80u;
            buf.push_back(b);
        } while (v);
    };

    // Helper: push string const to pool
    auto push_str_const = [&push_varint](std::vector<uint8_t>& pool, const std::string& s) {
        pool.push_back(0x01u);  // tag: string
        push_varint(pool, static_cast<uint32_t>(s.size()));
        pool.insert(pool.end(), s.begin(), s.end());
    };

    // const[0]: shader name
    push_str_const(pool, c.meta.name.empty() ? "shader" : c.meta.name);
    ++const_count;

    // const[1..N]: buffer names
    for (auto& bd : c.buffers) {
        push_str_const(pool, bd.name);
        ++const_count;
    }

    // Count [Pop] blocks (one Emit instruction per block)
    uint32_t func_count = 0;
    for (auto& raw : c.lines) {
        if (sw(trim(raw), POP_PFX)) ++func_count;
    }
    if (func_count == 0) func_count = 1;  // always emit at least one

    // Helper: push instruction to buffer
    auto push_instr = [&push_varint](std::vector<uint8_t>& ibuf,
                                      uint8_t packed,
                                      const std::vector<uint32_t>& args,
                                      int32_t target = -1) {
        ibuf.push_back(packed);
        push_varint(ibuf, static_cast<uint32_t>(args.size()));
        for (auto a : args) push_varint(ibuf, a);
        if (target >= 0) push_varint(ibuf, static_cast<uint32_t>(target));
    };

    // Emit instructions: one Emit per function block, then Return
    for (uint32_t i = 0; i < func_count; ++i) {
        // op=0x07 (Emit) requires extra target varint; argc=0, target=0 (shader name const)
        push_instr(ibuf, 0x07u, {}, 0);
        ++instr_count;
    }
    // Return: op=0x3F, argc=0, no target
    push_instr(ibuf, 0x3Fu, {});
    ++instr_count;

    // Helper: CRC32 calculation
    auto crc32 = [](const uint8_t* data, size_t len) -> uint32_t {
        uint32_t c = 0xffffffffu;
        for (size_t i = 0; i < len; ++i) {
            c ^= static_cast<uint32_t>(data[i]);
            for (int j = 0; j < 8; ++j)
                c = (c >> 1) ^ (0xedb88320u & static_cast<uint32_t>(
                        -(static_cast<int32_t>(c & 1u))));
        }
        return c ^ 0xffffffffu;
    };

    // Build the CRC window: varint(const_count) + varint(instr_count) + pool + ibuf
    std::vector<uint8_t> body;
    push_varint(body, const_count);
    push_varint(body, instr_count);
    body.insert(body.end(), pool.begin(), pool.end());
    body.insert(body.end(), ibuf.begin(), ibuf.end());

    const uint32_t crc = crc32(body.data(), body.size());

    // Assemble full SCX2 file
    std::vector<uint8_t> out;
    out.reserve(6 + body.size() + 4);
    out.push_back('S'); out.push_back('C'); out.push_back('X'); out.push_back('2');
    out.push_back(0x01u);  // version
    out.push_back(0x00u);  // flags
    out.insert(out.end(), body.begin(), body.end());
    out.push_back(static_cast<uint8_t>( crc        & 0xFFu));
    out.push_back(static_cast<uint8_t>((crc >>  8) & 0xFFu));
    out.push_back(static_cast<uint8_t>((crc >> 16) & 0xFFu));
    out.push_back(static_cast<uint8_t>((crc >> 24) & 0xFFu));

    res.xvm = std::move(out);
}

CompileResult compile(const std::string& source,
                      const std::string& filename,
                      const CompileOptions& opts)
{
    Compiler c;
    c.fname = filename.empty() ? "<klsl>" : filename;
    c.opts  = opts;

    std::istringstream ss(source);
    std::string ln;
    while (std::getline(ss, ln)) c.lines.push_back(ln);

    pass1(c);
    if (c.ok && opts.emitHLSL) pass2(c);

    CompileResult res;
    res.ok        = c.ok;
    res.errorLine = c.errLine;
    res.errorMsg  = c.errMsg;
    res.hlsl      = c.hlsl.str();

    if (c.ok && opts.emitXVM) pass3_xvm(c, res);

    return res;
}

} // namespace klsl
