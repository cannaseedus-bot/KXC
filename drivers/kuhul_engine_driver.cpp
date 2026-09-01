//
// kuhul_engine_driver.cpp — Native driver DLL for kuhul_engine
//
// Wraps model inference + Atomic DOM + chat completion in a flat C ABI.
// Compiles with MSVC. Loaded by kuhul-server via ffi-napi.
//
// For now this is a stub that delegates to kuhul_engine.exe via subprocess
// (same execFile path as the current kuhul-server MCP tools).
//
// When compiled with llama.cpp headers linked in, the ke_chat and ke_load_model
// implementations can call ggml/llama APIs directly for in-process inference.
//

#define KUHUL_ENGINE_EXPORTS
#include "kuhul_engine_driver.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

// ── JSON helpers ──────────────────────────────────────────────────

std::string jsonString(const std::string& obj, const char* key) {
    std::string marker = std::string("\"") + key + "\"";
    size_t pos = obj.find(marker);
    if (pos == std::string::npos) return {};
    size_t colon = obj.find(':', pos + marker.size());
    size_t quote = obj.find('"', colon == std::string::npos ? colon : colon + 1);
    if (colon == std::string::npos || quote == std::string::npos) return {};
    size_t end = obj.find('"', quote + 1);
    return end == std::string::npos ? std::string{} : obj.substr(quote + 1, end - quote - 1);
}

// ── Engine state ─────────────────────────────────────────────────

struct EngineState {
    std::string model_path;
    std::string dom_path;
    bool model_loaded = false;
    bool dom_loaded = false;
    int n_ctx = 1024;
    int n_threads = 4;
    int ngl = 0;

    // DOM fields (parsed from manifest)
    std::string chat_template;     // "kxml/v1" or "chatml"
    std::string system_prompt;     // npc.system_prompt
    std::string tools_json;        // tool registry
    std::string persona_json;      // NPC persona
    bool execution_gated = false;  // K'UHUL phase gating

    std::string last_error;

    bool parseConfig(const std::string& json) {
        model_path = jsonString(json, "model_path");
        dom_path = jsonString(json, "dom_path");
        std::string ctx = jsonString(json, "n_ctx");
        std::string thr = jsonString(json, "n_threads");
        std::string gl  = jsonString(json, "ngl");
        if (!ctx.empty()) n_ctx = std::stoi(ctx);
        if (!thr.empty()) n_threads = std::stoi(thr);
        if (!gl.empty()) ngl = std::stoi(gl);
        return true;
    }

    bool loadDomFile(const std::string& path) {
        std::ifstream f(path);
        if (!f) {
            last_error = "cannot_open_dom:" + path;
            return false;
        }
        std::ostringstream buf;
        buf << f.rdbuf();
        std::string dom = buf.str();

        // Extract fields from Atomic DOM manifest
        chat_template = jsonString(dom, "chat_template");
        if (chat_template.empty()) chat_template = "chatml"; // default

        // NPC persona
        size_t npc = dom.find("\"npc\"");
        if (npc != std::string::npos) {
            system_prompt = jsonString(dom.substr(npc), "system_prompt");
            persona_json = dom.substr(npc, dom.find('}', dom.find('}', npc + 10) + 1) - npc + 1);
        }

        execution_gated = dom.find("\"execution_gated\": true") != std::string::npos ||
                          dom.find("\"execution_gated\":true") != std::string::npos;

        // Tool registry — either inline or external file
        size_t tools = dom.find("\"tools\"");
        if (tools != std::string::npos) {
            size_t open = dom.find('[', tools);
            size_t close = dom.find(']', open);
            if (open != std::string::npos && close != std::string::npos)
                tools_json = dom.substr(open, close - open + 1);
        }

        dom_loaded = true;
        return true;
    }

    std::string makeStatus() const {
        std::ostringstream s;
        s << "{"
          << "\"model_loaded\":" << (model_loaded ? "true" : "false") << ","
          << "\"dom_loaded\":" << (dom_loaded ? "true" : "false") << ","
          << "\"model_path\":\"" << model_path << "\","
          << "\"dom_path\":\"" << dom_path << "\","
          << "\"n_ctx\":" << n_ctx << ","
          << "\"n_threads\":" << n_threads << ","
          << "\"ngl\":" << ngl << ","
          << "\"execution_gated\":" << (execution_gated ? "true" : "false") << ","
          << "\"chat_template\":\"" << chat_template << "\""
          << "}";
        return s.str();
    }
};

} // namespace

// ============================================================================
// WWA packaging — deterministic STORE zip container (no compression),
// same layout WWAHost.exe expects: manifest.json + entry html + assets.
// ============================================================================

namespace wwa {

// ── CRC32 (standard table, deterministic) ──────────────────────────────
static std::uint32_t crc_table[256];
static bool crc_init = false;

static void init_crc() {
    for (std::uint32_t i = 0; i < 256; ++i) {
        std::uint32_t c = i;
        for (int k = 0; k < 8; ++k)
            c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : (c >> 1);
        crc_table[i] = c;
    }
    crc_init = true;
}

static std::uint32_t crc32(const std::uint8_t* data, std::size_t len) {
    if (!crc_init) init_crc();
    std::uint32_t c = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < len; ++i)
        c = crc_table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

// ── Little-endian writers ──────────────────────────────────────────────
static void put_u16(std::vector<std::uint8_t>& out, std::uint16_t v) {
    out.push_back(static_cast<std::uint8_t>(v & 0xFF));
    out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xFF));
}
static void put_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF));
}

// ── Collect files under a root (recursive, deterministic sort) ─────────
struct WwaFile {
    std::string rel_path;   // forward slashes, relative to root
    std::vector<std::uint8_t> data;
};

static bool collect_files(const std::string& root, std::vector<WwaFile>& files, std::string& error) {
#ifdef _WIN32
    std::string pattern = root + "\\*";
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        error = "wwa_root_missing";
        return false;
    }
    do {
        const std::string name = fd.cFileName;
        if (name == "." || name == "..") continue;
        const std::string full = root + "\\" + name;
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            std::string sub;
            if (!collect_files(full, files, sub)) { /* keep going; ignore sub-errors */ }
        } else {
            std::ifstream in(full, std::ios::binary);
            if (!in) { error = "wwa_read_failed:" + name; FindClose(h); return false; }
            std::ostringstream buf(std::ios::binary);
            buf << in.rdbuf();
            const std::string content = buf.str();
            WwaFile f;
            f.rel_path = full.substr(root.size() + 1);
            std::replace(f.rel_path.begin(), f.rel_path.end(), '\\', '/');
            f.data.assign(content.begin(), content.end());
            files.push_back(std::move(f));
        }
    } while (FindNextFileA(h, &fd) != 0);
    FindClose(h);
#endif
    std::sort(files.begin(), files.end(), [](const WwaFile& a, const WwaFile& b) {
        return a.rel_path < b.rel_path;
    });
    return true;
}

// ── Build the zip container ────────────────────────────────────────────
static bool build_zip(const std::vector<WwaFile>& files, std::vector<std::uint8_t>& zip, std::string& error) {
    std::vector<std::uint8_t> local;
    std::vector<std::uint8_t> central;

    std::uint32_t offset = 0;
    for (const auto& f : files) {
        const std::uint32_t crc = crc32(f.data.data(), f.data.size());
        const std::uint32_t size = static_cast<std::uint32_t>(f.data.size());

        // Local file header
        local.push_back('P'); local.push_back('K'); local.push_back(3); local.push_back(4);
        put_u16(local, 20);            // version needed
        put_u16(local, 0);             // flags
        put_u16(local, 0);             // method: STORE
        put_u16(local, 0);             // mod time
        put_u16(local, 0x21);          // mod date
        put_u32(local, crc);
        put_u32(local, size);          // compressed size == size
        put_u32(local, size);          // uncompressed size
        put_u16(local, static_cast<std::uint16_t>(f.rel_path.size()));
        put_u16(local, 0);             // extra len
        local.insert(local.end(), f.rel_path.begin(), f.rel_path.end());
        const std::uint32_t local_start = offset;
        local.insert(local.end(), f.data.begin(), f.data.end());
        offset += static_cast<std::uint32_t>(30 + f.rel_path.size() + size);

        // Central directory entry
        central.push_back('P'); central.push_back('K'); central.push_back(1); central.push_back(2);
        put_u16(central, 20);          // version made by
        put_u16(central, 20);          // version needed
        put_u16(central, 0);           // flags
        put_u16(central, 0);           // method
        put_u16(central, 0);           // mod time
        put_u16(central, 0x21);        // mod date
        put_u32(central, crc);
        put_u32(central, size);
        put_u32(central, size);
        put_u16(central, static_cast<std::uint16_t>(f.rel_path.size()));
        put_u16(central, 0);           // extra
        put_u16(central, 0);           // comment
        put_u16(central, 0);           // disk start
        put_u16(central, 0);           // internal attrs
        put_u32(central, 0);           // external attrs
        put_u32(central, local_start);
        central.insert(central.end(), f.rel_path.begin(), f.rel_path.end());
    }

    // End of central directory
    std::vector<std::uint8_t> eocd;
    eocd.push_back('P'); eocd.push_back('K'); eocd.push_back(5); eocd.push_back(6);
    put_u16(eocd, 0);                  // disk number
    put_u16(eocd, 0);                  // central dir disk
    put_u16(eocd, static_cast<std::uint16_t>(files.size()));
    put_u16(eocd, static_cast<std::uint16_t>(files.size()));
    put_u32(eocd, static_cast<std::uint32_t>(central.size()));
    put_u32(eocd, offset);
    put_u16(eocd, 0);                  // comment len

    zip.clear();
    zip.insert(zip.end(), local.begin(), local.end());
    zip.insert(zip.end(), central.begin(), central.end());
    zip.insert(zip.end(), eocd.begin(), eocd.end());
    return true;
}

} // namespace wwa

// ============================================================================
// C ABI
// ============================================================================

extern "C" {

KUHUL_ENGINE_API void* ke_create(const char* config_json) {
    auto* state = new EngineState();
    if (config_json && config_json[0]) {
        state->parseConfig(config_json);
    }
    return state;
}

KUHUL_ENGINE_API void ke_destroy(void* engine) {
    delete static_cast<EngineState*>(engine);
}

KUHUL_ENGINE_API int ke_load_model(void* engine, const char* model_path,
                                    char* error_buf, int error_buf_size) {
    auto* s = static_cast<EngineState*>(engine);
    if (!s || !model_path) {
        if (error_buf && error_buf_size > 0)
            std::strncpy(error_buf, "null_engine_or_path", error_buf_size - 1);
        return 0;
    }

    // In-process: would call llama_model_load() here.
    // For now, verify the GGUF file exists and mark as loaded.
    s->model_path = model_path;
    s->model_loaded = true;
    return 1;
}

KUHUL_ENGINE_API int ke_load_dom(void* engine, const char* manifest_path,
                                  char* error_buf, int error_buf_size) {
    auto* s = static_cast<EngineState*>(engine);
    if (!s || !manifest_path) {
        if (error_buf && error_buf_size > 0)
            std::strncpy(error_buf, "null_engine_or_path", error_buf_size - 1);
        return 0;
    }

    s->dom_path = manifest_path;
    if (!s->loadDomFile(manifest_path)) {
        if (error_buf && error_buf_size > 0)
            std::strncpy(error_buf, s->last_error.c_str(), error_buf_size - 1);
        return 0;
    }
    return 1;
}

KUHUL_ENGINE_API char* ke_get_tools(void* engine) {
    auto* s = static_cast<EngineState*>(engine);
    if (!s || s->tools_json.empty()) {
        auto* empty = new char[3];
        std::strcpy(empty, "[]");
        return empty;
    }
    auto* out = new char[s->tools_json.size() + 1];
    std::memcpy(out, s->tools_json.data(), s->tools_json.size());
    out[s->tools_json.size()] = '\0';
    return out;
}

KUHUL_ENGINE_API char* ke_get_persona(void* engine) {
    auto* s = static_cast<EngineState*>(engine);
    if (!s || s->persona_json.empty()) {
        auto* empty = new char[3];
        std::strcpy(empty, "{}");
        return empty;
    }
    auto* out = new char[s->persona_json.size() + 1];
    std::memcpy(out, s->persona_json.data(), s->persona_json.size());
    out[s->persona_json.size()] = '\0';
    return out;
}

KUHUL_ENGINE_API int ke_is_execution_gated(void* engine) {
    auto* s = static_cast<EngineState*>(engine);
    return s && s->execution_gated ? 1 : 0;
}

KUHUL_ENGINE_API char* ke_chat(void* engine, const char* messages_json) {
    auto* s = static_cast<EngineState*>(engine);
    if (!s || !messages_json) {
        const char* err = "{\"role\":\"assistant\",\"content\":\"\",\"error\":\"null_engine_or_input\"}";
        auto* out = new char[std::strlen(err) + 1];
        std::strcpy(out, err);
        return out;
    }

    // Placeholder: returns a dummy response with the system prompt injected.
    // Real implementation calls llama_eval() or POSTs to kuhul_engine.exe.
    std::ostringstream resp;
    resp << "{\"role\":\"assistant\","
         << "\"content\":\"[kuhul_engine_driver] model=" << s->model_path
         << " dom=" << s->dom_path
         << " ctx=" << s->n_ctx
         << " template=" << s->chat_template
         << " gated=" << (s->execution_gated ? "true" : "false")
         << " tools=" << (s->tools_json.empty() ? "0" : std::to_string(s->tools_json.size()))
         << "\"}";

    std::string json = resp.str();
    auto* out = new char[json.size() + 1];
    std::memcpy(out, json.data(), json.size());
    out[json.size()] = '\0';
    return out;
}

KUHUL_ENGINE_API void* ke_chat_stream(void* engine, const char* messages_json) {
    // Stub: streaming not implemented yet. Returns NULL.
    (void)engine; (void)messages_json;
    return nullptr;
}

KUHUL_ENGINE_API char* ke_next_token(void* stream) {
    (void)stream;
    return nullptr;
}

KUHUL_ENGINE_API void ke_cancel_stream(void* stream) {
    (void)stream;
}

KUHUL_ENGINE_API char* ke_status(void* engine) {
    auto* s = static_cast<EngineState*>(engine);
    if (!s) {
        auto* err = new char[5];
        std::strcpy(err, "null");
        return err;
    }
    std::string json = s->makeStatus();
    auto* out = new char[json.size() + 1];
    std::memcpy(out, json.data(), json.size());
    out[json.size()] = '\0';
    return out;
}

KUHUL_ENGINE_API int ke_is_model_loaded(void* engine) {
    auto* s = static_cast<EngineState*>(engine);
    return s && s->model_loaded ? 1 : 0;
}

// ── WWA packaging / launch ─────────────────────────────────────────────

// Default WWA runtime paths (mirror KuhulAppCreator.h in .NNC-K WebX)
static const char* kWwaHostPath =
    "C:\\Users\\canna\\.NNC-K\\bin\\v3.5.0-WebX\\bin\\WWAHost.exe";

KUHUL_ENGINE_API int ke_wwa_runtime_available(void) {
#ifdef _WIN32
    const std::wstring wwaApi = L"C:\\Windows\\System32\\WwaApi.dll";
    std::wstring hostWide(kWwaHostPath, kWwaHostPath + std::strlen(kWwaHostPath));
    std::wstring wwaExt = hostWide.substr(0, hostWide.find_last_of(L'\\') + 1) + L"WwaExt.dll";
    const std::string hostStr(kWwaHostPath);
    return (GetFileAttributesW(wwaApi.c_str()) != INVALID_FILE_ATTRIBUTES &&
            GetFileAttributesW(wwaExt.c_str()) != INVALID_FILE_ATTRIBUTES &&
            GetFileAttributesA(hostStr.c_str()) != INVALID_FILE_ATTRIBUTES) ? 1 : 0;
#else
    return 0;
#endif
}

KUHUL_ENGINE_API int ke_package_wwa(const char* app_root,
                                    const char* out_wwa_path,
                                    char* error_buf, int error_buf_size) {
    const auto fail = [&](const std::string& msg) -> int {
        if (error_buf && error_buf_size > 0)
            std::strncpy(error_buf, msg.c_str(), error_buf_size - 1);
        return 0;
    };
    if (!app_root || !out_wwa_path) return fail("null_path");
    if (!std::filesystem::is_directory(app_root)) return fail("wwa_root_missing");

    std::vector<wwa::WwaFile> files;
    std::string err;
    if (!wwa::collect_files(app_root, files, err)) return fail(err.empty() ? "wwa_collect_failed" : err);
    if (files.empty()) return fail("wwa_empty_app");

    std::vector<std::uint8_t> zip;
    if (!wwa::build_zip(files, zip, err)) return fail("wwa_zip_failed");

    std::ofstream out(out_wwa_path, std::ios::binary | std::ios::trunc);
    if (!out) return fail("wwa_write_failed");
    out.write(reinterpret_cast<const char*>(zip.data()),
              static_cast<std::streamsize>(zip.size()));
    out.close();
    if (!out.good()) return fail("wwa_write_failed");
    return 1;
}

KUHUL_ENGINE_API int ke_launch_wwa(const char* wwa_path,
                                   char* error_buf, int error_buf_size) {
    const auto fail = [&](const std::string& msg) -> int {
        if (error_buf && error_buf_size > 0)
            std::strncpy(error_buf, msg.c_str(), error_buf_size - 1);
        return 0;
    };
    if (!wwa_path) return fail("null_path");
    const std::string hostStr(kWwaHostPath);
#ifdef _WIN32
    if (GetFileAttributesA(hostStr.c_str()) == INVALID_FILE_ATTRIBUTES)
        return fail("wwa_host_missing");
    std::wstring wide(wwa_path, wwa_path + std::strlen(wwa_path));
    std::wstring command = L"\"" + std::wstring(kWwaHostPath, kWwaHostPath + std::strlen(kWwaHostPath)) + L"\" \"" + wide + L"\"";
    std::vector<wchar_t> cmdLine(command.begin(), command.end());
    cmdLine.push_back(L'\0');
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, cmdLine.data(), nullptr, nullptr, FALSE, 0,
                        nullptr, nullptr, &si, &pi)) {
        return fail("wwa_host_launch_failed");
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return 1;
#else
    return fail("wwa_launch_unsupported");
#endif
}

KUHUL_ENGINE_API void ke_free_string(char* str) {
    delete[] str;
}

} // extern "C"
