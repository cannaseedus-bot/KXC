#include <iostream>
#include <unordered_map>
#include <functional>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <filesystem>
#include <windows.h>
#include "glyph.h"
#include "glyph_backend_abi.h"

using namespace std;
namespace fs = std::filesystem;

static unordered_map<uint32_t, function<void(GlyphEntry*, GlyphEntry*, void*)>> opcode_table;
static const char* GGML_WEBGPU_CPP = "C:\\Users\\canna\\.powernaut\\.llama.cpp\\ggml\\src\\ggml-webgpu\\ggml-webgpu.cpp";
static const char* GGML_WEBGPU_SHADER_LIB = "C:\\Users\\canna\\.powernaut\\.llama.cpp\\ggml\\src\\ggml-webgpu\\ggml-webgpu-shader-lib.hpp";
static const char* GGML_WEBGPU_WGSL_DIR = "C:\\Users\\canna\\.powernaut\\.llama.cpp\\ggml\\src\\ggml-webgpu\\wgsl-shaders";
static const char* GLYPH_GGML_LANES = "C:\\Users\\canna\\.powernaut\\tools\\Kuhul-c++\\glyph_ggml_lanes.json";

void register_opcode(uint32_t code, function<void(GlyphEntry*, GlyphEntry*, void*)> fn) {
    opcode_table[code] = fn;
}

void sample_det_noun_opcode(GlyphEntry* in, GlyphEntry* out, void* mem) {
    // simple compose: copy noun codepoint and set glyphType to NP code
    out->codepoint = in[1].codepoint; // assume in[1] is noun
    out->glyphType = 0x1001; // NP prototype
    memset(out->features, 0, sizeof(out->features));
    cout << "[native] Composed DET + NOUN -> NP (prototype)\n";
}

void print_path_status(const string& label, const fs::path& path) {
    cout << " - " << label << ": " << path.string();
    if (fs::exists(path)) {
        cout << " [exists";
        if (fs::is_regular_file(path)) {
            cout << ", " << fs::file_size(path) << " bytes";
        }
        cout << "]\n";
    } else {
        cout << " [missing]\n";
    }
}

int inspect_webgpu_backend() {
    cout << "WebGPU glyph backend surfaces\n";
    print_path_status("ggml-webgpu.cpp", GGML_WEBGPU_CPP);
    print_path_status("ggml-webgpu-shader-lib.hpp", GGML_WEBGPU_SHADER_LIB);
    print_path_status("wgsl-shaders", GGML_WEBGPU_WGSL_DIR);
    print_path_status("glyph_ggml_lanes.json", GLYPH_GGML_LANES);

    const vector<string> shader_names = {
        "binary.wgsl",
        "mul_mat.wgsl",
        "soft_max.wgsl",
        "rms_norm_mul.wgsl",
        "rope.wgsl",
        "flash_attn.wgsl",
        "unary.wgsl",
        "concat.wgsl"
    };

    cout << "Mapped WGSL lane shaders:\n";
    for (const auto& name : shader_names) {
        print_path_status(name, fs::path(GGML_WEBGPU_WGSL_DIR) / name);
    }

    cout << "Mode: descriptor/reference only. The prototype does not link ggml-webgpu.cpp yet.\n";
    return 0;
}

const char* lane_kind_name(uint32_t lane_kind) {
    switch (lane_kind) {
        case KUHUL_GLYPH_LANE_SCALAR: return "scalar";
        case KUHUL_GLYPH_LANE_TENSOR: return "tensor";
        case KUHUL_GLYPH_LANE_ATTENTION: return "attention";
        case KUHUL_GLYPH_LANE_PHASE: return "phase";
        default: return "unknown";
    }
}

int inspect_abi_lanes() {
    cout << "Glyph backend ABI v" << kuhul_glyph_abi_version() << "\n";
    const uint32_t count = kuhul_glyph_lane_count();
    cout << "Exported lane count: " << count << "\n";
    for (uint32_t i = 0; i < count; ++i) {
        KuhulGlyphLaneDesc desc{};
        const int32_t rc = kuhul_glyph_get_lane(i, &desc);
        if (rc != KUHUL_GLYPH_OK) {
            cout << " - lane[" << i << "] error=" << rc << "\n";
            continue;
        }
        cout << " - lane[" << i << "] opcode=0x" << hex << desc.opcode << dec
             << " name=" << desc.name
             << " glyph=" << desc.glyph
             << " kind=" << lane_kind_name(desc.lane_kind)
             << " arity=" << desc.arity
             << " dtype_mask=0x" << hex << desc.dtype_mask << dec << "\n";
    }
    return 0;
}

int run_lane_demo() {
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float out[4] = {};

    const int32_t rc = kuhul_glyph_execute_f32(KUHUL_GLYPH_WO_ADD, a, b, out, 4);
    if (rc != KUHUL_GLYPH_OK) {
        cerr << "WO_ADD lane failed: " << rc << "\n";
        return rc;
    }

    cout << "WO_ADD f32 lane demo: [";
    for (int i = 0; i < 4; ++i) {
        if (i) cout << ", ";
        cout << out[i];
    }
    cout << "]\n";

    const int32_t silu_rc = kuhul_glyph_execute_f32(KUHUL_GLYPH_WO_SILU, a, nullptr, out, 4);
    if (silu_rc != KUHUL_GLYPH_OK) {
        cerr << "WO_SILU lane failed: " << silu_rc << "\n";
        return silu_rc;
    }

    cout << "WO_SILU f32 lane demo: [";
    for (int i = 0; i < 4; ++i) {
        if (i) cout << ", ";
        cout << out[i];
    }
    cout << "]\n";
    return 0;
}

int run_matmul_demo() {
    float lhs[6] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    float rhs[6] = {
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };
    float out[4] = {};

    const int32_t rc = kuhul_glyph_matmul_f32(lhs, rhs, out, 2, 3, 2);
    if (rc != KUHUL_GLYPH_OK) {
        cerr << "WO_MATMUL lane failed: " << rc << "\n";
        return rc;
    }

    cout << "WO_MATMUL f32 row-major demo (2x3 @ 3x2 -> 2x2):\n";
    cout << "  [" << out[0] << ", " << out[1] << "]\n";
    cout << "  [" << out[2] << ", " << out[3] << "]\n";
    return 0;
}

int process_ipc_mapping(const string &map_name) {
    // Open existing named file mapping
    HANDLE hMap = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, map_name.c_str());
    if (!hMap) {
        cerr << "OpenFileMapping failed: " << GetLastError() << "\n";
        return 2;
    }
    // Map view
    LPVOID view = MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!view) {
        cerr << "MapViewOfFile failed: " << GetLastError() << "\n";
        CloseHandle(hMap);
        return 3;
    }
    // Interpret header
    IPCHeader *hdr = reinterpret_cast<IPCHeader*>(view);
    if (hdr->magic != GRAM_MAGIC) {
        cerr << "Invalid magic in mapping\n";
        UnmapViewOfFile(view); CloseHandle(hMap); return 4;
    }
    cout << "IPC mapping opened. glyphCount=" << hdr->glyphCount << " status=" << hdr->status << "\n";

    // wait until status==1 (ready)
    int wait_ms = 0;
    while (hdr->status != 1 && wait_ms < 10000) { // 10s timeout
        this_thread::sleep_for(chrono::milliseconds(50));
        wait_ms += 50;
    }
    if (hdr->status != 1) {
        cerr << "Timed out waiting for data (status=" << hdr->status << ")\n";
        UnmapViewOfFile(view); CloseHandle(hMap); return 5;
    }

    // read entries
    GlyphEntry* entries = reinterpret_cast<GlyphEntry*>((uint8_t*)view + sizeof(IPCHeader));
    uint32_t n = hdr->glyphCount;
    cout << "Reading " << n << " entries\n";
    for (uint32_t i=0;i<n;i++) {
        cout << " entry["<<i<<"] cp="<<entries[i].codepoint<<" type="<<entries[i].glyphType<<"\n";
    }

    // process: apply sample opcode if available
    auto it = opcode_table.find(0x02000001);
    GlyphEntry result{};
    if (it != opcode_table.end() && n>=2) {
        it->second(&entries[0], &result, nullptr);
    } else {
        // default: copy first
        result = entries[0];
    }

    // write result after entries
    GlyphEntry* result_slot = reinterpret_cast<GlyphEntry*>((uint8_t*)view + sizeof(IPCHeader) + sizeof(GlyphEntry)*n);
    *result_slot = result;

    // mark processed
    hdr->status = 2;

    cout << "Wrote result glyphType="<< result.glyphType <<" codepoint="<< result.codepoint <<"\n";

    // cleanup
    UnmapViewOfFile(view);
    CloseHandle(hMap);
    return 0;
}

int main(int argc, char** argv) {
    cout << "Native Glyph Engine Prototype (IPC-enabled)\n";
    // register a sample opcode
    register_opcode(0x02000001, sample_det_noun_opcode);

    // CLI
    if (argc >= 2) {
        string cmd = argv[1];
        if (cmd == "run-demo") {
            // legacy in-process demo
            GlyphEntry det{}; det.codepoint = 1; det.glyphType = 1;
            GlyphEntry noun{}; noun.codepoint = 2; noun.glyphType = 2;
            GlyphEntry out{};
            auto it = opcode_table.find(0x02000001);
            if (it != opcode_table.end()) {
                GlyphEntry inarr[2] = {det, noun};
                it->second(&inarr[0], &out, nullptr);
                cout << "Native demo completed. Out glyphType=" << out.glyphType << " codepoint=" << out.codepoint << "\n";
                return 0;
            }
            cout << "Opcode not registered\n";
            return 2;
        }
        if (cmd == "inspect") {
            cout << "Opcode table size: " << opcode_table.size() << "\n";
            for (auto &p: opcode_table) cout << " - opcode: 0x" << hex << p.first << dec << "\n";
            return 0;
        }
        if (cmd == "inspect-webgpu") {
            return inspect_webgpu_backend();
        }
        if (cmd == "inspect-abi") {
            return inspect_abi_lanes();
        }
        if (cmd == "run-lane-demo") {
            return run_lane_demo();
        }
        if (cmd == "run-matmul-demo") {
            return run_matmul_demo();
        }
        // new: --mmap-name <name>
        if (cmd == "--mmap-name" && argc >= 3) {
            string name = argv[2];
            return process_ipc_mapping(name);
        }
    }

    cout << "Usage:\n  native_glyph_engine.exe run-demo | run-lane-demo | run-matmul-demo | inspect | inspect-abi | inspect-webgpu | --mmap-name <tagname>\n";
    return 0;
}
