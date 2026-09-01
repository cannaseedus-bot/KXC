#include "kxc.h"
#include "parser.h"
#include "lower.h"
#include "optimize.h"
#include "validate_neutral_ir.h"
#include "validate_manifest.h"
#include "conformance.h"
#include "emit_hlsl.h"
#include "emit_wgsl.h"
#include "emit_cpp.h"
#include "emit_cpu.h"
#include "emit_opencl.h"
#include "emit_webgl2.h"
#include "emit_dml.h"
#include "emit_manifest.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <cstdio>

namespace fs = std::filesystem;

static void usage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " <source.kuhul> [--outdir <dir>] [--registry <dir>]\n";
}

static std::string file_cid(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "fnv1a64:0000000000000000";
    std::string data((std::istreambuf_iterator<char>(f)), {});
    char buf[48]; snprintf(buf, sizeof(buf), "fnv1a64:%016llx",
                           (unsigned long long)fnv1a64(data.data(), data.size()));
    return buf;
}

static std::string default_stack_manifest() {
    const char* env = std::getenv("KXC_STACK_MANIFEST");
    if (env && *env) return env;
    return "C:/public_html/MX2LM/codex/AS-XCFE/native/xvm-d3d12/asx_manifest.json";
}

static std::string default_registry_dir(const std::string& exeDir) {
    const char* env = std::getenv("KXC_REGISTRY");
    if (env && *env) return env;
    std::string beside = exeDir + "/registry";
    return beside;
}

int main(int argc, char* argv[]) {
    if (argc < 2) { usage(argv[0]); return 1; }

    std::string sourcePath;
    std::string outDir = ".";
    std::string registryDir;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--outdir") == 0 && i + 1 < argc) {
            outDir = argv[++i];
        } else if (strcmp(argv[i], "--registry") == 0 && i + 1 < argc) {
            registryDir = argv[++i];
        } else if (argv[i][0] != '-') {
            sourcePath = argv[i];
        } else {
            std::cerr << "unknown option: " << argv[i] << "\n";
            usage(argv[0]); return 1;
        }
    }

    if (sourcePath.empty()) { usage(argv[0]); return 1; }

    std::string exeDir = fs::path(argv[0]).parent_path().string();
    if (registryDir.empty()) registryDir = default_registry_dir(exeDir);

    CompileCtx ctx;
    ctx.sourcePath    = fs::absolute(sourcePath).string();
    ctx.outDir        = outDir;
    ctx.registryDir   = registryDir;
    ctx.stackManifest = default_stack_manifest();

    std::string err;

    // ── MATRIX: parse ─────────────────────────────────────────────────────────
    KernelDesc desc;
    if (!parse_kuhul(sourcePath, desc, err)) {
        std::cerr << "parse error: " << err << "\n"; return 1;
    }

    // ── SCXQ2: lower ──────────────────────────────────────────────────────────
    KernelIR ir;
    if (!lower(desc, ctx, ir, err)) {
        std::cerr << "lower error: " << err << "\n"; return 1;
    }

    // ── SCXQ7: optimize + validate ────────────────────────────────────────────
    if (!optimize(ir, err))            { std::cerr << "optimize: "    << err << "\n"; return 1; }
    if (!validate_neutral_ir(ir, err)) { std::cerr << "ir-validate: " << err << "\n"; return 1; }
    if (!validate_manifest(ir, ctx, err)) { std::cerr << "manifest: " << err << "\n"; return 1; }
    if (!conformance_check(ir, err))   { std::cerr << "conformance: " << err << "\n"; return 1; }

    // ── SCO/1: emit ───────────────────────────────────────────────────────────
    const std::string& kname = ir.desc.name;
    auto out = [&](const std::string& ext) { return outDir + "/" + kname + ext; };

    std::string cppPath    = out(".cpp");
    std::string hlslPath   = out(".hlsl");
    std::string wgslPath   = out(".wgsl");
    std::string cpuPath    = out(".cpu.cpp");
    std::string clPath     = out(".cl");
    std::string fragPath   = out(".frag");
    std::string dmlPath    = out(".dml.json");
    std::string smcaPath   = out(".smca.json");

    if (!emit_cpp    (ir, cppPath,  err)) { std::cerr << err << "\n"; return 1; }
    if (!emit_hlsl   (ir, hlslPath, err)) { std::cerr << err << "\n"; return 1; }
    if (!emit_wgsl   (ir, wgslPath, err)) { std::cerr << err << "\n"; return 1; }
    if (!emit_cpu    (ir, cpuPath,  err)) { std::cerr << err << "\n"; return 1; }
    if (!emit_opencl (ir, clPath,   err)) { std::cerr << err << "\n"; return 1; }
    if (!emit_webgl2 (ir, fragPath, err)) { std::cerr << err << "\n"; return 1; }
    if (!emit_dml    (ir, dmlPath,  err)) { std::cerr << err << "\n"; return 1; }
    if (!emit_manifest(ir, smcaPath, err)) { std::cerr << err << "\n"; return 1; }

    // ── IDB: sidecar ──────────────────────────────────────────────────────────
    std::vector<ArtifactEntry> artifacts = {
        {fs::path(cppPath).filename().string(),  file_cid(cppPath)},
        {fs::path(hlslPath).filename().string(), file_cid(hlslPath)},
        {fs::path(wgslPath).filename().string(), file_cid(wgslPath)},
        {fs::path(cpuPath).filename().string(),  file_cid(cpuPath)},
        {fs::path(clPath).filename().string(),   file_cid(clPath)},
        {fs::path(fragPath).filename().string(), file_cid(fragPath)},
        {fs::path(dmlPath).filename().string(),  file_cid(dmlPath)},
        {fs::path(smcaPath).filename().string(), file_cid(smcaPath)},
    };

    std::string reportPath = outDir + "/kxc.compile-report.json";
    std::string idbPath    = outDir + "/kxc.idb.jsonl";
    if (!emit_compile_report(ir, ctx, artifacts, reportPath, idbPath, err)) {
        std::cerr << err << "\n"; return 1;
    }

    std::cout << "emitted: "
              << fs::path(cppPath).filename().string()  << " "
              << fs::path(hlslPath).filename().string() << " "
              << fs::path(wgslPath).filename().string() << " "
              << fs::path(cpuPath).filename().string()  << " "
              << fs::path(clPath).filename().string()   << " "
              << fs::path(fragPath).filename().string() << " "
              << fs::path(dmlPath).filename().string()  << " "
              << fs::path(smcaPath).filename().string() << "\n";
    return 0;
}
