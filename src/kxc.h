#pragma once
#include <string>
#include <vector>
#include <cstdint>

// ── K'UHUL kernel descriptor (MATRIX layer) ─────────────────────────────────
struct KernelDesc {
    std::string name;
    uint32_t threads[3] = {1, 1, 1};
    // [Sek] properties
    bool needsDecompress    = false;
    bool needsSoftmax       = false;
    bool needsMatMul        = false;
    bool kvInt4             = false;
    bool needsMoERoute      = false;
    bool needsMoEExpertFFN  = false;
    bool needsMoECombine    = false;
    bool needsPhaseMatch    = false;
};

// ── Backend-neutral IR (SCXQ2 layer) ─────────────────────────────────────────
struct KernelIR {
    KernelDesc desc;

    // classification
    std::string kernelClass;    // "tensor_attention_fused" | "generic-compute" | "moe_route_top2"
    std::string collapseClass;  // "attention.fused" | "compute.generic" | "routing.top2"
    bool        lawful          = true;
    bool        registryMatched = false;
    std::vector<std::string> requires_;
    std::vector<std::string> forbids;

    // static caps profile (HD 4600 / FL 11.1 stack)
    bool waveOps    = false;
    int  heapTier   = 1;
    int  bindingTier = 1;
    bool uma        = true;
};

// ── Compile context ───────────────────────────────────────────────────────────
struct CompileCtx {
    std::string sourcePath;
    std::string outDir;         // output directory (default: cwd)
    std::string registryDir;    // path to registry/ dir (default: beside exe)
    std::string stackManifest;  // asx_manifest.json path
    std::string stackId    = "asx-xcfe-stack/v1";
    std::string stackCid   = "fnv1a64:1c36e0cc928a7591";
};

// ── FNV-1a 64-bit ─────────────────────────────────────────────────────────────
inline uint64_t fnv1a64(const char* data, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i)
        h = (h ^ static_cast<uint8_t>(data[i])) * 0x100000001b3ULL;
    return h;
}
inline uint64_t fnv1a64(const std::string& s) { return fnv1a64(s.data(), s.size()); }

// ── Hex formatting ────────────────────────────────────────────────────────────
inline std::string hex64(uint64_t v) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)v);
    return buf;
}
