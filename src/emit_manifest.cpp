#include "emit_manifest.h"
#include <fstream>
#include <sstream>
#include <cstdio>

static std::string cid_of_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "fnv1a64:0000000000000000";
    std::ostringstream ss; ss << f.rdbuf();
    std::string data = ss.str();
    uint64_t h = fnv1a64(data.data(), data.size());
    char buf[32]; snprintf(buf, sizeof(buf), "fnv1a64:%016llx", (unsigned long long)h);
    return buf;
}

static void write_str_array(std::ofstream& f, const std::vector<std::string>& v) {
    f << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) f << ", ";
        f << "\"" << v[i] << "\"";
    }
    f << "]";
}

bool emit_manifest(const KernelIR& ir, const std::string& smcaPath, std::string& err) {
    std::ofstream f(smcaPath);
    if (!f) { err = "cannot write: " + smcaPath; return false; }

    const auto& d = ir.desc;
    f << "{\n";
    f << "  \"kernel\": \"" << d.name << "\",\n";
    f << "  \"target\": \"all\",\n";
    f << "  \"threads\": [" << d.threads[0] << ", " << d.threads[1] << ", " << d.threads[2] << "],\n";
    f << "  \"caps\": {\n";
    f << "    \"waveOps\": " << (ir.waveOps ? "true" : "false") << ",\n";
    f << "    \"heapTier\": " << ir.heapTier << ",\n";
    f << "    \"bindingTier\": " << ir.bindingTier << ",\n";
    f << "    \"uma\": " << (ir.uma ? "true" : "false") << "\n";
    f << "  },\n";
    f << "  \"smca\": {\n";
    f << "    \"kernelClass\": \"" << ir.kernelClass << "\",\n";
    f << "    \"collapseClass\": \"" << ir.collapseClass << "\",\n";
    f << "    \"lawful\": " << (ir.lawful ? "true" : "false") << ",\n";
    f << "    \"registryMatched\": " << (ir.registryMatched ? "true" : "false") << ",\n";
    f << "    \"layers\": [\"MATRIX\", \"SCXQ2\", \"SCXQ7\", \"SCO/1\", \"IDB\"],\n";
    f << "    \"requires\": "; write_str_array(f, ir.requires_); f << ",\n";
    f << "    \"forbids\": ";  write_str_array(f, ir.forbids);   f << ",\n";
    f << "    \"notes\": [\n";
    f << "      \"Kernel class resolved via SMCA registry.\",\n";
    f << "      \"MATRIX: source K'uhul parsed into AST.\",\n";
    f << "      \"SCXQ2: semantic ops lowered into backend-neutral IR.\",\n";
    f << "      \"SCXQ7: legality and caps-aware optimization applied.\",\n";
    f << "      \"SCO/1: backend emitters produce executable artifacts.\",\n";
    f << "      \"IDB: sidecar metadata emitted for external verification.\"\n";
    f << "    ]\n";
    f << "  }\n}\n";
    return true;
}

bool emit_compile_report(const KernelIR& ir, const CompileCtx& ctx,
                         const std::vector<ArtifactEntry>& artifacts,
                         const std::string& reportPath, const std::string& idbPath,
                         std::string& err) {
    // ── run CID ──────────────────────────────────────────────────────────────
    uint64_t runCidVal = fnv1a64(ctx.sourcePath + ir.desc.name);
    char runCidStr[32]; snprintf(runCidStr, sizeof(runCidStr), "fnv1a64:%016llx", (unsigned long long)runCidVal);

    uint64_t srcCidVal = fnv1a64(ctx.sourcePath);
    char srcCidStr[32]; snprintf(srcCidStr, sizeof(srcCidStr), "fnv1a64:%016llx", (unsigned long long)srcCidVal);

    // ── kxc.compile-report.json ───────────────────────────────────────────────
    {
        std::ofstream f(reportPath);
        if (!f) { err = "cannot write: " + reportPath; return false; }

        uint64_t reportCidVal = fnv1a64(reportPath);
        char reportCidStr[32]; snprintf(reportCidStr, sizeof(reportCidStr), "fnv1a64:%016llx", (unsigned long long)reportCidVal);

        // Replace backslashes for JSON
        auto esc = [](std::string s) {
            std::string r; for (char c : s) { if (c == '\\') r += "\\\\"; else r += c; } return r;
        };

        f << "{\n";
        f << "  \"stackId\": \"" << ctx.stackId << "\",\n";
        f << "  \"stackManifest\": \"" << esc(ctx.stackManifest) << "\",\n";
        f << "  \"stackCid\": \"" << ctx.stackCid << "\",\n";
        f << "  \"sourceKind\": \"kuhul\",\n";
        f << "  \"sourcePath\": \"" << esc(ctx.sourcePath) << "\",\n";
        f << "  \"sourceCid\": \"" << srcCidStr << "\",\n";
        f << "  \"sourceSchema\": \"\",\n";
        f << "  \"originSourceKind\": \"\",\n";
        f << "  \"originSourcePath\": \"\",\n";
        f << "  \"originSourceCid\": \"\",\n";
        f << "  \"entries\": [\n    {\n";
        f << "      \"kernel\": \"" << ir.desc.name << "\",\n";
        f << "      \"lawful\": " << (ir.lawful ? "true" : "false") << ",\n";
        f << "      \"registryMatched\": " << (ir.registryMatched ? "true" : "false") << ",\n";
        f << "      \"artifacts\": [\n";
        for (size_t i = 0; i < artifacts.size(); ++i) {
            f << "        {\"name\": \"" << artifacts[i].name << "\", \"cid\": \"" << artifacts[i].cid << "\"}";
            if (i + 1 < artifacts.size()) f << ",";
            f << "\n";
        }
        f << "      ]\n    }\n  ]\n}\n";
    }

    // ── kxc.idb.jsonl ────────────────────────────────────────────────────────
    {
        std::ofstream f(idbPath);
        if (!f) { err = "cannot write: " + idbPath; return false; }

        auto esc = [](std::string s) {
            std::string r; for (char c : s) { if (c == '\\') r += "\\\\"; else r += c; } return r;
        };

        uint64_t smcaCidVal = fnv1a64(ir.desc.name + ir.kernelClass);
        char smcaCidStr[32]; snprintf(smcaCidStr, sizeof(smcaCidStr), "fnv1a64:%016llx", (unsigned long long)smcaCidVal);

        uint64_t rptCidVal = fnv1a64(reportPath);
        char rptCidStr[32]; snprintf(rptCidStr, sizeof(rptCidStr), "fnv1a64:%016llx", (unsigned long long)rptCidVal);

        // line 1: compile event
        f << "{\"kind\":\"kxc.compile\","
          << "\"kernel\":\"" << ir.desc.name << "\","
          << "\"target\":\"all\","
          << "\"stackId\":\"" << ctx.stackId << "\","
          << "\"stackRoot\":\"" << esc(ctx.stackManifest) << "\","
          << "\"stackCid\":\"" << ctx.stackCid << "\","
          << "\"runCid\":\"" << runCidStr << "\","
          << "\"sourceKind\":\"kuhul\","
          << "\"sourcePath\":\"" << esc(ctx.sourcePath) << "\","
          << "\"sourceCid\":\"" << srcCidStr << "\","
          << "\"sourceSchema\":\"\","
          << "\"originSourceKind\":\"\","
          << "\"originSourcePath\":\"\","
          << "\"originSourceCid\":\"\","
          << "\"cid\":\"" << smcaCidStr << "\","
          << "\"lawful\":" << (ir.lawful ? "true" : "false") << ","
          << "\"registryMatched\":" << (ir.registryMatched ? "true" : "false") << ","
          << "\"kernelClass\":\"" << ir.kernelClass << "\","
          << "\"collapseClass\":\"" << ir.collapseClass << "\","
          << "\"waveOps\":" << (ir.waveOps ? "true" : "false") << ","
          << "\"heapTier\":" << ir.heapTier << ","
          << "\"bindingTier\":" << ir.bindingTier << ","
          << "\"uma\":" << (ir.uma ? "true" : "false")
          << "}\n";

        // line 2: compile-report event
        f << "{\"kind\":\"kxc.compile-report\","
          << "\"stackId\":\"" << ctx.stackId << "\","
          << "\"stackRoot\":\"" << esc(ctx.stackManifest) << "\","
          << "\"stackCid\":\"" << ctx.stackCid << "\","
          << "\"runCid\":\"" << runCidStr << "\","
          << "\"reportPath\":\"" << esc(reportPath) << "\","
          << "\"reportCid\":\"" << rptCidStr << "\""
          << "}\n";
    }
    return true;
}
