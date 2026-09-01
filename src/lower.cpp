#include "lower.h"
#include <fstream>
#include <sstream>

// ── minimal JSON value reader ─────────────────────────────────────────────────
// Reads a JSON string value for a key in a flat object (no nesting needed here).
static std::string json_str(const std::string& json, const std::string& key) {
    std::string pat = "\"" + key + "\"";
    auto pos = json.find(pat);
    if (pos == std::string::npos) return {};
    pos = json.find(':', pos + pat.size());
    if (pos == std::string::npos) return {};
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return {};
    auto end = json.find('"', pos + 1);
    if (end == std::string::npos) return {};
    return json.substr(pos + 1, end - pos - 1);
}

static std::string slurp(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ── classification ────────────────────────────────────────────────────────────
static void classify(const KernelDesc& d, KernelIR& ir) {
    if (d.needsMoERoute || d.needsMoEExpertFFN || d.needsMoECombine) {
        ir.kernelClass   = "moe_route_top2";
        ir.collapseClass = "routing.top2";
        ir.requires_     = {"bounded_k"};
        ir.forbids       = {"order_dependence"};
    } else if (d.needsSoftmax && d.needsMatMul) {
        ir.kernelClass   = "tensor_attention_fused";
        ir.collapseClass = "attention.fused";
        ir.requires_     = {"deterministic_join", "bounded_reduction"};
        ir.forbids       = {"side_effects", "order_dependence"};
    } else if (d.needsMeshlet) {
        ir.kernelClass   = "mesh_meshlet_cull";
        ir.collapseClass = "mesh.cull";
        ir.requires_     = {"frustum_test", "cone_test"};
        ir.forbids       = {"side_effects"};
    } else if (d.needsNormalCompute || d.needsTangentFrame) {
        ir.kernelClass   = "mesh_normal_compute";
        ir.collapseClass = "mesh.normals";
        ir.requires_     = {"atomic_accumulate", "scalar_arrays"};
        ir.forbids       = {"vec3f_layout"};  // WGSL alignment rule
    } else if (d.needsVertexProcess) {
        ir.kernelClass   = "mesh_vertex_process";
        ir.collapseClass = "mesh.vertex";
        ir.requires_     = {"scalar_arrays", "stride_offset_uniform"};
        ir.forbids       = {"vec3f_layout"};
    } else {
        ir.kernelClass   = "generic-compute";
        ir.collapseClass = "compute.generic";
        ir.requires_     = {};
        ir.forbids       = {};
    }
}

// ── registry lookup ───────────────────────────────────────────────────────────
static bool registry_lookup(const std::string& registryDir,
                             const std::string& kernelClass,
                             bool& matched) {
    // kernel-classes.json: check kernelClass key exists
    std::string classJson = slurp(registryDir + "/kernel-classes.json");
    matched = (!classJson.empty() && classJson.find(kernelClass) != std::string::npos);
    return true;
}

// ── public ────────────────────────────────────────────────────────────────────
bool lower(const KernelDesc& desc, const CompileCtx& ctx, KernelIR& out, std::string& err) {
    if (desc.name.empty()) { err = "kernel name is empty"; return false; }
    if (desc.threads[0] == 0 || desc.threads[1] == 0 || desc.threads[2] == 0) {
        err = "thread dimensions must be non-zero"; return false;
    }

    out.desc = desc;
    // static caps (HD 4600 FL 11.1 profile)
    out.waveOps     = false;
    out.heapTier    = 1;
    out.bindingTier = 1;
    out.uma         = true;
    out.lawful      = true;

    classify(desc, out);
    registry_lookup(ctx.registryDir, out.kernelClass, out.registryMatched);
    return true;
}
