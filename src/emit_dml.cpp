#include "emit_dml.h"
#include <fstream>
#include <string>
#include <sstream>

// Maps KernelIR classification to KuhulIR::DMLOperatorDesc::Type int
// (mirrors the MapToDMLType() enum from kdml/src/lowering_to_dml.cpp)
static int dml_type(const KernelIR& ir) {
    // tensor_attention_fused → MatMul (type 2)
    if (ir.desc.needsSoftmax && ir.desc.needsMatMul) return 2; // MatMul
    // moe_route_top2 → Custom (type 0 reserved for extension)
    if (ir.desc.needsMoERoute)  return 19; // Custom
    // generic with matmul → MatMul
    if (ir.desc.needsMatMul)    return 2;  // MatMul
    // phase field / evolve → ElementWise (type 16)
    if (ir.desc.needsPhaseMatch) return 16; // ElementWise
    // decompress / INT4 → Custom
    if (ir.desc.needsDecompress) return 19; // Custom
    // generic-compute → Custom
    return 19;
}

static const char* dml_type_name(int t) {
    switch (t) {
        case 1:  return "Conv2D";
        case 2:  return "MatMul";
        case 3:  return "ReLU";
        case 4:  return "Softmax";
        case 5:  return "Add";
        case 6:  return "Multiply";
        case 7:  return "Pooling";
        case 16: return "ElementWise";
        case 17: return "Gemm";
        case 18: return "Activation";
        case 19: return "Custom";
        default: return "Unknown";
    }
}

bool emit_dml(const KernelIR& ir, const std::string& outPath, std::string& err) {
    std::ofstream f(outPath);
    if (!f) { err = "cannot open " + outPath; return false; }

    auto& d = ir.desc;
    int type_int = dml_type(ir);
    const char* type_str = dml_type_name(type_int);

    // Estimate input/output shape from thread dims
    // KXC kernels are flat compute dispatches: shape = [X*Y*Z] element output
    uint64_t elem = (uint64_t)d.threads[0] * d.threads[1] * d.threads[2];

    f << "{\n";
    f << "  \"kernel\": \"" << d.name << "\",\n";
    f << "  \"kdml_version\": \"0.1.0\",\n";
    f << "  \"dml_op\": \"" << type_str << "\",\n";
    f << "  \"dml_type_int\": " << type_int << ",\n";
    f << "  \"kernelClass\": \"" << ir.kernelClass << "\",\n";
    f << "  \"collapseClass\": \"" << ir.collapseClass << "\",\n";
    f << "  \"threads\": [" << d.threads[0] << ", " << d.threads[1] << ", " << d.threads[2] << "],\n";

    // Op-specific params block
    f << "  \"params\": {\n";
    if (d.needsMatMul && d.needsSoftmax) {
        // ScaledDotProduct attention
        f << "    \"transpose_b\": true,\n";
        f << "    \"alpha\": 0.125,\n";
        f << "    \"fused_softmax\": true\n";
    } else if (d.needsMatMul) {
        f << "    \"transpose_a\": false,\n";
        f << "    \"transpose_b\": false,\n";
        f << "    \"alpha\": 1.0\n";
    } else if (d.needsMoERoute) {
        f << "    \"top_k\": 2,\n";
        f << "    \"num_experts\": 8,\n";
        f << "    \"normalize\": true\n";
    } else if (d.needsPhaseMatch) {
        f << "    \"op\": \"field_evolve\",\n";
        f << "    \"damping\": 0.9,\n";
        f << "    \"dt\": 0.01\n";
    } else if (d.needsDecompress) {
        f << "    \"bits\": 4,\n";
        f << "    \"zero_point\": 8\n";
    } else {
        f << "    \"op\": \"passthrough\"\n";
    }
    f << "  },\n";

    // Tensor shape hints
    f << "  \"input_shapes\": [[" << elem << "]],\n";
    f << "  \"output_shapes\": [[" << elem << "]],\n";

    // Caps (HD 4600 static profile — matches lower.cpp)
    f << "  \"caps\": {\n";
    f << "    \"waveOps\": false,\n";
    f << "    \"heapTier\": 1,\n";
    f << "    \"bindingTier\": 1,\n";
    f << "    \"uma\": true\n";
    f << "  },\n";

    // Sek flags
    f << "  \"sek\": {\n";
    f << "    \"needsDecompress\":   " << (d.needsDecompress   ? "true" : "false") << ",\n";
    f << "    \"needsSoftmax\":      " << (d.needsSoftmax      ? "true" : "false") << ",\n";
    f << "    \"needsMatMul\":       " << (d.needsMatMul       ? "true" : "false") << ",\n";
    f << "    \"kvInt4\":            " << (d.kvInt4            ? "true" : "false") << ",\n";
    f << "    \"needsMoERoute\":     " << (d.needsMoERoute     ? "true" : "false") << ",\n";
    f << "    \"needsMoEExpertFFN\": " << (d.needsMoEExpertFFN ? "true" : "false") << ",\n";
    f << "    \"needsMoECombine\":   " << (d.needsMoECombine   ? "true" : "false") << ",\n";
    f << "    \"needsPhaseMatch\":   " << (d.needsPhaseMatch   ? "true" : "false") << "\n";
    f << "  },\n";

    // Pipeline provenance
    f << "  \"layers\": [\"MATRIX\", \"SCXQ2\", \"SCXQ7\", \"SCO/1\", \"IDB\"],\n";
    f << "  \"registryMatched\": " << (ir.registryMatched ? "true" : "false") << ",\n";
    f << "  \"lawful\": " << (ir.lawful ? "true" : "false") << "\n";
    f << "}\n";

    return true;
}
