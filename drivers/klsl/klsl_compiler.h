#pragma once
#include <string>
#include <vector>
#include <cstdint>

/**
 * klsl_compiler.h
 * KLSL (K'UHUL Language Shading Language) compiler API.
 *
 * Compiles .klsl source → HLSL (D3D12) or XVM bytecode (CPU fallback).
 *
 * Glyph map:
 *   ⟁ shader <name>    — begin shader block
 *   ⟁Xul⟁             — end shader block
 *   ⟁Wo⟁  <payload>   — declare (buffer, metadata, or local variable)
 *   ⟁Sek⟁ <payload>   — execute (statement / control flow)
 *   ⟁Ch'en⟁ <name>    — store top-of-expression into named variable
 *   ⟁Yax⟁  <name>     — load named variable
 *   [Pop <name>]       — begin function
 *   [Xul]              — end function
 *   ⟁K'ayab'⟁ <head>  — for-loop start:  var in start..end
 *   ⟁Kumk'u⟁          — for-loop end
 */

namespace klsl {

struct CompileOptions {
    bool emitHLSL = true;   // produce HLSL text
    bool emitXVM  = false;  // produce XVM bytecode (klsl_opcodes.h)
    bool emitWGSL = false;  // produce WGSL text (WebGPU — stub)
    std::string entryOverride;  // override which [Pop] is the shader entry point
};

struct CompileResult {
    bool        ok        = false;
    int         errorLine = 0;
    std::string errorMsg;
    std::string hlsl;              // HLSL output
    std::vector<uint8_t> xvm;      // XVM bytecode
    std::string wgsl;              // WGSL output (stub)
};

// Compile a .klsl source string.
CompileResult compile(const std::string& source,
                      const std::string& filename = "<klsl>",
                      const CompileOptions& opts  = {});

} // namespace klsl
