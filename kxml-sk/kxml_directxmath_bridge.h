// kxml_directxmath_bridge.h
// Bridges KXML graph bindings to DirectXMath primitives.
//
// Extends kxml_directxmath_runtime.h (TensorEngine + BindingResolver)
// with a low-level opcode dispatch layer: KUHUL XVM opcodes map directly
// to DirectXMath SIMD intrinsics on a 16-register XMVECTOR file.
//
// Division of labour:
//   kxml_directxmath_runtime.h  — high-level BindingResolver (C = A * B)
//   kxml_directxmath_bridge.h   — low-level opcode dispatch (OP_GEMM → XMMatrixMultiply)
//
// DirectXMath .inl files (Windows SDK 10.0.26100.0):
//   DirectXMathMatrix.inl   — 4x4 GEMM, transpose, inverse
//   DirectXMathVector.inl   — SIMD elementwise add/mul/dot
//   DirectXMathMisc.inl     — clamp, saturate, type conversions
//   DirectXMathConvert.inl  — float <-> half <-> packed
//   DirectXCollision.h      — bounding volumes (spatial graph indexing)
//   DirectXPackedVector.h   — compressed vectors (SCXQ2 tensor compression)
//   DirectXColors.h         — colour spaces (visualisation projection)

#pragma once

#include "DirectXMath.h"
#include "DirectXMathConvert.inl"
#include "DirectXMathMatrix.inl"
#include "DirectXMathMisc.inl"
#include "DirectXMathVector.inl"
#include "DirectXCollision.h"
#include "DirectXColors.h"
#include <vector>

using namespace DirectX;

namespace KXML {
namespace DXBridge {

// KUHUL XVM compute opcodes (subset used by KXML graph execution)
static constexpr uint32_t OP_GEMM        = 0x10;  // C = A * B  (4×4 matmul)
static constexpr uint32_t OP_VECTOR_ADD  = 0x12;  // C = A + B
static constexpr uint32_t OP_VECTOR_MUL  = 0x13;  // C = A * B  (elementwise)
static constexpr uint32_t OP_DOT_PRODUCT = 0x14;  // result = A · B
static constexpr uint32_t OP_TRANSPOSE   = 0x15;  // B = A^T
static constexpr uint32_t OP_NORMALISE   = 0x16;  // B = norm(A)
static constexpr uint32_t OP_RELU        = 0x17;  // B = max(A, 0)
static constexpr uint32_t OP_SOFTMAX     = 0x18;  // B = softmax(A)
static constexpr uint32_t OP_REDUCE_SUM  = 0x19;  // scalar = Σ A
static constexpr uint32_t OP_REDUCE_MAX  = 0x1A;  // scalar = max(A)
static constexpr uint32_t OP_SCALE       = 0x1B;  // B = A * scalar
static constexpr uint32_t OP_COPY        = 0x1C;  // B = A

// ─── DirectXMathBackend ───────────────────────────────────────────────────────
//
// 16-register XMVECTOR file (each = 128 bits = 4 × float32).
// Operands in Execute() are register indices (r0-r15).
// Matrix operations use 4 consecutive registers as a 4×4 XMFLOAT4X4.
//
// Register layout convention:
//   r0-r3   : matrix A  (XMFLOAT4X4, row-major)
//   r4-r7   : matrix B
//   r8-r11  : matrix C  (output)
//   r12     : scalar / result
//   r13-r15 : scratch

class DirectXMathBackend {
public:
    static constexpr size_t REGS = 16;
    XMVECTOR reg[REGS] = {};

    // ── Primary dispatch ────────────────────────────────────────────────────
    // Execute a single KUHUL opcode. r0,r1,r2 are register indices.
    void Execute(uint32_t opcode, uint32_t r0, uint32_t r1, uint32_t r2 = 0) {
        switch (opcode) {

            case OP_GEMM: {
                // 4×4 GEMM: reg[r2] = reg[r0] * reg[r1]
                XMMATRIX A = XMLoadFloat4x4(mat_ptr(r0));
                XMMATRIX B = XMLoadFloat4x4(mat_ptr(r1));
                XMStoreFloat4x4(mat_ptr(r2), XMMatrixMultiply(A, B));
                break;
            }
            case OP_VECTOR_ADD:
                reg[r2] = XMVectorAdd(reg[r0], reg[r1]);
                break;

            case OP_VECTOR_MUL:
                reg[r2] = XMVectorMultiply(reg[r0], reg[r1]);
                break;

            case OP_DOT_PRODUCT:
                reg[r2] = XMVector4Dot(reg[r0], reg[r1]);
                break;

            case OP_TRANSPOSE: {
                XMMATRIX A = XMLoadFloat4x4(mat_ptr(r0));
                XMStoreFloat4x4(mat_ptr(r1), XMMatrixTranspose(A));
                break;
            }
            case OP_NORMALISE:
                reg[r1] = XMVector3Normalize(reg[r0]);
                break;

            case OP_RELU: {
                XMVECTOR zero = XMVectorZero();
                reg[r1] = XMVectorMax(reg[r0], zero);
                break;
            }
            case OP_SCALE: {
                // r1 = r0 * scalar stored in r2.x
                float s = XMVectorGetX(reg[r2]);
                reg[r1] = XMVectorScale(reg[r0], s);
                break;
            }
            case OP_COPY:
                reg[r1] = reg[r0];
                break;

            case OP_REDUCE_SUM: {
                // Horizontal sum of 4 floats in r0 → r1.x
                XMVECTOR s = XMVectorAdd(
                    XMVectorSwizzle<0,1,0,1>(reg[r0]),
                    XMVectorSwizzle<2,3,2,3>(reg[r0]));
                reg[r1] = XMVectorAdd(
                    XMVectorSwizzle<0,0,0,0>(s),
                    XMVectorSwizzle<1,1,1,1>(s));
                break;
            }
            case OP_REDUCE_MAX:
                reg[r1] = XMVectorMax(
                    XMVectorMax(
                        XMVectorSwizzle<0,0,0,0>(reg[r0]),
                        XMVectorSwizzle<1,1,1,1>(reg[r0])),
                    XMVectorMax(
                        XMVectorSwizzle<2,2,2,2>(reg[r0]),
                        XMVectorSwizzle<3,3,3,3>(reg[r0])));
                break;

            default:
                break;  // unknown opcode — no-op
        }
    }

    // ── Convenience: load/store ─────────────────────────────────────────────
    void LoadVec4(uint32_t r, const float* src) {
        reg[r] = XMLoadFloat4(reinterpret_cast<const XMFLOAT4*>(src));
    }
    void StoreVec4(uint32_t r, float* dst) const {
        XMStoreFloat4(reinterpret_cast<XMFLOAT4*>(dst), reg[r]);
    }

    // Load a 4×4 matrix into registers [base, base+3]
    void LoadMat4x4(uint32_t base, const float* src) {
        for (int i = 0; i < 4; i++)
            reg[base + i] = XMLoadFloat4(reinterpret_cast<const XMFLOAT4*>(src + i * 4));
    }
    void StoreMat4x4(uint32_t base, float* dst) const {
        for (int i = 0; i < 4; i++)
            XMStoreFloat4(reinterpret_cast<XMFLOAT4*>(dst + i * 4), reg[base + i]);
    }

    float GetScalar(uint32_t r) const { return XMVectorGetX(reg[r]); }
    void  SetScalar(uint32_t r, float v) { reg[r] = XMVectorReplicate(v); }

private:
    // Treat 4 consecutive XMVECTOR regs as a XMFLOAT4X4 (for matrix ops)
    XMFLOAT4X4* mat_ptr(uint32_t base) {
        return reinterpret_cast<XMFLOAT4X4*>(&reg[base]);
    }
    const XMFLOAT4X4* mat_ptr(uint32_t base) const {
        return reinterpret_cast<const XMFLOAT4X4*>(&reg[base]);
    }
};

// ─── KXML opcode sequence for a single node ───────────────────────────────────
//
// Compiled from a KXMLNode's <ops> / <bind> list.
// Each NodeProgram is a flat list of (opcode, r0, r1, r2) tuples.

struct NodeInstruction {
    uint32_t opcode;
    uint32_t r0, r1, r2;
};

struct NodeProgram {
    std::string          node_id;
    std::string          phase;
    std::vector<NodeInstruction> instrs;
};

// ─── Graph program executor ───────────────────────────────────────────────────

class GraphExecutor {
public:
    void AddProgram(NodeProgram prog) { programs_.push_back(std::move(prog)); }

    void Execute() {
        for (const auto& prog : programs_) {
            for (const auto& instr : prog.instrs) {
                backend_.Execute(instr.opcode, instr.r0, instr.r1, instr.r2);
            }
        }
    }

    DirectXMathBackend& backend() { return backend_; }

private:
    DirectXMathBackend         backend_;
    std::vector<NodeProgram>   programs_;
};

} // namespace DXBridge
} // namespace KXML
