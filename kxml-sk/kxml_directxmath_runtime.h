// kxml_directxmath_runtime.h
// DirectXMath-based tensor contraction engine for KXML graphs
//
// Division of labor:
//   CPU (KXML Runtime): parse graph, resolve bindings, validate shapes,
//                       build command list, dispatch to backend
//   DirectXMath (SIMD): XMVectorAdd, XMMatrixMultiply, XMVector3Dot,
//                       XMVectorMax, horizontal sums, aligned copies
//
// The math is already in MathML/K'UHUL/Graphs. The GPU just executes
// Σ(a*b) and moves memory. DirectXMath gives you the SIMD to do it fast.
//
// DirectXMath .inl files (from Windows SDK):
//   DirectXMathMatrix.inl   -- 4x4 matrix ops (GEMM, transpose, inverse)
//   DirectXMathVector.inl   -- SIMD vector ops (elementwise add/mul/dot)
//   DirectXMathMisc.inl     -- utility: clamp, saturate, conversions
//   DirectXMathConvert.inl  -- type conversion: float <-> half <-> packed
//   DirectXCollision.h      -- bounding volumes (spatial indexing for graphs)
//   DirectXPackedVector.h   -- compressed vectors (SCXQ2 tensor compression)
//   DirectXColors.h         -- colour spaces (visualisation projection)
//
// Dependencies: Windows SDK 10.0.26100.0 DirectXMath headers

#pragma once

#include <DirectXMath.h>
#include <DirectXCollision.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <cstring>
#include <cmath>

using namespace DirectX;

namespace KXML {
namespace Runtime {

// ─── Tensor contraction primitives ───────────────────────────────────────────

class TensorEngine {
public:
    // Batch GEMM: C[b,i,j] = Σ_k A[b,i,k] * B[b,k,j]
    // Tiled 4×4 using 128-bit XMVECTOR (one XMVECTOR = 4 floats)
    void BatchGEMM(
        const float* __restrict A,
        const float* __restrict B,
              float* __restrict C,
        size_t M, size_t N, size_t K,
        size_t batch_size = 1,
        float alpha = 1.0f, float beta = 0.0f)
    {
        const size_t TILE = 4;
        for (size_t b = 0; b < batch_size; b++) {
            const float* Ab = A + b * M * K;
            const float* Bb = B + b * K * N;
                  float* Cb = C + b * M * N;

            for (size_t i = 0; i < M; i += TILE) {
                for (size_t j = 0; j < N; j += TILE) {
                    XMVECTOR acc[TILE][TILE];
                    for (size_t ti = 0; ti < TILE; ti++)
                        for (size_t tj = 0; tj < TILE; tj++)
                            acc[ti][tj] = XMVectorZero();

                    for (size_t k = 0; k < K; k++) {
                        for (size_t ti = 0; ti < TILE && i+ti < M; ti++) {
                            float a_val = Ab[(i+ti)*K + k];
                            XMVECTOR a  = XMVectorReplicate(a_val);
                            for (size_t tj = 0; tj < TILE && j+tj < N; tj++) {
                                XMVECTOR b_val = XMVectorReplicate(Bb[k*N + j+tj]);
                                acc[ti][tj] = XMVectorMultiplyAdd(a, b_val, acc[ti][tj]);
                            }
                        }
                    }
                    // Store tile with alpha/beta scaling
                    for (size_t ti = 0; ti < TILE && i+ti < M; ti++)
                        for (size_t tj = 0; tj < TILE && j+tj < N; tj++) {
                            float prev = beta != 0.0f ? Cb[(i+ti)*N+(j+tj)] : 0.0f;
                            Cb[(i+ti)*N+(j+tj)] =
                                alpha * XMVectorGetX(acc[ti][tj]) + beta * prev;
                        }
                }
            }
        }
    }

    // Elementwise ReLU — SIMD max(x, 0)
    void ReLU(float* data, size_t count) {
        XMVECTOR zero = XMVectorZero();
        size_t i = 0;
        for (; i + 4 <= count; i += 4) {
            XMVECTOR v = XMLoadFloat4(reinterpret_cast<const XMFLOAT4*>(data + i));
            XMStoreFloat4(reinterpret_cast<XMFLOAT4*>(data + i), XMVectorMax(v, zero));
        }
        for (; i < count; i++) data[i] = data[i] > 0.0f ? data[i] : 0.0f;
    }

    // Softmax — numerically stable: exp(x - max) / Σexp
    void Softmax(float* data, size_t count) {
        float max_val = data[0];
        for (size_t i = 1; i < count; i++)
            if (data[i] > max_val) max_val = data[i];

        float sum = 0.0f;
        for (size_t i = 0; i < count; i++) {
            data[i] = expf(data[i] - max_val);
            sum += data[i];
        }
        float inv = 1.0f / sum;
        for (size_t i = 0; i < count; i++) data[i] *= inv;
    }

    // Dot product of two float arrays
    float Dot(const float* a, const float* b, size_t n) {
        XMVECTOR acc = XMVectorZero();
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            XMVECTOR va = XMLoadFloat4(reinterpret_cast<const XMFLOAT4*>(a + i));
            XMVECTOR vb = XMLoadFloat4(reinterpret_cast<const XMFLOAT4*>(b + i));
            acc = XMVectorMultiplyAdd(va, vb, acc);
        }
        XMFLOAT4 tmp; XMStoreFloat4(&tmp, acc);
        float s = tmp.x + tmp.y + tmp.z + tmp.w;
        for (; i < n; i++) s += a[i] * b[i];
        return s;
    }

    // L2 normalise in-place
    void Normalise(float* data, size_t n) {
        float norm = sqrtf(Dot(data, data, n));
        if (norm < 1e-9f) return;
        float inv = 1.0f / norm;
        for (size_t i = 0; i < n; i++) data[i] *= inv;
    }
};

// ─── Binding types ────────────────────────────────────────────────────────────

struct TensorBinding {
    std::string            name;
    float*                 data  = nullptr;
    size_t                 bytes = 0;
    std::vector<size_t>    shape;  // row-major, e.g. {batch, M, K}

    size_t numel() const {
        size_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
};

struct NodeBinding {
    std::string                            node_id;
    std::string                            op;        // gemm|relu|softmax|dot|norm|copy
    std::vector<std::string>               from;      // input tensor names
    std::vector<std::string>               to;        // output tensor names
    std::unordered_map<std::string,float>  params;    // alpha, beta, etc.
};

// ─── Binding resolver — walks the KXML node list in phase order ───────────────

class BindingResolver {
public:
    explicit BindingResolver() = default;

    void RegisterTensor(const std::string& name, float* data,
                        size_t bytes, std::vector<size_t> shape) {
        tensors_[name] = { name, data, bytes, std::move(shape) };
    }

    void AddNode(NodeBinding nb) { nodes_.push_back(std::move(nb)); }

    // Execute all nodes in registration order (caller supplies phase ordering)
    void ExecuteGraph() {
        for (const auto& node : nodes_) ExecuteNode(node);
    }

private:
    std::unordered_map<std::string, TensorBinding> tensors_;
    std::vector<NodeBinding>                       nodes_;
    TensorEngine                                   engine_;

    TensorBinding* tensor(const std::string& name) {
        auto it = tensors_.find(name);
        return it != tensors_.end() ? &it->second : nullptr;
    }

    void ExecuteNode(const NodeBinding& node) {
        if (node.op == "gemm" && node.from.size() >= 2 && !node.to.empty()) {
            auto* A = tensor(node.from[0]);
            auto* B = tensor(node.from[1]);
            auto* C = tensor(node.to[0]);
            if (!A || !B || !C) return;

            size_t batch = A->shape.size() >= 3 ? A->shape[0] : 1;
            size_t M     = A->shape[A->shape.size()-2];
            size_t K     = A->shape[A->shape.size()-1];
            size_t N     = B->shape[B->shape.size()-1];
            float alpha  = node.params.count("alpha") ? node.params.at("alpha") : 1.0f;
            float beta   = node.params.count("beta")  ? node.params.at("beta")  : 0.0f;

            engine_.BatchGEMM(A->data, B->data, C->data, M, N, K, batch, alpha, beta);

        } else if (node.op == "relu" && !node.to.empty()) {
            auto* T = tensor(node.to[0]);
            if (T) engine_.ReLU(T->data, T->numel());

        } else if (node.op == "softmax" && !node.to.empty()) {
            auto* T = tensor(node.to[0]);
            if (T) engine_.Softmax(T->data, T->numel());

        } else if (node.op == "norm" && !node.to.empty()) {
            auto* T = tensor(node.to[0]);
            if (T) engine_.Normalise(T->data, T->numel());

        } else if (node.op == "copy" &&
                   node.from.size() >= 1 && !node.to.empty()) {
            auto* src = tensor(node.from[0]);
            auto* dst = tensor(node.to[0]);
            if (src && dst) memcpy(dst->data, src->data, dst->bytes);
        }
    }
};

// ─── Opcode dispatch bridge ───────────────────────────────────────────────────
//
// Maps KUHUL XVM opcodes to DirectXMath intrinsics for the
// CPU fallback path of AdaptiveHardwareRuntime.

static constexpr uint32_t OP_GEMM        = 0x10;
static constexpr uint32_t OP_VECTOR_ADD  = 0x12;
static constexpr uint32_t OP_VECTOR_MUL  = 0x13;
static constexpr uint32_t OP_DOT_PRODUCT = 0x14;
static constexpr uint32_t OP_TRANSPOSE   = 0x15;
static constexpr uint32_t OP_NORMALISE   = 0x16;
static constexpr uint32_t OP_RELU        = 0x17;
static constexpr uint32_t OP_SOFTMAX     = 0x18;

class DirectXMathBackend {
public:
    static constexpr size_t REGS = 16;
    XMVECTOR reg[REGS] = {};        // 128-bit SIMD registers

    void Execute(uint32_t opcode, uint32_t r0, uint32_t r1, uint32_t r2) {
        switch (opcode) {
            case OP_VECTOR_ADD:
                reg[r2] = XMVectorAdd(reg[r0], reg[r1]);  break;
            case OP_VECTOR_MUL:
                reg[r2] = XMVectorMultiply(reg[r0], reg[r1]);  break;
            case OP_DOT_PRODUCT:
                reg[r2] = XMVector4Dot(reg[r0], reg[r1]);  break;
            case OP_NORMALISE:
                reg[r1] = XMVector3Normalize(reg[r0]);  break;
            case OP_GEMM: {
                XMMATRIX A = XMLoadFloat4x4(reinterpret_cast<const XMFLOAT4X4*>(&reg[r0]));
                XMMATRIX B = XMLoadFloat4x4(reinterpret_cast<const XMFLOAT4X4*>(&reg[r1]));
                XMMATRIX C = XMMatrixMultiply(A, B);
                XMStoreFloat4x4(reinterpret_cast<XMFLOAT4X4*>(&reg[r2]), C);
                break;
            }
            case OP_TRANSPOSE: {
                XMMATRIX A = XMLoadFloat4x4(reinterpret_cast<const XMFLOAT4X4*>(&reg[r0]));
                XMMATRIX B = XMMatrixTranspose(A);
                XMStoreFloat4x4(reinterpret_cast<XMFLOAT4X4*>(&reg[r1]), B);
                break;
            }
        }
    }
};

} // namespace Runtime
} // namespace KXML
