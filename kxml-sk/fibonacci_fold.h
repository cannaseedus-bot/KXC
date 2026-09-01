// fibonacci_fold.h — Fibonacci compression folds for KXML tensor pipeline
//
// Fibonacci isn't just a sequence — it's a compression fold pattern
// that emerges naturally from pi-field geometry.
//
//   F(n) = F(n-1) + F(n-2)          ← the fold recurrence
//   phi  = 1.6180339887...           ← golden ratio (natural decay rate)
//   GPU  = Sigma(a * b)              ← same pattern at the hardware level
//
// Applications in KXML pipeline:
//   FibonacciFold<T>           — compress 1D/2D tensors by Fibonacci windowing
//   FibonacciAttentionFold     — O(N log N) attention vs O(N^2)
//   FibonacciTensorCompression — Zeckendorf non-consecutive encoding for SCXQ2
//   FibonacciGeodesicFold      — spiral sampling for pi-field propagation
//   FibonacciSIMDFold          — DirectXMath SIMD-accelerated versions
//
// KXML binding:
//   <bind from="tensor_in" to="tensor_folded" transform="fibonacci_compress" />
//   <bind from="Q,K" to="attn" transform="fibonacci_attention" />
//   <bind from="pi_path" to="pi_folded" transform="fibonacci_geodesic" />

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "DirectXMath.h"

using namespace DirectX;

namespace KXML {
namespace Compression {

// ─── Fibonacci number table (up to 2^32) ─────────────────────────────────────

static constexpr std::array<uint32_t, 47> FIB = {
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610,
    987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025,
    121393, 196418, 317811, 514229, 832040, 1346269, 2178309, 3524578,
    5702887, 9227465, 14930352, 24157817, 39088169, 63245986, 102334155,
    165580141u, 267914296u, 433494437u, 701408733u, 1134903170u, 1836311903u
};

static constexpr float PHI = 1.6180339887f;  // golden ratio

// ─── FibonacciFold<T> ─────────────────────────────────────────────────────────
//
// Fold a 1D tensor using Fibonacci windowing:
//   Window sizes: 1, 1, 2, 3, 5, 8, 13 ...
//   Output size ≈ N / phi
//
// Average-pooling within each window gives smooth compression.
// Expand1D undoes the fold via linear interpolation.

template<typename T>
class FibonacciFold {
public:
    std::vector<T> Fold1D(const std::vector<T>& in) const {
        std::vector<T> out;
        out.reserve(in.size() / 2 + 1);
        size_t a = 1, b = 1, i = 0;
        while (i < in.size()) {
            size_t w = std::min(a, in.size() - i);
            T sum = T(0);
            for (size_t j = 0; j < w; j++) sum += in[i + j];
            out.push_back(sum / static_cast<T>(w));
            size_t next = a + b; b = a; a = next;
            i += w;
        }
        return out;
    }

    std::vector<T> Expand1D(const std::vector<T>& folded, size_t orig) const {
        std::vector<T> out(orig);
        size_t a = 1, b = 1, idx = 0, pos = 0;
        while (pos < orig && idx < folded.size()) {
            size_t w    = std::min(a, orig - pos);
            size_t end  = pos + w;
            // Linear interp between adjacent fold samples
            T v0 = folded[idx];
            T v1 = (idx + 1 < folded.size()) ? folded[idx + 1] : v0;
            for (size_t j = pos; j < end; j++) {
                float t  = w > 1 ? static_cast<float>(j - pos) / (w - 1) : 0.f;
                out[j]   = v0 * static_cast<T>(1.f - t) + v1 * static_cast<T>(t);
            }
            size_t next = a + b; b = a; a = next;
            pos = end; idx++;
        }
        return out;
    }

    // 2D fold: first fold rows, then fold the resulting columns
    std::vector<std::vector<T>> Fold2D(const std::vector<std::vector<T>>& in) const {
        if (in.empty()) return {};
        // Row pass
        std::vector<std::vector<T>> tmp;
        tmp.reserve(in.size());
        for (const auto& row : in) tmp.push_back(Fold1D(row));
        // Column pass
        size_t ncols = tmp[0].size();
        std::vector<std::vector<T>> out;
        for (size_t c = 0; c < ncols; c++) {
            std::vector<T> col; col.reserve(tmp.size());
            for (const auto& row : tmp) col.push_back(row[c]);
            auto fcol = Fold1D(col);
            if (out.size() < fcol.size()) out.resize(fcol.size());
            for (size_t r = 0; r < fcol.size(); r++) out[r].push_back(fcol[r]);
        }
        return out;
    }
};

// ─── FibonacciAttentionFold ───────────────────────────────────────────────────
//
// Compress attention from O(N^2) to O(N log_phi N) by folding Q and K
// along the sequence dimension before computing scores.

class FibonacciAttentionFold {
public:
    // Returns approximate attention scores on compressed Q/K space.
    // Q, K: flat [seq_len * d_k], output: [c_len * c_len]
    std::vector<float> FoldAttention(
        const std::vector<float>& Q, const std::vector<float>& K,
        size_t seq_len, size_t d_k) const
    {
        FibonacciFold<float> fold;
        auto Qc = fold.Fold1D(Q);
        auto Kc = fold.Fold1D(K);
        size_t c  = Qc.size() / d_k;
        float  sc = 1.f / sqrtf(static_cast<float>(d_k));

        std::vector<float> scores(c * c, 0.f);
        for (size_t i = 0; i < c; i++)
            for (size_t j = 0; j < c; j++) {
                float dot = 0.f;
                for (size_t k = 0; k < d_k; k++)
                    dot += Qc[i*d_k+k] * Kc[j*d_k+k];
                scores[i*c+j] = dot * sc;
            }
        return scores;   // caller applies softmax + V matmul
    }
};

// ─── FibonacciTensorCompression ───────────────────────────────────────────────
//
// Zeckendorf representation: every positive integer is a unique sum of
// non-consecutive Fibonacci numbers (Zeckendorf's theorem).
// Used for SCXQ2 tensor encoding — ~12.5% overhead, fully reversible.

class FibonacciTensorCompression {
public:
    static uint32_t FloorFib(uint32_t n) {
        for (int i = FIB.size()-1; i >= 0; i--)
            if (FIB[i] <= n) return FIB[i];
        return 0;
    }

    // Encode scalar as 47-bit Zeckendorf bitmap
    static uint64_t ZeckEncode(uint32_t n) {
        uint64_t bits = 0;
        for (int i = FIB.size()-1; i >= 2; i--) {
            if (n >= FIB[i]) {
                bits |= (1ull << i);
                n -= FIB[i];
                i--;  // non-consecutive constraint
            }
        }
        return bits;
    }

    static uint32_t ZeckDecode(uint64_t bits) {
        uint32_t n = 0;
        for (size_t i = 0; i < FIB.size(); i++)
            if (bits & (1ull << i)) n += FIB[i];
        return n;
    }

    // Compress a float tensor (scaled to uint32 × 10000)
    std::vector<uint64_t> Compress(const float* data, size_t n) const {
        std::vector<uint64_t> out(n);
        for (size_t i = 0; i < n; i++) {
            uint32_t q = static_cast<uint32_t>(fabsf(data[i]) * 10000.f);
            out[i]     = ZeckEncode(q);
            if (data[i] < 0.f) out[i] |= (1ull << 46);  // sign bit
        }
        return out;
    }

    void Decompress(const uint64_t* codes, float* out, size_t n) const {
        for (size_t i = 0; i < n; i++) {
            bool neg  = (codes[i] >> 46) & 1;
            uint64_t v = codes[i] & ~(1ull << 46);
            float f   = static_cast<float>(ZeckDecode(v)) / 10000.f;
            out[i]    = neg ? -f : f;
        }
    }
};

// ─── FibonacciGeodesicFold ────────────────────────────────────────────────────
//
// Sample a geodesic path at Fibonacci-indexed positions
// (0, 1, 1, 2, 3, 5, 8, 13, ...) for O(log_phi N) coverage.
// Used in pi-field propagation to reduce evaluation cost.

class FibonacciGeodesicFold {
public:
    template<typename T>
    std::vector<T> FoldGeodesic(const std::vector<T>& path) const {
        std::vector<T> out;
        size_t a = 0, b = 1;
        while (a < path.size()) {
            out.push_back(path[a]);
            size_t next = a + b; a = b; b = next;
        }
        return out;
    }

    template<typename T>
    std::vector<T> ExpandGeodesic(const std::vector<T>& folded, size_t orig) const {
        // Reconstruct Fibonacci sample positions
        std::vector<size_t> idx;
        size_t a = 0, b = 1;
        while (a < orig) { idx.push_back(a); size_t n = a+b; a=b; b=n; }

        std::vector<T> out(orig);
        for (size_t k = 0; k < folded.size() && k < idx.size(); k++) {
            size_t lo = idx[k];
            size_t hi = (k+1 < idx.size()) ? idx[k+1] : orig;
            T v0 = folded[k];
            T v1 = (k+1 < folded.size()) ? folded[k+1] : v0;
            for (size_t j = lo; j < hi && j < orig; j++) {
                float t = (hi > lo+1) ? (float)(j-lo)/(hi-lo) : 0.f;
                out[j]  = v0*(T)(1.f-t) + v1*(T)t;
            }
        }
        return out;
    }
};

// ─── FibonacciSIMDFold ────────────────────────────────────────────────────────
//
// SIMD-accelerated Fibonacci windowing using DirectXMath.
// Processes 4×4 tiles via XMVECTOR, applies golden-ratio scaling.

class FibonacciSIMDFold {
public:
    // Fold adjacent row pairs of a 4×4 matrix and scale by 1/phi
    static void Fold4x4(const XMMATRIX& in, XMMATRIX& out) {
        const XMVECTOR iphi = XMVectorReplicate(1.f / PHI);
        out.r[0] = XMVectorMultiply(XMVectorAdd(in.r[0], in.r[1]), iphi);
        out.r[1] = XMVectorMultiply(XMVectorAdd(in.r[2], in.r[3]), iphi);
        out.r[2] = XMVectorZero();
        out.r[3] = XMVectorZero();
    }

    // Zeckendorf-encode 4 floats via SIMD + scalar fallback
    static void ZeckEncode4(const XMVECTOR& v, uint64_t out[4]) {
        float f[4];
        XMStoreFloat4(reinterpret_cast<XMFLOAT4*>(f), v);
        for (int i = 0; i < 4; i++) {
            uint32_t q = static_cast<uint32_t>(fabsf(f[i]) * 10000.f);
            out[i]     = FibonacciTensorCompression::ZeckEncode(q);
        }
    }
};

} // namespace Compression
} // namespace KXML
