#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

// Explicit row-major 5x4 f32 block. This is a storage/tile contract;
// backends may vectorize it but must not assume a particular SIMD width.
struct Matrix5x4 {
    float lanes[5][4]{};
};

static_assert(sizeof(Matrix5x4) == 20 * sizeof(float), "Matrix5x4 must be 80 bytes");
static_assert(alignof(Matrix5x4) == alignof(float), "Matrix5x4 alignment must remain scalar-compatible");

inline float& matrix5x4_at(Matrix5x4& value, std::size_t lane, std::size_t channel) {
    return value.lanes[lane][channel];
}

inline const float& matrix5x4_at(const Matrix5x4& value, std::size_t lane, std::size_t channel) {
    return value.lanes[lane][channel];
}

inline Matrix5x4 matrix5x4_mul(const Matrix5x4& a, const Matrix5x4& b) {
    Matrix5x4 out{};
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            out.lanes[i][j] = a.lanes[i][j] * b.lanes[i][j];
    return out;
}

inline Matrix5x4 matrix5x4_add(const Matrix5x4& a, const Matrix5x4& b) {
    Matrix5x4 out{};
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            out.lanes[i][j] = a.lanes[i][j] + b.lanes[i][j];
    return out;
}

inline Matrix5x4 matrix5x4_swiglu(const Matrix5x4& x, const Matrix5x4& gate) {
    Matrix5x4 out{};
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 4; ++j) {
            const float g = gate.lanes[i][j];
            const float sigmoid = 1.0f / (1.0f + std::exp(-g));
            out.lanes[i][j] = x.lanes[i][j] * sigmoid;
        }
    return out;
}
