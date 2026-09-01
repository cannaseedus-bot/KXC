// dxm_cpu_fallback.cpp - CPU fallback for tensor blocks.
// This path deliberately bypasses the Intel OpenCL CPU ICD.
#include <DirectXMath.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace DirectX;

static void add_f32(const float* a, const float* b, float* out, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const XMVECTOR av = XMLoadFloat4(reinterpret_cast<const XMFLOAT4*>(a + i));
        const XMVECTOR bv = XMLoadFloat4(reinterpret_cast<const XMFLOAT4*>(b + i));
        XMStoreFloat4(reinterpret_cast<XMFLOAT4*>(out + i), XMVectorAdd(av, bv));
    }
    for (; i < count; ++i) out[i] = a[i] + b[i];
}

int main() {
    constexpr size_t count = 5 * 4;
    const float a[count] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    };
    const float b[count] = {
        20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
        10, 9, 8, 7, 6, 5, 4, 3, 2, 1
    };
    float out[count] = {};
    add_f32(a, b, out, count);

    float max_error = 0.0f;
    for (size_t i = 0; i < count; ++i)
        max_error = (std::max)(max_error, std::fabs(out[i] - 21.0f));

    std::printf("DirectXMath CPU fallback: Matrix5x4 add, max error %.3e\n", max_error);
    std::printf("RESULT: %s\n", max_error <= 1e-6f ? "PASS" : "FAIL");
    return max_error <= 1e-6f ? 0 : 1;
}
