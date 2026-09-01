#include "../include/simd_kernels.h"
#include <immintrin.h>
#include <cstring>
#include <vector>
#include <algorithm>

void simd_add_i64(const int64_t* a, const int64_t* b, int64_t* out, size_t count){
#if defined(__AVX2__) || defined(_MSC_VER)
    size_t i=0; const size_t step = 4; // 256-bit / 64-bit
    for (; i + step <= count; i += step){
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vr = _mm256_add_epi64(va, vb);
        _mm256_storeu_si256((__m256i*)(out + i), vr);
    }
    for (; i < count; ++i) out[i] = a[i] + b[i];
#else
    for (size_t i=0;i<count;++i) out[i] = a[i] + b[i];
#endif
}

void executeAddLane(LaneBatch &batch, Buffers &buf){
    size_t n = batch.nodes.size();
    if (n==0) return;
    std::vector<int64_t> a(n), b(n), r(n);
    for (size_t i=0;i<n;++i){
        IRNode* node = batch.nodes[i];
        a[i] = buf.i64[node->a_idx];
        b[i] = buf.i64[node->b_idx];
    }
    simd_add_i64(a.data(), b.data(), r.data(), n);
    for (size_t i=0;i<n;++i){
        IRNode* node = batch.nodes[i];
        buf.i64[node->out_idx] = r[i];
    }
}
