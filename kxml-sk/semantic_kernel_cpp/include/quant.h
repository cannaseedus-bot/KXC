#pragma once
#include <vector>
#include <cstdint>

// Quantize floats to int8 (per-tensor symmetric) — CPU fallback
void quantize_int8_cpu(const float* in, int8_t* out, size_t n, float &scale);

// Dequantize int8 to floats
void dequantize_int8_cpu(const int8_t* in, float* out, size_t n, float scale);

// Helper that works with std::vector
inline void quantize_int8_vec(const std::vector<float> &in, std::vector<int8_t> &out, float &scale){
    out.resize(in.size());
    quantize_int8_cpu(in.data(), out.data(), in.size(), scale);
}
inline void dequantize_int8_vec(const std::vector<int8_t> &in, std::vector<float> &out, float scale){
    out.resize(in.size());
    dequantize_int8_cpu(in.data(), out.data(), in.size(), scale);
}
