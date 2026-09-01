#include "../include/quant.h"
#include <cmath>
#include <algorithm>

void quantize_int8_cpu(const float* in, int8_t* out, size_t n, float &scale){
    float maxv = 0.0f;
    for (size_t i=0;i<n;i++){
        float a = std::abs(in[i]);
        if (a > maxv) maxv = a;
    }
    scale = maxv / 127.0f + 1e-8f;
    if (scale <= 0.0f) scale = 1e-8f;
    for (size_t i=0;i<n;i++){
        float v = in[i] / scale;
        int32_t q = (int32_t)std::lround(v);
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        out[i] = (int8_t)q;
    }
}

void dequantize_int8_cpu(const int8_t* in, float* out, size_t n, float scale){
    for (size_t i=0;i<n;i++) out[i] = (float)in[i] * scale;
}
