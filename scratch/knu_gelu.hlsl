static const float SQRT_2_OVER_PI = 0.7978845608f;
static const float COEFF = 0.044715f;
cbuffer GeluParams : register(b0) { uint numel; uint x_in_offset; uint2 pad; };
StructuredBuffer<float>   x_in : register(t0);
RWStructuredBuffer<float> y    : register(u0);
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x; if (i >= numel) return;
    const float x = x_in[i + x_in_offset];
    const float k = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    const float kc = clamp(k, -10.0f, 10.0f);
    y[i] = 0.5f * x * (1.0f + tanh(kc));
}
