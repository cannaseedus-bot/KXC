cbuffer BiasParams : register(b0) { uint rows; uint N; uint2 pad; };
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float>   b : register(t0);
[numthreads(256, 1, 1)]
void main(uint3 t : SV_DispatchThreadID) {
    uint i = t.x; if (i >= rows * N) return;
    y[i] = y[i] + b[i % N];
}
