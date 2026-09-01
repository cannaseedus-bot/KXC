cbuffer AddParams : register(b0) { uint len; uint3 pad; };
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float>   r : register(t0);
[numthreads(256, 1, 1)]
void main(uint3 t : SV_DispatchThreadID) {
    uint i = t.x; if (i >= len) return;
    y[i] = y[i] + r[i];
}
