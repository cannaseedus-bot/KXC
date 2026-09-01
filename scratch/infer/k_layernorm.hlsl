cbuffer LNFwdParams : register(b0) { uint n_embd; uint seq_len; float eps; uint pad; };
StructuredBuffer<float>   x_in   : register(t0);  // [S, E]
StructuredBuffer<float>   gamma  : register(t1);  // [E]
StructuredBuffer<float>   beta   : register(t2);  // [E]
RWStructuredBuffer<float> y_out  : register(u0);  // [S, E]
RWStructuredBuffer<float> xhat   : register(u1);  // [S, E]
RWStructuredBuffer<float> inv_std: register(u2);  // [S]
groupshared float gs_s[256];
groupshared float gs_s2[256];
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    const uint s = gid.x; const uint tid = lid.x; const uint base = s * n_embd;
    float lsum = 0.f, lsum2 = 0.f;
    for (uint i = tid; i < n_embd; i += 256) { float v = x_in[base + i]; lsum += v; lsum2 += v*v; }
    gs_s[tid] = lsum; gs_s2[tid] = lsum2;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = 128; stride >= 1; stride >>= 1) {
        if (tid < stride) { gs_s[tid] += gs_s[tid+stride]; gs_s2[tid] += gs_s2[tid+stride]; }
        GroupMemoryBarrierWithGroupSync();
    }
    const float mean = gs_s[0] / (float)n_embd;
    const float var  = gs_s2[0] / (float)n_embd - mean * mean;
    const float istd = 1.0f / sqrt(var + eps);
    if (tid == 0) inv_std[s] = istd;
    for (uint i = tid; i < n_embd; i += 256) {
        float xh = (x_in[base + i] - mean) * istd;
        xhat[base + i] = xh; y_out[base + i] = gamma[i] * xh + beta[i];
    }
}
