cbuffer EmbedParams : register(b0) { uint seq_len; uint n_embd; uint2 pad; };
StructuredBuffer<int>     tokens : register(t0);  // [S]
StructuredBuffer<float>   wte    : register(t1);  // [V, E]
StructuredBuffer<float>   wpe    : register(t2);  // [ctx, E]
RWStructuredBuffer<float> h_out  : register(u0);  // [S, E]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    const uint i = gid.x; if (i >= seq_len) return;
    const uint tok = (uint)tokens[i];
    for (uint d = lid.x; d < n_embd; d += 256)
        h_out[i * n_embd + d] = wte[tok * n_embd + d] + wpe[i * n_embd + d];
}
