// cs_gravity_field_layer_.hlsl
// Per-layer gravity well: LBS-overlap bias written into P_buf_ (in-place, post-softmax).
// Reads LN1 activations + 4-bone LBS data for the current sequence.
// Dispatch(H, S, 1)  numthreads(128,1,1) — same pattern as cs_kuhul_think_bias_.
// Constraint: S <= 128 (default block_size=128 satisfied).
// Layer index controls a depth-scaling curve so deeper layers contribute less gravity.

cbuffer GravityLayerCB : register(b0)
{
    uint  seq_len;        // S — current sequence length
    uint  n_head;         // H — number of attention heads
    uint  layer_idx;      // l — current transformer layer (0..NL-1)
    float gravity_scale;  // antigravity * 0.5 — from KuhulPhysicsSolver
};

StructuredBuffer<float>   ln1_out  : register(t0); // [S * E] — LN1 output activations (reserved)
Buffer<int>               bone_ids : register(t1); // [S * 4] — 4 LBS bone IDs per token
StructuredBuffer<float>   bone_w   : register(t2); // [S * 4] — LBS blend weights, sum=1 per token

RWStructuredBuffer<float> P_buf    : register(u0); // [H * S * S] — attention weights (in-place)

[numthreads(128, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint head    = gid.x;   // 0 .. H-1
    uint query_i = gid.y;   // 0 .. S-1
    uint key_j   = tid.x;   // 0 .. 127 (covers S when block_size <= 128)

    if (key_j >= seq_len || query_i >= seq_len) return;

    // Causal mask: gravity only within the visible window
    if (key_j > query_i) return;

    // LBS overlap: sum of shared-bone weighted products
    // lbs_overlap(i,j) = sum_k sum_m w_i[k]*w_j[m]*(bone_ids_i[k]==bone_ids_j[m])
    float overlap = 0.0f;
    [unroll] for (uint k = 0; k < 4; ++k)
    {
        int   bid_i = bone_ids[query_i * 4 + k];
        float bw_i  = bone_w  [query_i * 4 + k];
        if (bid_i < 0) continue;
        [unroll] for (uint m = 0; m < 4; ++m)
        {
            int   bid_j = bone_ids[key_j * 4 + m];
            float bw_j  = bone_w  [key_j * 4 + m];
            if (bid_j < 0) continue;
            if (bid_i == bid_j)
                overlap += bw_i * bw_j;
        }
    }

    // Depth decay: early layers carry stronger gravity (bone structure shapes early routing)
    // layer 0 → scale * 1.0, layer 5 → scale * ~0.54 (exp(-l/NL_eff))
    float depth_decay = exp(-0.1f * (float)layer_idx);
    float bias = gravity_scale * depth_decay * overlap;

    // All heads see the same bone-structure bias (LBS is head-agnostic)
    uint idx = head * seq_len * seq_len + query_i * seq_len + key_j;
    P_buf[idx] += bias;
}
