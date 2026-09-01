// cs_fold_kernel_compute_.hlsl — COMPUTE_FOLD XVM linear cluster kernel
// Default kernel for the TENSOR FOLD CORE: processes intra-cluster token pairs only.
// Token pairs within the same bone cluster receive amplified gravity (2× global).
// This is the per-cluster equivalent of cs_gravity_field_layer_ — same math,
// but scoped to a sorted contiguous token group (the "XVM thread cluster").
//
// Dispatch(n_head, cluster_count, 1)  numthreads(128, 1, 1)
// gid.x = attention head h
// gid.y = qi_local — query slot within this cluster (0..cluster_count-1)
// tid.x = kj_local — key   slot within this cluster (0..cluster_count-1, ≤128)

cbuffer FoldDispatchCB : register(b0) {
    uint  seq_len;        // S — full sequence length (for P_buf stride)
    uint  n_head;         // H — number of attention heads (unused: comes from dispatch dim)
    uint  cluster_start;  // start offset in sorted_idx for this cluster
    float gravity_scale;  // antigravity * 0.5 * fold_pressure_weight
};

StructuredBuffer<uint>    sorted_idx : register(t0); // [S]   — token indices in cluster order
StructuredBuffer<int>     bone_ids   : register(t1); // [S*4] — full sequence bone IDs
StructuredBuffer<float>   bone_w     : register(t2); // [S*4] — full sequence blend weights
RWStructuredBuffer<float> P_buf      : register(u0); // [H*S*S] — attention logits (in-place)

[numthreads(128, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint head     = gid.x;
    uint qi_local = gid.y;   // position within the cluster's sorted token list
    uint kj_local = lid.x;   // key slot — covers up to 128 intra-cluster tokens

    // Resolve to sequence-order token indices via sorted list
    uint query_i = sorted_idx[cluster_start + qi_local];
    uint key_j   = sorted_idx[cluster_start + kj_local];

    // Causal mask: use original sequence positions, not sorted positions
    if (key_j > query_i) return;

    // Intra-cluster LBS overlap
    float overlap = 0.0f;
    [unroll] for (uint k = 0; k < 4; ++k) {
        int   bid_i = bone_ids[query_i * 4 + k];
        float bw_i  = bone_w  [query_i * 4 + k];
        if (bid_i < 0) continue;
        [unroll] for (uint m = 0; m < 4; ++m) {
            int   bid_j = bone_ids[key_j * 4 + m];
            float bw_j  = bone_w  [key_j * 4 + m];
            if (bid_j < 0) continue;
            if (bid_i == bid_j) overlap += bw_i * bw_j;
        }
    }

    // Intra-cluster gravity: 2× the global gravity (same-fold tokens attract harder)
    float bias = gravity_scale * 2.0f * overlap;

    uint idx = head * seq_len * seq_len + query_i * seq_len + key_j;
    P_buf[idx] += bias;
}
