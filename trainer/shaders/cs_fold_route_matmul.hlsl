// cs_fold_route_matmul.hlsl — MoE fold-routed matmul for the TENSOR FOLD CORE
// Derived from ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id.wgsl (expert-ID-routed matmul).
// HLSL adaptation: one dispatch per active expert cluster, scatters output back
// to original token positions via sorted_idx.  No preprocessing templates needed.
//
// Semantics: for each token in the cluster → output[token_i, row] = dot(src[token_i], W[row])
// where W is the cluster's dedicated weight submatrix (expert weight).
//
// Dispatch(ceil(n_rows / 128), cluster_count, 1)  numthreads(128, 1, 1)
// gid.x = output-row tile,  gid.y = token-within-cluster,  lid.x = row within tile

cbuffer FoldRouteMatmulCB : register(b0) {
    uint n_rows;        // output feature dimension (rows of W)
    uint n_cols;        // input feature dimension (cols of W / activations)
    uint n_tokens;      // number of tokens assigned to this cluster
    uint cluster_start; // offset into sorted_idx for this cluster
};

StructuredBuffer<uint>    sorted_idx : register(t0); // [S]            — full sequence sort order
StructuredBuffer<float>   src        : register(t1); // [S * n_cols]   — token activations (all tokens)
StructuredBuffer<float>   weight     : register(t2); // [n_rows * n_cols] — expert weight matrix (row-major)

RWStructuredBuffer<float> dst        : register(u0); // [S * n_rows]   — output accumulator

[numthreads(128, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint row     = gid.x * 128u + lid.x; // which output row (feature index)
    uint t_local = gid.y;                 // which token within this cluster

    if (row >= n_rows || t_local >= n_tokens) return;

    // Map cluster-local token index → original sequence position
    uint token_i = sorted_idx[cluster_start + t_local];

    // Dot product: row of W · activation vector for token_i
    float acc = 0.0f;
    uint src_base = token_i * n_cols;
    uint w_base   = row     * n_cols;
    for (uint c = 0u; c < n_cols; ++c)
        acc += src[src_base + c] * weight[w_base + c];

    // Scatter back: clusters are disjoint so each (token_i, row) is written by exactly one thread.
    dst[token_i * n_rows + row] = acc;
}
