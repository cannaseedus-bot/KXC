// cs_bone_argsort_.hlsl — Counting sort on primary bone cluster ID
// Produces the XVM linear cluster layout: contiguous token groups per bone cluster.
// Dispatch(1, 1, 1)  numthreads(128, 1, 1)
// Constraints: S <= 128, N_CLUSTERS <= 64 (both fit in groupshared)

cbuffer BoneSortCB : register(b0) {
    uint seq_len;    // S — current sequence length
    uint n_clusters; // active cluster count (≤ 64)
    uint pad0, pad1;
};

StructuredBuffer<int>    bone_ids     : register(t0); // [S*4] — primary bone at [i*4+0]
RWStructuredBuffer<uint> sorted_idx   : register(u0); // [S]   — token indices in cluster order
RWStructuredBuffer<uint> cluster_start: register(u1); // [n_clusters] — start offset per cluster
RWStructuredBuffer<uint> cluster_count: register(u2); // [n_clusters] — token count per cluster

groupshared uint gs_count  [64];   // phase 1: occurrence count per cluster
groupshared uint gs_start  [64];   // phase 2: exclusive prefix sum (start offsets)
groupshared uint gs_scatter[64];   // phase 3: scatter write counters
groupshared int  gs_cluster[128];  // resolved primary cluster ID per token

[numthreads(128, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint i = tid.x;

    // ── Phase 0: initialise groupshared ──────────────────────────────────────
    if (i < 64) {
        gs_count  [i] = 0;
        gs_start  [i] = 0;
        gs_scatter[i] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // ── Phase 1: count tokens per cluster ────────────────────────────────────
    if (i < seq_len) {
        int cid = bone_ids[i * 4];  // primary bone = dominant cluster
        if (cid < 0 || cid >= (int)n_clusters) cid = 0;
        gs_cluster[i] = cid;
        InterlockedAdd(gs_count[(uint)cid], 1);
    }
    GroupMemoryBarrierWithGroupSync();

    // ── Phase 2: exclusive prefix sum → cluster start offsets (single thread) ─
    if (i == 0) {
        uint offset = 0;
        for (uint c = 0; c < n_clusters; ++c) {
            gs_start[c] = offset;
            offset += gs_count[c];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Seed scatter counters from start offsets
    if (i < 64) gs_scatter[i] = gs_start[i];
    GroupMemoryBarrierWithGroupSync();

    // ── Phase 3: scatter token indices into sorted output ────────────────────
    if (i < seq_len) {
        uint pos;
        InterlockedAdd(gs_scatter[(uint)gs_cluster[i]], 1, pos);
        sorted_idx[pos] = i;
    }
    GroupMemoryBarrierWithGroupSync();

    // ── Phase 4: write cluster tables ────────────────────────────────────────
    if (i < n_clusters) {
        cluster_start[i] = gs_start[i];
        cluster_count[i] = gs_count[i];
    }
}
