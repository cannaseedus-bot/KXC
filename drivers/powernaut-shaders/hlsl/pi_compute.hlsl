/**
 * pi_compute.hlsl — π-KUHUL parallel field evaluator
 *
 * Computes a π-weighted scalar field over a mesh by accumulating
 * source contributions with geodesic RBF decay:
 *
 *   field(i) = Σ_s  (focus_s + motion_s - safety_s) · exp(-d(i,s)² / 2σ²)
 *
 * where d(i,s) is geodesic distance from node i to source node s,
 * computed by bounded BFS over the sorted adjacency edge buffer.
 *
 * Buffers:
 *   b0  PiParams           — constant params
 *   t0  Node[]             — position + weight per node
 *   t1  Edge[]             — (a, b, dist) sorted by a, then b
 *   t2  NodeAdjacency[]    — (offset, count) into edge buffer per node
 *   t3  PiTensor[]         — source field descriptors
 *   u0  NodeResult[]       — output: value + geodesic_min per node
 *
 * Originated: .llama.cpp/tools/server/shaders/pi_compute.hlsl
 * Migrated to: kuhul/shaders/hlsl/pi_compute.hlsl
 */

#pragma pack_matrix(row_major)

struct Node {
    float3 position;
    float  weight;
};

struct Edge {
    uint  a;
    uint  b;
    float distance;
    float _pad;
};

struct NodeAdjacency {
    uint offset;
    uint count;
};

struct PiTensor {
    float focus;
    float motion;
    float safety;
    float entropy;
    uint  node_index;
    uint3 _pad;
};

struct NodeResult {
    float value;
    float geodesic_min;
    float2 _pad;
};

cbuffer PiParams : register(b0) {
    uint  nodeCount;
    uint  edgeCount;
    uint  sourceCount;
    float sigma;
};

StructuredBuffer<Node>          nodes     : register(t0);
StructuredBuffer<Edge>          edges     : register(t1);
StructuredBuffer<NodeAdjacency> adjacency : register(t2);
StructuredBuffer<PiTensor>      sources   : register(t3);
RWStructuredBuffer<NodeResult>  results   : register(u0);

#define MAX_FRONTIER  64u
#define MAX_NEIGHBORS  8u
#define FLT_INF        3.402823466e+38f

float geodesic_bfs(uint start, uint target) {
    if (start == target) return 0.0f;

    uint  queue[MAX_FRONTIER];
    float qdist[MAX_FRONTIER];
    uint  seen_id[MAX_FRONTIER];
    uint  qhead = 0u;
    uint  qtail = 0u;
    uint  seen_count = 0u;

    queue[qtail]      = start;
    qdist[qtail]      = 0.0f;
    seen_id[seen_count] = start;
    qtail++;
    seen_count++;

    [loop] while (qhead < qtail) {
        uint  cur  = queue[qhead];
        float dcur = qdist[qhead];
        qhead++;

        if (cur == target) return dcur;

        uint off = adjacency[cur].offset;
        uint cnt = min(adjacency[cur].count, MAX_NEIGHBORS);

        [loop] for (uint e = 0u; e < cnt; e++) {
            uint idx = off + e;
            if (idx >= edgeCount) break;

            uint  nb = edges[idx].b;
            float dn = dcur + edges[idx].distance;

            bool found = false;
            [loop] for (uint k = 0u; k < seen_count; k++) {
                if (seen_id[k] == nb) { found = true; break; }
            }

            if (!found && qtail < MAX_FRONTIER && seen_count < MAX_FRONTIER) {
                seen_id[seen_count] = nb;
                seen_count++;
                queue[qtail] = nb;
                qdist[qtail] = dn;
                qtail++;
            }
        }
    }

    return FLT_INF;
}

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= nodeCount) return;

    float total   = 0.0f;
    float geo_min = FLT_INF;
    float sigma2  = max(sigma * sigma, 1e-6f);

    [loop] for (uint s = 0u; s < sourceCount; s++) {
        float d = geodesic_bfs(i, sources[s].node_index);

        if (d < geo_min) geo_min = d;

        float decay = (d < FLT_INF) ? exp(-d * d / (2.0f * sigma2)) : 0.0f;

        float val = (sources[s].focus + sources[s].motion - sources[s].safety) * decay;
        total += val;
    }

    results[i].value        = total;
    results[i].geodesic_min = (geo_min < FLT_INF) ? geo_min : -1.0f;
    results[i]._pad         = float2(0.0f, 0.0f);
}
