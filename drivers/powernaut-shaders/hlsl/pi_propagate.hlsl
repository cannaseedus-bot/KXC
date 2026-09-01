/**
 * pi_propagate.hlsl — π-KUHUL graph diffusion pass
 *
 * One step of field diffusion over the mesh:
 *   field_out(i) = α · field_in(i) + (1-α) · Σ_j w_ij · field_in(j)
 *
 * where j ranges over neighbours of i, and w_ij = 1 / count.
 *
 * Originated: .llama.cpp/tools/server/shaders/pi_propagate.hlsl
 * Migrated to: kuhul/shaders/hlsl/pi_propagate.hlsl
 */

#pragma pack_matrix(row_major)

struct Edge { uint a; uint b; float distance; float _pad; };
struct NodeAdjacency { uint offset; uint count; };
struct NodeResult { float value; float geodesic_min; float2 _pad; };

cbuffer PropParams : register(b0) {
    uint  nodeCount;
    uint  edgeCount;
    float retain;
    float _pad;
};

StructuredBuffer<Edge>          edges     : register(t1);
StructuredBuffer<NodeAdjacency> adjacency : register(t2);
StructuredBuffer<NodeResult>    field_in  : register(t4);
RWStructuredBuffer<NodeResult>  field_out : register(u1);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= nodeCount) return;

    float own   = field_in[i].value;
    float geo   = field_in[i].geodesic_min;
    float accum = 0.0f;
    uint  cnt   = min(adjacency[i].count, 8u);
    uint  off   = adjacency[i].offset;

    [loop] for (uint e = 0u; e < cnt; e++) {
        uint idx = off + e;
        if (idx >= edgeCount) break;
        accum += field_in[edges[idx].b].value;
    }

    float neighbour_mean = (cnt > 0u) ? (accum / float(cnt)) : own;

    NodeResult r;
    r.value        = retain * own + (1.0f - retain) * neighbour_mean;
    r.geodesic_min = geo;
    r._pad         = float2(0.0f, 0.0f);
    field_out[i]   = r;
}
