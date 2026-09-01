#define TS 16
StructuredBuffer<float>   A : register(t0);   // [M,K] row-major
StructuredBuffer<float>   B : register(t1);   // [K,N] row-major
RWStructuredBuffer<float> C : register(u0);   // [M,N] row-major
cbuffer GemmCB : register(b0) { uint M; uint N; uint K; uint _pad; };
groupshared float As[TS][TS];
groupshared float Bs[TS][TS];
[numthreads(TS, TS, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID) {
    uint row = dtid.y, col = dtid.x;
    float acc = 0.0f;
    uint nT = (K + TS - 1) / TS;
    for (uint t = 0; t < nT; ++t) {
        uint aC = t * TS + lid.x, bR = t * TS + lid.y;
        As[lid.y][lid.x] = (row < M && aC < K) ? A[row * K + aC] : 0.0f;
        Bs[lid.y][lid.x] = (bR < K && col < N) ? B[bR * N + col] : 0.0f;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint k = 0; k < TS; ++k) acc += As[lid.y][k] * Bs[k][lid.x];
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
