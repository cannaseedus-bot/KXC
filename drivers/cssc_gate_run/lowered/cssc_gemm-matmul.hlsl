// Generated HLSL from KLSL manifest shader: matmul
cbuffer Params : register(b0) {
  uint M;
  uint N;
  uint K;
};

StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 global_id : SV_DispatchThreadID,
            uint3 local_id : SV_GroupThreadID,
            uint3 workgroup_id : SV_GroupID) {
uint col = global_id.x;
uint row = global_id.y;
if (row >= M) { return; }
if (col >= N) { return; }
float acc = 0.0;
for (uint k = 0u; k < K; k = k + 1u) { acc = acc + A[row * K + k] * B[k * N + col]; }
C[row * N + col] = acc;
}
