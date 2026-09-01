cbuffer XformCB : register(b0) {
    row_major float4x4 M;
    uint vertexCount;
    uint3 _pad;
};
ByteAddressBuffer   inPos  : register(t0);   // tight float3 stream (12 B/vertex)
RWByteAddressBuffer outPos : register(u0);
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= vertexCount) { return; }
    float3 p = asfloat(inPos.Load3(i * 12));
    float3 q = mul(M, float4(p, 1.0f)).xyz;   // manifold transform
    outPos.Store3(i * 12, asuint(q));
}
