cbuffer VertexUniforms : register(b0) {
    uint positionStride; uint positionOffset;
    uint normalStride;   uint normalOffset;
    uint vertexCount;    float3 packingPadding;  // cbuffer 16-byte alignment
};
ByteAddressBuffer          positions    : register(t0);
ByteAddressBuffer          normals      : register(t1);
ByteAddressBuffer          weights      : register(t2);
Buffer<uint4>              joints       : register(t3);
StructuredBuffer<float4x4>  skinMatrices : register(t4);
RWByteAddressBuffer        outVerts     : register(u0);
float4x4 SkinMatrix(uint i) {
    uint4 j = joints[i];
    float4 w = asfloat(weights.Load4(i * 16));
    float4x4 m = skinMatrices[j.x] * w.x;
    m += skinMatrices[j.y] * w.y;
    m += skinMatrices[j.z] * w.z;
    m += skinMatrices[j.w] * w.w;
    return m;
}
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x; if (i >= vertexCount) { return; }
    float3 p   = asfloat(positions.Load3((i * positionStride + positionOffset) * 4));
    float3 nrm = asfloat(normals.Load3((i * normalStride + normalOffset) * 4));
    float4x4 m = SkinMatrix(i);
    float3 sp = mul(m, float4(p, 1.0f)).xyz;
    float3 sn = mul((float3x3)m, nrm);
    uint b = i * 24;
    outVerts.Store3(b, asuint(sp));
    outVerts.Store3(b + 12, asuint(sn));
}
