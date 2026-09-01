cbuffer VertexUniforms : register(b0)
{
    uint positionStride;
    uint positionOffset;
    uint normalStride;
    uint normalOffset;
    uint vertexCount;
    float3 packingPadding;
};
ByteAddressBuffer   positions    : register(t0);
ByteAddressBuffer   normals      : register(t1);
ByteAddressBuffer   weights      : register(t2);
Buffer<uint4>       joints       : register(t3);
StructuredBuffer<float4x4> skinMatrices        : register(t4);
StructuredBuffer<float4x4> inverseBindMatrices : register(t5);
RWByteAddressBuffer outVertsScalar : register(u0);
float3 GetPosition(uint index){ uint b=(index*positionStride+positionOffset)*4; return asfloat(positions.Load3(b)); }
float3 GetNormal(uint index){ uint b=(index*normalStride+normalOffset)*4; return asfloat(normals.Load3(b)); }
float4x4 GetSkinMatrix(uint index){
    uint4 j=joints[index]; uint wo=index*4*4; float4 w=asfloat(weights.Load4(wo));
    float4x4 m=skinMatrices[j.x]*w.x; m+=skinMatrices[j.y]*w.y; m+=skinMatrices[j.z]*w.z; m+=skinMatrices[j.w]*w.w; return m;
}
[numthreads(64,1,1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID){
    uint index=dispatchThreadID.x; if(index>=vertexCount){return;}
    float3 rawPos=GetPosition(index); float3 rawNorm=GetNormal(index); float4x4 skinMatrix=GetSkinMatrix(index);
    float3 skinnedPos=mul(skinMatrix,float4(rawPos,1.0f)).xyz;
    float3 skinnedNorm=mul((float3x3)skinMatrix,rawNorm);
    uint outByteBase=index*24;
    outVertsScalar.Store3(outByteBase, asuint(skinnedPos));
    outVertsScalar.Store3(outByteBase+12, asuint(skinnedNorm));
}
