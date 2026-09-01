// cs_vertex_skin.hlsl — Linear Blend Skinning (LBS) compute shader
// HLSL port of models/khanary-geometry-v0.3.0/kernels/vertex_skin.wgsl
// Applies 4-bone LBS to positions and normals; outputs interleaved [px,py,pz, nx,ny,nz].
//
// Matrix convention: row-major (DirectXMath). Position: mul(float4(p,1), M).
// Normal:  upper-left 3x3 of blended matrix, no w component.
//
// Dispatch(ceil(vertexCount / 64), 1, 1)  numthreads(64, 1, 1)

cbuffer SkinCB : register(b0) {
    uint positionStride;   // floats between successive vertex positions
    uint positionOffset;   // float index of first vertex's first component
    uint normalStride;     // floats between successive vertex normals
    uint normalOffset;     // float index of first vertex's normal
    uint vertexCount;      // total vertex count
    uint pad0, pad1, pad2;
};

StructuredBuffer<float>  positions : register(t0); // [vertexCount * positionStride]
StructuredBuffer<float>  normals   : register(t1); // [vertexCount * normalStride]
StructuredBuffer<float>  boneWts   : register(t2); // [vertexCount * 4] blend weights
StructuredBuffer<uint>   boneJoints: register(t3); // [vertexCount * 4] joint indices
StructuredBuffer<float4> skinMats  : register(t4); // [nBones * 4] — 4 rows per bone (row-major)

RWStructuredBuffer<float> outVerts : register(u0); // [vertexCount * 6] = [px,py,pz,nx,ny,nz]

// Fetch bone matrix as float4x4 from the flat row-array layout
float4x4 fetchMatrix(uint joint) {
    uint b = joint * 4u;
    return float4x4(skinMats[b], skinMats[b + 1u], skinMats[b + 2u], skinMats[b + 3u]);
}

// Weighted blend of 4 bone matrices for vertex i
float4x4 blendMatrix(uint i) {
    uint jb = i * 4u;
    float4x4 m  = fetchMatrix(boneJoints[jb])      * boneWts[jb];
    m += fetchMatrix(boneJoints[jb + 1u]) * boneWts[jb + 1u];
    m += fetchMatrix(boneJoints[jb + 2u]) * boneWts[jb + 2u];
    m += fetchMatrix(boneJoints[jb + 3u]) * boneWts[jb + 3u];
    return m;
}

[numthreads(64, 1, 1)]
void CSMain(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= vertexCount) return;

    uint pb = i * positionStride + positionOffset;
    float3 p = float3(positions[pb], positions[pb + 1u], positions[pb + 2u]);

    uint nb = i * normalStride + normalOffset;
    float3 nrm = float3(normals[nb], normals[nb + 1u], normals[nb + 2u]);

    float4x4 m = blendMatrix(i);

    // Position: row-vector × row-major matrix → (p,1) * M, take xyz
    float3 sp = mul(float4(p, 1.0f), m).xyz;

    // Normal: upper-left 3×3 of row-major matrix (rotation only, no translation)
    float3x3 m3 = float3x3(
        m._m00, m._m01, m._m02,
        m._m10, m._m11, m._m12,
        m._m20, m._m21, m._m22
    );
    float3 sn = mul(nrm, m3);

    uint ob = i * 6u;
    outVerts[ob]      = sp.x;
    outVerts[ob + 1u] = sp.y;
    outVerts[ob + 2u] = sp.z;
    outVerts[ob + 3u] = sn.x;
    outVerts[ob + 4u] = sn.y;
    outVerts[ob + 5u] = sn.z;
}
