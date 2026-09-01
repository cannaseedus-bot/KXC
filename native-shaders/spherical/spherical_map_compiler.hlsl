// spherical_map_compiler.hlsl — iGPU Spherical Semantic Map Compiler
//
// Runs on Intel HD 4600 iGPU (cs_5_0 DXBC).
// Each thread computes one point's full geodesic neighbourhood.
// 96 EUs × 64 threads = 6144 parallel geodesic computations.
//
// What the map provides:
//   position   = WHERE a concept lives on the semantic sphere
//   metric     = LOCAL geometry at that point (how space curves there)
//   geodesic   = HOW FAR to every neighbour along great-circle arcs
//   exp_cache  = PRECOMPUTED exponential maps (8 cardinal directions)
//
// K'UHUL phase mapping:
//   Pop   = allocate buffers, set SphericalParams
//   Wo    = bind UAVs (position, metric, geodesic, expmap)
//   Sek   = dispatch this shader (parallel compile)
//   Ch'en = readback geodesicCache → CPU fine-tuner can query zero-copy
//   Xul   = close fold, record compile time metric
//
// Connection to pi_kuhul_bridge.h:
//   GeoSphereToTokens() feeds this shader's positionBuffer
//   After dispatch, geodesicCache = GeodesicDistance table
//   Token.phase = longitude mapped from expmap output

RWStructuredBuffer<float4> positionBuffer  : register(u0);  // Points on sphere
RWStructuredBuffer<float4> tangentBuffer   : register(u1);  // Tangent vectors
RWStructuredBuffer<float>  metricBuffer    : register(u2);  // 9 floats per point (3×3)
RWStructuredBuffer<float>  geodesicCache  : register(u3);  // 3 floats per direction × 8 + neighbour distances

cbuffer SphericalParams : register(b0) {
    float  curvature;       // κ  (default 0.1)
    float  radius;          // R = 1/sqrt(κ)
    uint   pointCount;
    uint   neighbourCount;  // how many nearest neighbours to compute (32 max)
    uint   expDirs;         // number of precomputed exp-map directions (8)
};

// ─── Metric tensor ────────────────────────────────────────────────────────────

float3x3 SphericalMetric(float4 p, float R) {
    float theta = acos(clamp(p.z / R, -1.0f, 1.0f));
    float sinT  = sin(theta);
    float R2    = R * R;
    // g = R² diag(1, sin²θ, sin²θ)
    float3x3 g;
    g[0] = float3(R2, 0, 0);
    g[1] = float3(0, R2 * sinT * sinT, 0);
    g[2] = float3(0, 0, R2 * sinT * sinT);
    return g;
}

// ─── Geodesic distance ────────────────────────────────────────────────────────

float GeodesicDist(float4 p, float4 q, float R) {
    float d = dot(p.xyz, q.xyz) / (R * R);
    return R * acos(clamp(d, -1.0f, 1.0f));
}

// ─── Exponential map: exp_p(v) ────────────────────────────────────────────────

float3 ExpMap(float3 p, float3 v, float R) {
    float nv = length(v);
    if (nv < 1e-6f) return p;
    return cos(nv / R) * p + sin(nv / R) * (v / nv);
}

// ─── 8 cardinal tangent directions ───────────────────────────────────────────

float3 TangentDir(uint dir, float3 p) {
    // Construct an orthonormal frame at p and return one of 8 directions
    float3 up    = abs(p.z) < 0.9f ? float3(0,0,1) : float3(1,0,0);
    float3 e1    = normalize(cross(p, up));
    float3 e2    = normalize(cross(p, e1));
    float  angle = dir * 3.14159265f / 4.0f;  // 0,45,90,135,180,225,270,315 deg
    return cos(angle) * e1 + sin(angle) * e2;
}

// ─── Parallel transport along geodesic ───────────────────────────────────────

float3 ParallelTransport(float3 v, float3 from, float3 to, float R) {
    float3 axis  = cross(from, to);
    float  al    = length(axis);
    if (al < 1e-6f) return v;
    axis /= al;
    float dist   = GeodesicDist(float4(from,0), float4(to,0), R);
    float angle  = dist / R;
    float c = cos(angle), s = sin(angle);
    return c * v + s * cross(axis, v) + (1.0f - c) * dot(axis, v) * axis;
}

// ─── Compute entry point ──────────────────────────────────────────────────────

[numthreads(64, 1, 1)]
void CompileSphericalMap(uint3 id : SV_DispatchThreadID) {
    uint pi = id.x;
    if (pi >= pointCount) return;

    float4 p = positionBuffer[pi];
    float  R = radius;

    // 1. Metric tensor (9 floats)
    float3x3 g = SphericalMetric(p, R);
    uint mBase = pi * 9;
    metricBuffer[mBase+0]=g[0][0]; metricBuffer[mBase+1]=g[0][1]; metricBuffer[mBase+2]=g[0][2];
    metricBuffer[mBase+3]=g[1][0]; metricBuffer[mBase+4]=g[1][1]; metricBuffer[mBase+5]=g[1][2];
    metricBuffer[mBase+6]=g[2][0]; metricBuffer[mBase+7]=g[2][1]; metricBuffer[mBase+8]=g[2][2];

    // 2. Exponential map cache: 8 directions × 3 floats
    uint eBase = pi * expDirs * 3;
    for (uint dir = 0; dir < expDirs; dir++) {
        float3 tDir = TangentDir(dir, p.xyz);
        float3 exp  = ExpMap(p.xyz, tDir, R);
        geodesicCache[eBase + dir*3 + 0] = exp.x;
        geodesicCache[eBase + dir*3 + 1] = exp.y;
        geodesicCache[eBase + dir*3 + 2] = exp.z;
    }

    // 3. Geodesic distances to neighbours
    uint dBase = pi * neighbourCount + pointCount * expDirs * 3;
    for (uint nb = 0; nb < neighbourCount && nb < pointCount; nb++) {
        if (nb == pi) { geodesicCache[dBase + nb] = 0.0f; continue; }
        float4 q = positionBuffer[nb];
        geodesicCache[dBase + nb] = GeodesicDist(p, q, R);
    }

    // 4. Store mean tangent direction (Wo declarative output)
    float3 meanTangent = float3(0,0,0);
    for (uint d2 = 0; d2 < expDirs; d2++) meanTangent += TangentDir(d2, p.xyz);
    tangentBuffer[pi] = float4(normalize(meanTangent), 0.0f);
}
