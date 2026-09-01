// arc_replay_kernel.hlsl — iGPU Parallel ARC Replay
//
// Replays thousands of geodesic ARCs simultaneously.
// Each thread = one ARC replay: traverse path, add noise, evaluate, update entropy.
//
// Intel HD 4600: 96 EUs × 64 threads = 6144 parallel ARC replays per dispatch.
// cs_5_0 DXBC (no DXIL — HD 4600 compatible).
//
// K'UHUL phase: Sek (all replay happens here), Ch'en (entropy update written back).
//
// Connection to replayable-arcs.js:
//   arcStartPoints / arcEndPoints = ReplayableArc.start / .end
//   arcEntropies                  = ReplayableArc.entropies
//   replayOutputs[arcIdx][iter]   = quality score from _evalQuality()
//   After dispatch, JS reads replayOutputs and calls clearFogFromSuccesses()

RWStructuredBuffer<float4> arcStartPoints    : register(u0);  // start on sphere
RWStructuredBuffer<float4> arcEndPoints      : register(u1);  // end on sphere
RWStructuredBuffer<float4> arcPaths          : register(u2);  // recorded waypoints
RWStructuredBuffer<float>  arcEntropies      : register(u3);  // entropy per waypoint
RWStructuredBuffer<float>  replayOutputs     : register(u4);  // quality per arc per iter

cbuffer ReplayParams : register(b0) {
    uint   numArcs;
    uint   numSteps;         // waypoints per arc
    float  explorationNoise; // max perturbation (scaled by entropy)
    uint   replayIteration;  // current iteration index
    float  radius;           // sphere radius
    float  fogReduction;     // entropy reduction per successful replay (0.05)
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

float GeodesicDist(float3 p, float3 q, float R) {
    float d = dot(p,q)/(R*R);
    return R * acos(clamp(d,-1.0f,1.0f));
}

float3 GeodesicInterp(float3 from, float3 to, float t, float R) {
    // Slerp on sphere
    float cosA = dot(from,to)/(R*R);
    cosA = clamp(cosA,-1.0f,1.0f);
    float A = acos(cosA);
    if (A < 1e-6f) return from;
    return (sin((1-t)*A)*from + sin(t*A)*to) / sin(A);
}

// Pseudo-random noise (Wang hash)
float Rand(uint seed) {
    seed ^= seed >> 16; seed *= 0x45d9f3b; seed ^= seed >> 16;
    return float(seed) / 4294967295.0f;
}

// ─── Main kernel ─────────────────────────────────────────────────────────────

[numthreads(64, 1, 1)]
void ReplayArc(uint3 id : SV_DispatchThreadID) {
    uint arcIdx = id.x;
    if (arcIdx >= numArcs) return;

    float4 start = arcStartPoints[arcIdx];
    float4 end   = arcEndPoints[arcIdx];
    float  R     = radius;

    // Replay: traverse from start to end with noise scaled by entropy
    float  totalLength = 0.0f;
    float  totalEntropy = 0.0f;
    float3 prev = start.xyz;

    for (uint step = 0; step <= numSteps; step++) {
        float  t     = float(step) / float(numSteps);
        float3 point = GeodesicInterp(start.xyz, end.xyz, t, R);

        // Load entropy at this waypoint
        uint  eIdx   = arcIdx * (numSteps + 1) + step;
        float entropy = arcEntropies[eIdx];
        totalEntropy += entropy;

        // Add exploration noise (higher entropy = more noise)
        float noise = explorationNoise * entropy;
        uint  seed  = arcIdx * 65537u + step * 131u + replayIteration;
        float nx    = (Rand(seed)     - 0.5f) * 2.0f * noise;
        float ny    = (Rand(seed+1u)  - 0.5f) * 2.0f * noise;
        float nz    = (Rand(seed+2u)  - 0.5f) * 2.0f * noise;
        point += float3(nx, ny, nz);
        point  = normalize(point) * R;

        // Accumulate path length
        totalLength += GeodesicDist(prev, point, R);
        prev = point;

        // Write perturbed path back
        uint  pIdx = arcIdx * (numSteps + 1) + step;
        arcPaths[pIdx] = float4(point, entropy);

        // Reduce entropy on this waypoint if step is deterministic
        if (noise < 0.01f) {
            arcEntropies[eIdx] = max(0.0f, entropy * (1.0f - fogReduction));
        }
    }

    // Compute replay quality: short path + low entropy = high quality
    float meanEntropy   = totalEntropy / float(numSteps + 1);
    float lengthScore   = 1.0f / (1.0f + totalLength);
    float entropyScore  = 1.0f - meanEntropy;
    float quality       = 0.5f * lengthScore + 0.5f * entropyScore;

    // Write quality for this iteration
    uint outIdx = arcIdx * (replayIteration + 1) + replayIteration;
    replayOutputs[outIdx] = quality;

    // If quality is high, clear more fog along path (Ch'en write-back)
    if (quality > 0.6f) {
        float reduction = fogReduction * quality;
        for (uint step2 = 0; step2 <= numSteps; step2++) {
            uint ei = arcIdx * (numSteps + 1) + step2;
            arcEntropies[ei] = max(0.0f, arcEntropies[ei] * (1.0f - reduction));
        }
    }
}
