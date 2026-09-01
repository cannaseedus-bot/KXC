#define K_ROWS 2880
#define WIDTH 2880

Buffer<float> Q : register(t0);
Buffer<float> K : register(t1);
RWBuffer<float> P : register(u0);
groupshared float scores[K_ROWS];
groupshared float maximum;
groupshared float denominator;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint row = gid.x;
    for (uint col = tid.x; col < K_ROWS; col += 256) {
        float score = 0.0f;
        for (uint i = 0; i < WIDTH; ++i)
            score += Q[row * WIDTH + i] * K[col * WIDTH + i];
        scores[col] = score / sqrt((float)WIDTH);
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid.x == 0) {
        maximum = scores[0];
        for (uint col = 1; col < K_ROWS; ++col)
            maximum = max(maximum, scores[col]);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint col = tid.x; col < K_ROWS; col += 256)
        scores[col] = exp(scores[col] - maximum);
    GroupMemoryBarrierWithGroupSync();
    if (tid.x == 0) {
        denominator = 0.0f;
        for (uint col = 0; col < K_ROWS; ++col)
            denominator += scores[col];
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint col = tid.x; col < K_ROWS; col += 256)
        P[row * K_ROWS + col] = scores[col] / denominator;
}
