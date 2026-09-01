#define Q_ROWS 2880
#define K_ROWS 2880
#define V_COLS 2880

Buffer<float> P : register(t0);
Buffer<float> V : register(t1);
RWBuffer<float> Out : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint row = id.x;
    uint col = id.y;
    if (row >= Q_ROWS || col >= V_COLS) return;
    float value = 0.0f;
    for (uint i = 0; i < K_ROWS; ++i)
        value += P[row * K_ROWS + i] * V[i * V_COLS + col];
    Out[row * V_COLS + col] = value;
}
