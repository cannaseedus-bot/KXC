// OpenCL C 1.2 — batched metric geodesic distance.
// metric is [batch, hidden, hidden], positions is [batch, hidden].
__kernel void geodesic_distance(
    __global const float* metric,
    __global const float* positions,
    __global float* distances,
    const int batch,
    const int hidden)
{
    const int i = (int)get_global_id(0);
    const int j = (int)get_global_id(1);
    if (i >= batch || j >= batch) return;

    float sum = 0.0f;
    for (int k = 0; k < hidden; ++k) {
        const float diff_k = positions[i * hidden + k] - positions[j * hidden + k];
        for (int l = 0; l < hidden; ++l) {
            const float diff_l = positions[i * hidden + l] - positions[j * hidden + l];
            sum += diff_k * metric[i * hidden * hidden + k * hidden + l] * diff_l;
        }
    }
    distances[i * batch + j] = sqrt(fmax(sum, 0.0f) + 1e-8f);
}
