// Sobel convolution family for OpenCL C 1.2.

inline int clamp_index(int value, int limit) { return min(max(value, 0), limit - 1); }

__kernel void sobel_x(__global const float* input, __global float* output,
                      const int height, const int width) {
    const int row = get_global_id(0), col = get_global_id(1);
    if (row >= height || col >= width) return;
    const int weights[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    float sum = 0.0f;
    for (int ky = -1; ky <= 1; ++ky)
        for (int kx = -1; kx <= 1; ++kx)
            sum += (float)weights[ky + 1][kx + 1] *
                   input[clamp_index(row + ky, height) * width + clamp_index(col + kx, width)];
    output[row * width + col] = sum;
}

__kernel void sobel_y(__global const float* input, __global float* output,
                      const int height, const int width) {
    const int row = get_global_id(0), col = get_global_id(1);
    if (row >= height || col >= width) return;
    const int weights[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    float sum = 0.0f;
    for (int ky = -1; ky <= 1; ++ky)
        for (int kx = -1; kx <= 1; ++kx)
            sum += (float)weights[ky + 1][kx + 1] *
                   input[clamp_index(row + ky, height) * width + clamp_index(col + kx, width)];
    output[row * width + col] = sum;
}

__kernel void sobel_magnitude(__global const float* gx, __global const float* gy,
                              __global float* output, const int height, const int width) {
    const int row = get_global_id(0), col = get_global_id(1);
    if (row >= height || col >= width) return;
    const int index = row * width + col;
    output[index] = min(sqrt(gx[index] * gx[index] + gy[index] * gy[index]), 1.0f);
}

__kernel void sobel_threshold(__global const float* magnitude, __global float* output,
                              const int height, const int width,
                              const float low_threshold, const float high_threshold) {
    const int row = get_global_id(0), col = get_global_id(1);
    if (row >= height || col >= width) return;
    const float value = magnitude[row * width + col];
    int connected = value > high_threshold;
    if (!connected && value > low_threshold) {
        for (int ky = -1; ky <= 1; ++ky)
            for (int kx = -1; kx <= 1; ++kx) {
                const int r = row + ky, c = col + kx;
                if (r >= 0 && r < height && c >= 0 && c < width &&
                    magnitude[r * width + c] > high_threshold) connected = 1;
            }
    }
    output[row * width + col] = connected ? 1.0f : 0.0f;
}

__kernel void elementwise_mul(__global const float* a, __global const float* b,
                              __global float* output, const int count) {
    const int index = get_global_id(0);
    if (index < count) output[index] = a[index] * b[index];
}

__kernel void reduce_pair_sum(__global const float* a, __global const float* b,
                              __global float* output, const int count) {
    const int index = get_global_id(0);
    if (index < count) output[index] = a[index] + b[index];
}
