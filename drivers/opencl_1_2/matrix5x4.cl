// Matrix5x4 backend: row-major 5 lanes x 4 channels.

__kernel void matrix5x4_mul(__global const float* a, __global const float* b, __global float* out) {
    const int lane = get_global_id(0);
    if (lane >= 5) return;
    const float4 av = vload4(0, a + lane * 4);
    const float4 bv = vload4(0, b + lane * 4);
    vstore4(av * bv, 0, out + lane * 4);
}

__kernel void matrix5x4_add(__global const float* a, __global const float* b, __global float* out) {
    const int lane = get_global_id(0);
    if (lane >= 5) return;
    vstore4(vload4(0, a + lane * 4) + vload4(0, b + lane * 4), 0, out + lane * 4);
}

__kernel void matrix5x4_swiglu(__global const float* x, __global const float* gate, __global float* out) {
    const int lane = get_global_id(0);
    if (lane >= 5) return;
    const float4 xv = vload4(0, x + lane * 4);
    const float4 gv = vload4(0, gate + lane * 4);
    const float4 sigmoid = 1.0f / (1.0f + exp(-gv));
    vstore4(xv * sigmoid, 0, out + lane * 4);
}

// B is row-major 4x5. The output keeps the first four columns of A(5x4)*B(4x5).
__kernel void matrix5x4_matmul(__global const float* a, __global const float* b, __global float* out) {
    const int lane = get_global_id(0);
    const int channel = get_global_id(1);
    if (lane >= 5 || channel >= 4) return;
    float sum = 0.0f;
    for (int k = 0; k < 4; ++k) sum += a[lane * 4 + k] * b[k * 5 + channel];
    out[lane * 4 + channel] = sum;
}
