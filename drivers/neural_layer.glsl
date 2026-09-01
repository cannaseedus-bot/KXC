// neural_layer.glsl — dense forward + ReLU for OpenGL 4.3 compute
// Mirrors neural_layer.wgsl; targets Intel HD 4600 via ig75icd64.dll

#version 430 core

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer InputBuffer {
    float input_buf[];
};

layout(std430, binding = 1) readonly buffer WeightBuffer {
    float weight_buf[];
};

layout(std430, binding = 2) readonly buffer BiasBuffer {
    float bias_buf[];
};

layout(std430, binding = 3) writeonly buffer OutputBuffer {
    float out_buf[];
};

layout(std140, binding = 4) uniform Params {
    uint in_dim;
    uint out_rows;
    uint out_cols;
} cb;

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;

    if (row >= cb.out_rows || col >= cb.out_cols) {
        return;
    }

    float acc = 0.0f;
    for (uint k = 0u; k < cb.in_dim; k++) {
        float a = input_buf[row * cb.in_dim + k];
        float w = weight_buf[col * cb.in_dim + k];
        acc += a * w;
    }

    acc += bias_buf[col];
    if (acc < 0.0f) {
        acc = 0.0f;
    }

    out_buf[row * cb.out_cols + col] = acc;
}
