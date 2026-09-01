#version 300 es
precision highp float;

uniform highp sampler2D u_input_0;
uniform highp sampler2D u_input_1;
uniform ivec2 u_A_shape; // [M,K]
uniform ivec2 u_B_shape; // [K,N]
uniform ivec2 u_C_shape; // [M,N]
out vec4 fragColor;

void main() {
    ivec2 ij = ivec2(gl_FragCoord.xy);
    if (ij.x >= u_C_shape.x || ij.y >= u_C_shape.y) discard;
    float sum = 0.0;
    for (int k = 0; k < u_A_shape.y; ++k) {
        sum += texelFetch(u_input_0, ivec2(ij.x, k), 0).r * texelFetch(u_input_1, ivec2(k, ij.y), 0).r;
    }
    fragColor = vec4(sum, 0.0, 0.0, 1.0);
}
