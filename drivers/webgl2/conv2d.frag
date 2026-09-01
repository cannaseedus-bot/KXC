#version 300 es
precision highp float;

uniform highp sampler2D u_input_0;
uniform highp sampler2D u_input_1;
uniform ivec2 u_shape;
uniform ivec2 u_kernel_shape;
uniform ivec2 u_stride;
uniform ivec2 u_padding;
out vec4 fragColor;

void main() {
    ivec2 out_coord = ivec2(gl_FragCoord.xy);
    if (out_coord.x >= u_shape.x || out_coord.y >= u_shape.y) discard;
    float sum = 0.0;
    for (int ky = 0; ky < u_kernel_shape.y; ++ky) {
        for (int kx = 0; kx < u_kernel_shape.x; ++kx) {
            ivec2 input_coord = out_coord * u_stride + ivec2(kx, ky) - u_padding;
            if (input_coord.x >= 0 && input_coord.y >= 0 &&
                input_coord.x < textureSize(u_input_0, 0).x &&
                input_coord.y < textureSize(u_input_0, 0).y) {
                float x = texelFetch(u_input_0, input_coord, 0).r;
                float w = texelFetch(u_input_1, ivec2(kx, ky), 0).r;
                sum += x * w;
            }
        }
    }
    fragColor = vec4(sum, 0.0, 0.0, 1.0);
}
