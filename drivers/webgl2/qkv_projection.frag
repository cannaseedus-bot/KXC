#version 300 es
precision highp float;
uniform highp sampler2D u_input_0;
uniform highp sampler2D u_input_1;
uniform ivec2 u_shape;
uniform ivec2 u_weight_shape;
out vec4 fragColor;
void main() {
    ivec2 ij = ivec2(gl_FragCoord.xy);
    if (ij.x >= u_shape.x || ij.y >= u_shape.y) discard;
    float sum = 0.0;
    for (int k = 0; k < u_weight_shape.x; ++k)
        sum += texelFetch(u_input_0, ivec2(k, ij.x), 0).r * texelFetch(u_input_1, ivec2(ij.y, k), 0).r;
    fragColor = vec4(sum, 0.0, 0.0, 1.0);
}
