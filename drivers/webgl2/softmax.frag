#version 300 es
precision highp float;

uniform highp sampler2D u_input_0;
uniform ivec2 u_shape;
uniform int u_axis;
out vec4 fragColor;

void main() {
    ivec2 ij = ivec2(gl_FragCoord.xy);
    if (ij.x >= u_shape.x || ij.y >= u_shape.y) discard;
    float maximum = -3.402823466e+38;
    for (int k = 0; k < u_shape.y; ++k) maximum = max(maximum, texelFetch(u_input_0, ivec2(ij.x, k), 0).r);
    float denominator = 0.0;
    for (int k = 0; k < u_shape.y; ++k) denominator += exp(texelFetch(u_input_0, ivec2(ij.x, k), 0).r - maximum);
    float value = texelFetch(u_input_0, ij, 0).r;
    fragColor = vec4(exp(value - maximum) / max(denominator, 1e-8), 0.0, 0.0, 1.0);
}
