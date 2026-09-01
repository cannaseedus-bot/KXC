#version 300 es
precision highp float;
uniform highp sampler2D u_input_0;
uniform ivec2 u_shape;
out vec4 fragColor;
void main() {
    ivec2 ij = ivec2(gl_FragCoord.xy);
    if (ij.x >= u_shape.x || ij.y >= u_shape.y) discard;
    float x = texelFetch(u_input_0, ij, 0).r;
    float c = 0.7978845608 * (x + 0.044715 * x * x * x);
    float y = 0.5 * x * (1.0 + tanh(c));
    fragColor = vec4(y, 0.0, 0.0, 1.0);
}
