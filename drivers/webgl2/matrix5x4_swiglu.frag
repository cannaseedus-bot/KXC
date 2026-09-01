#version 300 es
precision highp float;
uniform sampler2D u_input_0;
uniform sampler2D u_input_1;
out vec4 fragColor;
void main() {
    ivec2 ij = ivec2(gl_FragCoord.xy);
    if (ij.x >= 5 || ij.y >= 4) discard;
    float x = texelFetch(u_input_0, ij, 0).r;
    float g = texelFetch(u_input_1, ij, 0).r;
    fragColor = vec4(x / (1.0 + exp(-g)), 0.0, 0.0, 1.0);
}
