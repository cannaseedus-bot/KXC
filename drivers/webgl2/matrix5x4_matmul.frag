#version 300 es
precision highp float;
uniform sampler2D u_input_0;
uniform sampler2D u_input_1;
out vec4 fragColor;
void main() {
    ivec2 ij = ivec2(gl_FragCoord.xy);
    if (ij.x >= 5 || ij.y >= 4) discard;
    float sum = 0.0;
    for (int k = 0; k < 4; ++k)
        sum += texelFetch(u_input_0, ivec2(k, ij.x), 0).r * texelFetch(u_input_1, ivec2(ij.y, k), 0).r;
    fragColor = vec4(sum, 0.0, 0.0, 1.0);
}
