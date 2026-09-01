#version 300 es
precision highp float;
uniform sampler2D u_input_0;
uniform sampler2D u_input_1;
out vec4 fragColor;
void main() {
    ivec2 ij = ivec2(gl_FragCoord.xy);
    if (ij.x >= 5 || ij.y >= 4) discard;
    fragColor = vec4(texelFetch(u_input_0, ij, 0).r * texelFetch(u_input_1, ij, 0).r, 0.0, 0.0, 1.0);
}
