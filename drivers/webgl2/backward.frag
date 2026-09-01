#version 300 es
precision highp float;
uniform highp sampler2D u_input_0;
uniform ivec2 u_shape;
out vec4 fragColor;
void main() {
    ivec2 ij = ivec2(gl_FragCoord.xy);
    if (ij.x >= u_shape.x || ij.y >= u_shape.y) discard;
    // The backward glyph consumes the upstream gradient and preserves its
    // shape; parameter-specific accumulation is a separate training pass.
    fragColor = texelFetch(u_input_0, ij, 0);
}
