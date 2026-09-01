#version 300 es
precision highp float;

uniform sampler2D u_hidden;
uniform sampler2D u_gate;
uniform ivec2 u_hidden_size;
uniform float u_residual_scale;
out vec4 out_value;

// Browser projection of the fold/node expert residual. The native OpenCL
// expert computes the gate; WebGL2 applies the correction to tensor texels.
void main() {
    ivec2 texel = ivec2(gl_FragCoord.xy);
    vec4 hidden = texelFetch(u_hidden, texel, 0);
    float gate = texelFetch(u_gate, ivec2(0, texel.y), 0).r;
    out_value = hidden + hidden * (gate * u_residual_scale);
}
