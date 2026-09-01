#version 300 es
precision highp float;
uniform sampler2D u_input_0;
uniform sampler2D u_input_1;
uniform ivec2 u_shape;
out vec4 fragColor;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (p.x >= u_shape.x || p.y >= u_shape.y) discard;
    fragColor = vec4(texelFetch(u_input_0,p,0).r + texelFetch(u_input_1,p,0).r,0.0,0.0,1.0);
}
