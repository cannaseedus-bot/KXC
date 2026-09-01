#version 300 es
precision highp float;
uniform sampler2D u_input_0;
uniform ivec2 u_shape;
out vec4 fragColor;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (p.x >= u_shape.x || p.y >= u_shape.y) discard;
    float w[9] = float[9](-1.0,0.0,1.0,-2.0,0.0,2.0,-1.0,0.0,1.0);
    float sum = 0.0; int n = 0;
    for (int ky=-1; ky<=1; ++ky) for (int kx=-1; kx<=1; ++kx) {
        ivec2 q = clamp(p + ivec2(kx,ky), ivec2(0), u_shape - ivec2(1));
        sum += w[n++] * texelFetch(u_input_0, q, 0).r;
    }
    fragColor = vec4(sum,0.0,0.0,1.0);
}
