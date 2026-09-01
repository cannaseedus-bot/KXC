#version 300 es
precision highp float;
uniform sampler2D u_input_0;
uniform ivec2 u_shape;
uniform float u_low_threshold;
uniform float u_high_threshold;
out vec4 fragColor;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (p.x >= u_shape.x || p.y >= u_shape.y) discard;
    float value = texelFetch(u_input_0,p,0).r;
    bool connected = value > u_high_threshold;
    if (!connected && value > u_low_threshold) {
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                ivec2 q = p + ivec2(kx, ky);
                if (q.x >= 0 && q.y >= 0 &&
                    q.x < u_shape.x && q.y < u_shape.y &&
                    texelFetch(u_input_0, q, 0).r > u_high_threshold) {
                    connected = true;
                }
            }
        }
    }
    float outputValue = connected ? 1.0 : 0.0;
    fragColor = vec4(outputValue,0.0,0.0,1.0);
}
