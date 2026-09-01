#version 300 es
precision highp float;

uniform sampler2D u_hidden;
uniform sampler2D u_face_scores;
uniform float u_residual_scale;
out vec4 out_value;

// Shared micronaut face projection. The host selects the face from result.json
// and uploads bounded face scores; this shader never mutates model weights.
void main() {
    ivec2 texel = ivec2(gl_FragCoord.xy);
    vec4 hidden = texelFetch(u_hidden, texel, 0);
    float face_score = texelFetch(u_face_scores, ivec2(0, texel.y), 0).r;
    float gate = clamp(face_score, -1.0, 1.0);
    out_value = hidden + hidden * (gate * u_residual_scale);
}
