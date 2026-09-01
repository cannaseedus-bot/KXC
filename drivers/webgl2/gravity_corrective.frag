#version 300 es
precision highp float;

// π-KUHUL bounded weight-field corrective.
// u_weights and u_gradients are RGBA32F textures: four gradient lanes/texel.
uniform sampler2D u_weights;
uniform sampler2D u_gradients;
uniform float u_scale;
uniform float u_clip;
uniform float u_gradient_rms;

out vec4 out_gradient;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 weight = texelFetch(u_weights, p, 0);
    vec4 gradient = texelFetch(u_gradients, p, 0);

    // U(w)=0.5*w², F=-∇U=-w.  The optimizer-facing correction is +scale*w.
    vec4 correction = weight * u_scale;
    float limit = max(u_gradient_rms, 1e-8) * u_clip;
    correction = clamp(correction, vec4(-limit), vec4(limit));
    out_gradient = gradient + correction;
}
