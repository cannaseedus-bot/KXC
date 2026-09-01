// OpenCL C 1.2 — shared micronaut face projection.
// The host supplies one bounded score per Semantic Cube face.

__kernel void micronaut_face_projection(
    __global const float* hidden,
    __global const float* face_scores,
    __global float* correction,
    const int token_count,
    const int hidden_size,
    const int face_id,
    const float residual_scale)
{
    const int token = (int)get_global_id(0);
    if (token >= token_count) return;
    const float gate = clamp(face_scores[face_id], -1.0f, 1.0f);
    for (int h = 0; h < hidden_size; ++h) {
        const int index = token * hidden_size + h;
        correction[index] = hidden[index] * gate * residual_scale;
    }
}
