// OpenCL C 1.2 — entropy-guided ARC replay.
// Each work item owns one step so quality/entropy updates are race-free.
__kernel void arc_replay(
    __global const float* arc_inputs,
    __global float* paths,
    __global float* entropy,
    __global float* quality,
    const int arc_len,
    const int hidden,
    const float temperature)
{
    const int step = (int)get_global_id(0);
    if (step >= arc_len) return;

    const float ent = entropy[step];
    float total_error = 0.0f;
    for (int dim = 0; dim < hidden; ++dim) {
        const int offset = step * hidden + dim;
        const float input = arc_inputs[offset];
        const float reconstructed = input * (1.0f + ent * temperature);
        total_error += fabs(reconstructed - input);
        paths[offset] = reconstructed;
    }
    const float mean_error = total_error / fmax((float)hidden, 1.0f);
    quality[step] = 1.0f / (1.0f + mean_error);
    entropy[step] = ent * 0.95f + 0.05f * mean_error;
}
