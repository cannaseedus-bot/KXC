// OpenCL C 1.2 — deterministic KSON fold/node expert.
// Produces a small residual correction from fold and node scores.
// Buffers are row-major: [token, feature].

__kernel void kson_fold_node_gate(
    __global const float* hidden,
    __global const float* fold_scores,
    __global const float* node_scores,
    __global float* correction,
    const int token_count,
    const int hidden_size,
    const int fold_count,
    const int node_count)
{
    const int token = (int)get_global_id(0);
    if (token >= token_count) return;

    float fold_max = -FLT_MAX;
    for (int f = 0; f < fold_count; ++f) {
        fold_max = fmax(fold_max, fold_scores[token * fold_count + f]);
    }

    float node_max = -FLT_MAX;
    for (int n = 0; n < node_count; ++n) {
        node_max = fmax(node_max, node_scores[token * node_count + n]);
    }

    // Bounded residual gate: the expert can correct flow without replacing
    // the language-model hidden state.
    const float gate = clamp(0.5f * (fold_max + node_max), -1.0f, 1.0f);
    for (int h = 0; h < hidden_size; ++h) {
        const int index = token * hidden_size + h;
        correction[index] = hidden[index] * (0.1f * gate);
    }
}
