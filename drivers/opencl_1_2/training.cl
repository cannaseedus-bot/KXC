// OpenCL C 1.2 training primitives. All row reductions use one work-item per
// row so this source is deterministic on the HD 4600 without subgroup features.

__kernel void swiglu_forward(__global const float* x, __global const float* gate,
                             __global const float* up, __global float* output, const int size) {
    int i = get_global_id(0);
    if (i >= size) return;
    float g = gate[i];
    output[i] = x[i] * (1.0f / (1.0f + exp(-g))) * up[i];
}

__kernel void silu_forward(__global const float* input, __global float* output, const int size) {
    int i = get_global_id(0);
    if (i < size) { float x = input[i]; output[i] = x / (1.0f + exp(-x)); }
}

__kernel void gelu_forward(__global const float* input, __global float* output, const int size) {
    int i = get_global_id(0);
    if (i < size) { float x = input[i]; output[i] = 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x*x*x))); }
}

__kernel void rms_norm_forward(__global const float* input, __global float* output,
                               const int rows, const int hidden, const float eps) {
    int row = get_global_id(0);
    if (row >= rows) return;
    float sum_sq = 0.0f;
    for (int j = 0; j < hidden; ++j) { float x = input[row * hidden + j]; sum_sq += x * x; }
    float inv_rms = 1.0f / sqrt(sum_sq / (float)hidden + eps);
    for (int j = 0; j < hidden; ++j) output[row * hidden + j] = input[row * hidden + j] * inv_rms;
}

__kernel void softmax_forward(__global const float* input, __global float* output,
                              const int batch, const int classes) {
    int row = get_global_id(0);
    if (row >= batch) return;
    int base = row * classes;
    float maximum = -FLT_MAX;
    for (int c = 0; c < classes; ++c) maximum = max(maximum, input[base + c]);
    float denominator = 0.0f;
    for (int c = 0; c < classes; ++c) denominator += exp(input[base + c] - maximum);
    for (int c = 0; c < classes; ++c) output[base + c] = exp(input[base + c] - maximum) / denominator;
}

__kernel void cross_entropy_loss_back(__global const float* logits, __global const int* labels,
                                      __global float* grad, const int batch, const int classes) {
    int row = get_global_id(0);
    if (row >= batch) return;
    int base = row * classes;
    float maximum = -FLT_MAX;
    for (int c = 0; c < classes; ++c) maximum = max(maximum, logits[base + c]);
    float denominator = 0.0f;
    for (int c = 0; c < classes; ++c) denominator += exp(logits[base + c] - maximum);
    int label = labels[row];
    for (int c = 0; c < classes; ++c)
        grad[base + c] = exp(logits[base + c] - maximum) / denominator - (c == label ? 1.0f : 0.0f);
}

__kernel void rms_norm_back(__global const float* input, __global const float* grad_output,
                            __global float* grad_input, const int rows, const int hidden, const float eps) {
    int row = get_global_id(0);
    if (row >= rows) return;
    int base = row * hidden;
    float sum_sq = 0.0f, dot = 0.0f;
    for (int j = 0; j < hidden; ++j) { sum_sq += input[base+j] * input[base+j]; dot += grad_output[base+j] * input[base+j]; }
    float inv_rms = 1.0f / sqrt(sum_sq / (float)hidden + eps);
    float correction = dot * inv_rms * inv_rms * inv_rms / (float)hidden;
    for (int j = 0; j < hidden; ++j) grad_input[base+j] = grad_output[base+j] * inv_rms - input[base+j] * correction;
}

__kernel void silu_back(__global const float* input, __global const float* grad_output,
                        __global float* grad_input, const int size) {
    int i = get_global_id(0);
    if (i < size) { float s = 1.0f / (1.0f + exp(-input[i])); grad_input[i] = grad_output[i] * (s + input[i] * s * (1.0f - s)); }
}

__kernel void gelu_back(__global const float* input, __global const float* grad_output,
                        __global float* grad_input, const int size) {
    int i = get_global_id(0);
    if (i < size) {
        float x = input[i], x2 = x*x, u = 0.7978845608f * (x + 0.044715f*x*x2);
        float t = tanh(u), du = 0.7978845608f * (1.0f + 3.0f*0.044715f*x2);
        grad_input[i] = grad_output[i] * (0.5f*(1.0f+t) + 0.5f*x*(1.0f-t*t)*du);
    }
}

__kernel void adamw_step(__global float* params, __global const float* grads,
                         __global float* m, __global float* v, const int size,
                         const float lr, const float beta1, const float beta2,
                         const float eps, const float weight_decay, const int step) {
    int i = get_global_id(0);
    if (i >= size) return;
    float g = grads[i], mn = beta1*m[i] + (1.0f-beta1)*g, vn = beta2*v[i] + (1.0f-beta2)*g*g;
    float mh = mn / (1.0f - pow(beta1, (float)step)), vh = vn / (1.0f - pow(beta2, (float)step));
    params[i] -= lr * (mh / (sqrt(vh) + eps) + weight_decay * params[i]); m[i] = mn; v[i] = vn;
}

__kernel void sgd_step(__global float* params, __global const float* grads,
                       const int size, const float lr, const float weight_decay) {
    int i = get_global_id(0);
    if (i < size) params[i] -= lr * (grads[i] + weight_decay * params[i]);
}
