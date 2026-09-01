// WebGPU Expert Forward Pass Kernel
// Implements feedforward computation for a single expert

struct ExpertConfig {
    hidden_dim: u32,      // 3072 for Phi-3
    ffn_dim: u32,         // 8192 for Phi-3
    expert_id: u32,
    dtype: u32,           // 0=float32, 1=float16, 2=int4
};

struct ExpertWeights {
    w_gate: array<f32, 3072 * 8192>,   // Token→FFN projection
    w_up: array<f32, 3072 * 8192>,     // Token→FFN gating
    w_down: array<f32, 8192 * 3072>,   // FFN→Token projection
};

struct TokenState {
    id: u32,
    embedding: array<f32, 3072>,
    output: array<f32, 3072>,
};

@group(0) @binding(0) var<storage, read_write> tokens: array<TokenState>;
@group(0) @binding(1) var<storage, read> weights: ExpertWeights;
@group(0) @binding(2) var<storage, read> config: ExpertConfig;
@group(0) @binding(3) var<storage, read_write> activation_log: array<atomic<u32>>;

// Swish activation: x * sigmoid(x)
fn swish(x: f32) -> f32 {
    return x * (1.0 / (1.0 + exp(-x)));
}

// RMS Normalization
fn rms_norm(x: array<f32, 3072>) -> array<f32, 3072> {
    var ss: f32 = 0.0;
    for (var i: u32 = 0u; i < 3072u; i = i + 1u) {
        ss = ss + x[i] * x[i];
    }
    let rms = sqrt(ss / 3072.0 + 1e-6);

    var result: array<f32, 3072>;
    for (var i: u32 = 0u; i < 3072u; i = i + 1u) {
        result[i] = x[i] / rms;
    }
    return result;
}

// Matrix multiplication: (3072) x (3072x8192) -> (8192)
fn matmul_expand(input: array<f32, 3072>, weights: array<f32, 3072 * 8192>) -> array<f32, 8192> {
    var result: array<f32, 8192>;

    for (var j: u32 = 0u; j < 8192u; j = j + 1u) {
        var acc: f32 = 0.0;
        for (var i: u32 = 0u; i < 3072u; i = i + 1u) {
            let idx = i * 8192u + j;
            acc = acc + input[i] * weights[idx];
        }
        result[j] = acc;
    }

    return result;
}

// Matrix multiplication: (8192) x (8192x3072) -> (3072)
fn matmul_project(input: array<f32, 8192>, weights: array<f32, 8192 * 3072>) -> array<f32, 3072> {
    var result: array<f32, 3072>;

    for (var j: u32 = 0u; j < 3072u; j = j + 1u) {
        var acc: f32 = 0.0;
        for (var i: u32 = 0u; i < 8192u; i = i + 1u) {
            let idx = i * 3072u + j;
            acc = acc + input[i] * weights[idx];
        }
        result[j] = acc;
    }

    return result;
}

// Expert forward pass: input -> norm -> gate up -> swish -> down -> output
@compute @workgroup_size(256)
fn forward_pass(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let token_idx = global_id.x;
    if (token_idx >= arrayLength(&tokens)) {
        return;
    }

    var token = tokens[token_idx];

    // 1. RMS Normalization
    let x_norm = rms_norm(token.embedding);

    // 2. Gate up projection: (3072) -> (8192)
    let hidden_gate = matmul_expand(x_norm, config.w_gate);

    // 3. Up projection: (3072) -> (8192)
    let hidden_up = matmul_expand(x_norm, config.w_up);

    // 4. Gating: gate * swish(up)
    var gated: array<f32, 8192>;
    for (var i: u32 = 0u; i < 8192u; i = i + 1u) {
        gated[i] = hidden_gate[i] * swish(hidden_up[i]);
    }

    // 5. Down projection: (8192) -> (3072)
    let output_proj = matmul_project(gated, config.w_down);

    // 6. Residual connection
    for (var i: u32 = 0u; i < 3072u; i = i + 1u) {
        token.output[i] = token.embedding[i] + output_proj[i];
    }

    // Log activation for analytics
    let activation_sum = u32(0);
    for (var i: u32 = 0u; i < 8192u; i = i + 1u) {
        if (gated[i] > 0.1) {
            atomicAdd(&activation_log[0], 1u);
        }
    }

    // Store result
    tokens[token_idx] = token;
}

// Compute expert utilization statistics
@compute @workgroup_size(1)
fn compute_utilization() {
    let total_activations = atomicLoad(&activation_log[0]);
    let capacity = 8192u; // FFN dimension
    let utilization = f32(total_activations) / f32(capacity);

    // Store utilization metric (simplified)
    atomicStore(&activation_log[1], u32(utilization * 1000.0));
}
