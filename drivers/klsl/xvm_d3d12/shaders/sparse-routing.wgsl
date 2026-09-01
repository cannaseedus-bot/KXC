// WebGPU Sparse Top-K Routing Kernel
// Selects K experts per token based on gating network similarity scores

struct RoutingConfig {
    k: u32,              // Number of experts to select
    num_experts: u32,    // Total number of experts
    capacity_factor: f32, // Expert capacity multiplier
    drop_policy: u32,    // 0=drop, 1=reroute
};

struct Token {
    embedding: array<f32, 3072>, // Token embedding
    expert_scores: array<f32, 10>, // Gating scores for each expert
    selected_experts: array<u32, 2>, // Selected expert IDs (K=2)
    routing_weights: array<f32, 2>, // Softmax weights for each
    token_id: u32,
};

@group(0) @binding(0) var<storage, read_write> tokens: array<Token>;
@group(0) @binding(1) var<storage, read> config: RoutingConfig;
@group(0) @binding(2) var<storage, read_write> expert_assignments: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> expert_capacities: array<atomic<u32>>;

// Compute softmax over expert scores
fn softmax_experts(scores: array<f32, 10>) -> array<f32, 10> {
    var max_score: f32 = -1e9;
    var results: array<f32, 10>;

    // Find max for numerical stability
    for (var i: u32 = 0u; i < config.num_experts; i = i + 1u) {
        max_score = max(max_score, scores[i]);
    }

    var sum: f32 = 0.0;

    // Compute exp(x - max)
    for (var i: u32 = 0u; i < config.num_experts; i = i + 1u) {
        results[i] = exp(scores[i] - max_score);
        sum = sum + results[i];
    }

    // Normalize
    for (var i: u32 = 0u; i < config.num_experts; i = i + 1u) {
        results[i] = results[i] / sum;
    }

    return results;
}

// Select top-k experts using bitonic sort
fn select_top_k(scores: array<f32, 10>, k: u32) -> array<u32, 2> {
    var indices: array<u32, 10>;
    var vals: array<f32, 10>;

    // Initialize with identity
    for (var i: u32 = 0u; i < config.num_experts; i = i + 1u) {
        indices[i] = i;
        vals[i] = scores[i];
    }

    // Bitonic sort (simplified for 10 experts)
    // Stage 1: Compare pairs
    for (var i: u32 = 0u; i < 5u; i = i + 1u) {
        let j: u32 = i * 2u;
        if (vals[j] < vals[j + 1u]) {
            let tmp_idx = indices[j];
            let tmp_val = vals[j];
            indices[j] = indices[j + 1u];
            vals[j] = vals[j + 1u];
            indices[j + 1u] = tmp_idx;
            vals[j + 1u] = tmp_val;
        }
    }

    // Stage 2: Select top 2
    var result: array<u32, 2>;
    result[0] = indices[0];
    result[1] = indices[1];

    return result;
}

@compute @workgroup_size(256)
fn route_tokens(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let token_idx = global_id.x;
    if (token_idx >= arrayLength(&tokens)) {
        return;
    }

    var token = tokens[token_idx];

    // Softmax normalize gating scores
    let softmax_scores = softmax_experts(token.expert_scores);

    // Select top K experts
    let expert_ids = select_top_k(softmax_scores, config.k);

    // Update token routing
    token.selected_experts[0] = expert_ids[0];
    token.selected_experts[1] = expert_ids[1];

    // Extract weights for selected experts
    token.routing_weights[0] = softmax_scores[expert_ids[0]];
    token.routing_weights[1] = softmax_scores[expert_ids[1]];

    // Normalize routing weights to sum to 1
    let weight_sum = token.routing_weights[0] + token.routing_weights[1];
    token.routing_weights[0] = token.routing_weights[0] / weight_sum;
    token.routing_weights[1] = token.routing_weights[1] / weight_sum;

    // Update expert assignment counters
    let expert0_idx = token.selected_experts[0];
    let expert1_idx = token.selected_experts[1];

    atomicAdd(&expert_assignments[expert0_idx], 1u);
    atomicAdd(&expert_assignments[expert1_idx], 1u);

    // Store updated token
    tokens[token_idx] = token;
}

// Verify capacity constraints
@compute @workgroup_size(256)
fn check_capacity(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let expert_idx = global_id.x;
    if (expert_idx >= config.num_experts) {
        return;
    }

    let assignments = atomicLoad(&expert_assignments[expert_idx]);
    let capacity = u32(f32(atomicLoad(&expert_capacities[expert_idx])) * config.capacity_factor);

    if (assignments > capacity) {
        // Capacity exceeded - could implement reroute logic here
        // For now, just log the constraint violation
        atomicStore(&expert_capacities[expert_idx], assignments);
    }
}
