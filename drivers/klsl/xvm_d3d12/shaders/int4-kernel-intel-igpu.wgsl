// Deterministic INT4 Kernel for Intel Integrated GPU
// Optimized for Intel iGPU (xe-hpg architecture)
// Implements efficient 4-bit quantized matrix multiply

struct QuantizationMetadata {
    scale: f32,           // Dequantization scale
    zero_point: i32,      // Zero offset
    min_value: f32,       // Stored for invertibility
    max_value: f32,       // Stored for invertibility
}

struct Int4Config {
    hidden_dim: u32,      // Input dimension (3072)
    output_dim: u32,      // Output dimension (8192)
    batch_size: u32,      // Batch size
    expert_id: u32,       // Expert identifier
    tile_size: u32,       // Tiling for cache efficiency (32 typical)
}

// ============================================================================
// INT4 Unpacking Utilities
// ============================================================================

// Unpack two 4-bit values from single byte
fn unpack_int4(byte: u32) -> vec2<i32> {
    let low_nibble = i32(byte & 0xFu);
    let high_nibble = i32((byte >> 4u) & 0xFu);

    // Convert from unsigned 4-bit to signed range [-8, 7]
    let low_signed = low_nibble - 8;
    let high_signed = high_nibble - 8;

    return vec2<i32>(low_signed, high_signed);
}

// Dequantize 4-bit integer to float
fn dequantize_int4(
    int4_val: i32,
    meta: QuantizationMetadata
) -> f32 {
    // int4_val is in range [-8, 7]
    let dequant = f32(int4_val - meta.zero_point) * meta.scale + meta.min_value;
    return dequant;
}

// Hash-based deterministic bit selection for reproducibility
fn deterministic_hash(token_idx: u32, expert_id: u32) -> u32 {
    var h: u32 = 0x9e3779b9u;  // Golden ratio
    h = h ^ ((token_idx << 5u) | (token_idx >> 27u));
    h = h ^ ((expert_id << 7u) | (expert_id >> 25u));
    h = (h ^ (h >> 16u)) * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    return h;
}

// ============================================================================
// Main INT4 Expert Forward Kernel
// Optimized for Intel iGPU with shared memory tiling
// ============================================================================

@group(0) @binding(0)
var<storage, read> input_data: array<f32>;

@group(0) @binding(1)
var<uniform> config: Int4Config;

@group(1) @binding(0)
var<storage, read> weights_packed: array<u32>;  // INT4 weights packed as nibbles

@group(1) @binding(1)
var<storage, read> quant_metadata: array<QuantizationMetadata>;

@group(2) @binding(0)
var<storage, read_write> output_data: array<f32>;

@group(2) @binding(1)
var<storage, read_write> trace_log: array<u32>;

var<workgroup> shared_input: array<f32, 256>;    // Local cache
var<workgroup> shared_sum: array<f32, 32>;       // Reduction buffer

@compute @workgroup_size(128, 2, 1)
fn int4_expert_forward(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let batch_idx = workgroup_id.x;
    let output_idx = workgroup_id.y;
    let local_idx = local_id.x;
    let lane_idx = local_id.y;  // Used for reduction within group

    // Bounds check
    if (batch_idx >= config.batch_size || output_idx >= config.output_dim) {
        return;
    }

    // Deterministic seed for this computation
    let seed = deterministic_hash(batch_idx, config.expert_id);

    // ========================================================================
    // Phase 1: Load input chunk into shared memory
    // ========================================================================

    let input_offset = batch_idx * config.hidden_dim;
    let chunk_size = config.hidden_dim;

    // Cooperatively load input into shared memory (2 threads load 256 elements)
    let elements_per_thread = (chunk_size + 127u) / 128u;

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let elem_idx = local_idx + i * 128u;
        if (elem_idx < chunk_size) {
            shared_input[elem_idx] = input_data[input_offset + elem_idx];
        }
    }

    workgroupBarrier();  // Ensure all threads have loaded data

    // ========================================================================
    // Phase 2: Compute output element using INT4 weights
    // ========================================================================

    // Each thread handles one output element row-wise
    // Matrix multiplication: (batch_size, hidden_dim) @ (hidden_dim, output_dim)

    var accumulator: f32 = 0.0;

    // Process weights in chunks for cache efficiency
    for (var chunk: u32 = 0u; chunk < config.tile_size; chunk = chunk + 1u) {
        let chunk_start = chunk * 128u;
        let chunk_end = min(chunk_start + 128u, config.hidden_dim);

        // Compute dot product for this chunk
        for (var h: u32 = chunk_start + local_idx; h < chunk_end; h = h + 128u) {
            // Load input value (cached in shared memory)
            let input_val = shared_input[h];

            // Compute weight linear index: output_idx * hidden_dim + h
            let weight_linear_idx = output_idx * config.hidden_dim + h;

            // Unpack INT4: 2 weights per byte
            let byte_idx = weight_linear_idx / 2u;
            let nibble_select = weight_linear_idx % 2u;

            let packed_byte = weights_packed[byte_idx];
            let int4_vals = unpack_int4(packed_byte);
            let int4_weight = int4_vals[nibble_select];

            // Dequantize using metadata
            let meta = quant_metadata[output_idx];
            let weight_val = dequantize_int4(int4_weight, meta);

            // Accumulate
            accumulator = accumulator + input_val * weight_val;
        }
    }

    // ========================================================================
    // Phase 3: Reduction across threads
    // ========================================================================

    // Use shared memory for parallel reduction
    shared_sum[local_idx] = accumulator;
    workgroupBarrier();

    // Tree reduction (local_idx must be < 128)
    if (local_idx < 64u) {
        shared_sum[local_idx] = shared_sum[local_idx] + shared_sum[local_idx + 64u];
    }
    workgroupBarrier();

    if (local_idx < 32u) {
        shared_sum[local_idx] = shared_sum[local_idx] + shared_sum[local_idx + 32u];
    }
    workgroupBarrier();

    if (local_idx < 16u) {
        shared_sum[local_idx] = shared_sum[local_idx] + shared_sum[local_idx + 16u];
    }
    workgroupBarrier();

    if (local_idx < 8u) {
        shared_sum[local_idx] = shared_sum[local_idx] + shared_sum[local_idx + 8u];
    }
    workgroupBarrier();

    if (local_idx < 4u) {
        shared_sum[local_idx] = shared_sum[local_idx] + shared_sum[local_idx + 4u];
    }
    workgroupBarrier();

    if (local_idx < 2u) {
        shared_sum[local_idx] = shared_sum[local_idx] + shared_sum[local_idx + 2u];
    }
    workgroupBarrier();

    if (local_idx == 0u) {
        shared_sum[0] = shared_sum[0] + shared_sum[1];
    }
    workgroupBarrier();

    // ========================================================================
    // Phase 4: Write result and trace
    // ========================================================================

    if (local_idx == 0u) {
        let output_offset = batch_idx * config.output_dim + output_idx;
        output_data[output_offset] = shared_sum[0];

        // Log for deterministic replay verification
        let trace_idx = batch_idx * config.output_dim + output_idx;
        if (trace_idx < arrayLength(&trace_log)) {
            // Store hash of computation for verification
            var trace_hash: u32 = seed;
            trace_hash = trace_hash ^ bitcast<u32>(shared_sum[0]);
            trace_log[trace_idx] = trace_hash;
        }
    }

    workgroupBarrier();
}

// ============================================================================
// Optimized INT4 Batch Processing Kernel
// Processes multiple tokens in parallel
// ============================================================================

@compute @workgroup_size(256)
fn int4_batch_forward(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let global_idx = global_id.x;
    let total_elements = config.batch_size * config.output_dim;

    if (global_idx >= total_elements) {
        return;
    }

    let batch_idx = global_idx / config.output_dim;
    let output_idx = global_idx % config.output_dim;

    // Deterministic seed
    let seed = deterministic_hash(batch_idx, config.expert_id);

    // ================================================================
    // Inline matrix multiply without shared memory (fallback)
    // ================================================================

    var accumulator: f32 = 0.0;
    let input_offset = batch_idx * config.hidden_dim;

    for (var h: u32 = 0u; h < config.hidden_dim; h = h + 1u) {
        let input_val = input_data[input_offset + h];

        // Unpack INT4 weight
        let weight_linear_idx = output_idx * config.hidden_dim + h;
        let byte_idx = weight_linear_idx / 2u;
        let nibble_select = weight_linear_idx % 2u;

        let packed_byte = weights_packed[byte_idx];
        let int4_vals = unpack_int4(packed_byte);
        let int4_weight = int4_vals[nibble_select];

        // Dequantize
        let meta = quant_metadata[output_idx];
        let weight_val = dequantize_int4(int4_weight, meta);

        accumulator = accumulator + input_val * weight_val;
    }

    // Write output
    let output_offset = batch_idx * config.output_dim + output_idx;
    output_data[output_offset] = accumulator;

    // Deterministic trace
    let trace_idx = global_idx;
    if (trace_idx < arrayLength(&trace_log)) {
        var trace_hash: u32 = seed;
        trace_hash = trace_hash ^ bitcast<u32>(accumulator);
        trace_log[trace_idx] = trace_hash;
    }
}

// ============================================================================
// INT4 Quantization Utility Kernel
// Used to pack float weights into INT4 during setup
// ============================================================================

@group(0) @binding(0)
var<storage, read> float_weights: array<f32>;

@group(1) @binding(0)
var<storage, read_write> int4_weights: array<u32>;

@group(2) @binding(0)
var<storage, read_write> metadata: array<QuantizationMetadata>;

@compute @workgroup_size(256)
fn quantize_to_int4(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    if (idx >= arrayLength(&float_weights)) {
        return;
    }

    // Find min/max for this shard
    let shard_size = 8192u;
    let shard_idx = idx / shard_size;
    let shard_start = shard_idx * shard_size;
    let shard_end = min(shard_start + shard_size, arrayLength(&float_weights));

    var min_val: f32 = float_weights[shard_start];
    var max_val: f32 = float_weights[shard_start];

    for (var i: u32 = shard_start; i < shard_end; i = i + 1u) {
        min_val = min(min_val, float_weights[i]);
        max_val = max(max_val, float_weights[i]);
    }

    // Compute scale and zero point
    let scale = (max_val - min_val) / 15.0;  // 15 = 2^4 - 1
    let zero_point = i32((0.0 - min_val) / scale + 0.5);

    // Quantize
    let val = float_weights[idx];
    let normalized = (val - min_val) / (max_val - min_val + 1e-8);
    let quantized = i32(normalized * 15.0 + 0.5);

    // Pack into nibble
    let byte_idx = idx / 2u;
    let nibble_select = idx % 2u;
    let int4_val = u32(max(0, min(15, quantized)) & 0xFu);

    // Atomic write with packing
    if (nibble_select == 0u) {
        int4_weights[byte_idx] = (int4_weights[byte_idx] & 0xFFFFFFF0u) | int4_val;
    } else {
        int4_weights[byte_idx] = (int4_weights[byte_idx] & 0xFFFFFF0Fu) | (int4_val << 4u);
    }

    // Store metadata for this shard
    metadata[shard_idx] = QuantizationMetadata(scale, zero_point, min_val, max_val);
}

// ============================================================================
// Performance Monitoring Kernel
// Collects statistics about INT4 operations
// ============================================================================

struct PerformanceStats {
    total_operations: u32,
    successful_ops: u32,
    failed_ops: u32,
    avg_latency_ms: f32,
}

@group(0) @binding(0)
var<storage, read_write> perf_stats: PerformanceStats;

@compute @workgroup_size(1)
fn collect_performance_stats() {
    // Placeholder for performance collection
    // In real implementation, would aggregate across kernels
    atomicAdd(&perf_stats.total_operations, 1u);
}
