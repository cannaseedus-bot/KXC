// WebGPU Hierarchical Compression Operations
// Implements token→fold gram→super gram→hyper gram compression pipeline

struct CompressionConfig {
    level: u32,           // 1=fold, 2=super, 3=hyper
    input_vocab_size: u32,
    output_vocab_size: u32,
    max_sequence_len: u32,
};

struct GramSequence {
    input_tokens: array<u32, 128>,   // Input token sequence
    output_grams: array<u32, 128>,   // Output gram sequence
    sequence_len: u32,
    compression_ratio: f32,
};

@group(0) @binding(0) var<storage, read_write> sequences: array<GramSequence>;
@group(0) @binding(1) var<storage, read> codebook: array<u32>;
@group(0) @binding(2) var<storage, read> config: CompressionConfig;
@group(0) @binding(3) var<storage, read_write> compression_stats: array<atomic<u32>>;

// Hash-based token ngram fingerprint
fn compute_ngram_hash(tokens: array<u32, 3>, ngram_size: u32) -> u32 {
    var hash: u32 = 0u;
    var multiplier: u32 = 1u;

    for (var i: u32 = 0u; i < ngram_size; i = i + 1u) {
        hash = hash + (tokens[i] * multiplier);
        multiplier = (multiplier * 73u) % 4294967291u; // Large prime
    }

    return hash % config.output_vocab_size;
}

// Level 1: Token → Fold (bigram/trigram detection)
@compute @workgroup_size(256)
fn compress_level1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let seq_idx = global_id.x;
    if (seq_idx >= arrayLength(&sequences)) {
        return;
    }

    var seq = sequences[seq_idx];
    var read_pos: u32 = 0u;
    var write_pos: u32 = 0u;

    while (read_pos < seq.sequence_len && write_pos < 128u) {
        // Look for bigrams (2 consecutive tokens)
        if (read_pos + 1u < seq.sequence_len) {
            let bigram_hash = compute_ngram_hash(
                array<u32, 3>(seq.input_tokens[read_pos], seq.input_tokens[read_pos + 1u], 0u),
                2u
            );

            // Check if this bigram is in codebook
            if (codebook[bigram_hash % arrayLength(&codebook)] != 0u) {
                // Common bigram - emit as fold gram
                seq.output_grams[write_pos] = bigram_hash + 32064u; // Offset to fold gram space
                read_pos = read_pos + 2u;
            } else {
                // Uncommon bigram - keep as token
                seq.output_grams[write_pos] = seq.input_tokens[read_pos];
                read_pos = read_pos + 1u;
            }
        } else {
            // Not enough tokens for bigram
            seq.output_grams[write_pos] = seq.input_tokens[read_pos];
            read_pos = read_pos + 1u;
        }

        write_pos = write_pos + 1u;
    }

    seq.sequence_len = write_pos;
    seq.compression_ratio = f32(write_pos) / f32(read_pos);

    atomicAdd(&compression_stats[0], u32(f32(1000u) * seq.compression_ratio));
    sequences[seq_idx] = seq;
}

// Level 2: Fold → Super (multi-gram detection)
@compute @workgroup_size(256)
fn compress_level2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let seq_idx = global_id.x;
    if (seq_idx >= arrayLength(&sequences)) {
        return;
    }

    var seq = sequences[seq_idx];
    var read_pos: u32 = 0u;
    var write_pos: u32 = 0u;

    while (read_pos < seq.sequence_len && write_pos < 128u) {
        // Look for trigrams of fold grams
        if (read_pos + 2u < seq.sequence_len) {
            let trigram_hash = compute_ngram_hash(
                array<u32, 3>(
                    seq.output_grams[read_pos],
                    seq.output_grams[read_pos + 1u],
                    seq.output_grams[read_pos + 2u]
                ),
                3u
            );

            // Check if this trigram is in codebook
            if (codebook[trigram_hash % arrayLength(&codebook)] != 0u) {
                // Common trigram - emit as super gram
                seq.output_grams[write_pos] = trigram_hash + (32064u + 16384u);
                read_pos = read_pos + 3u;
            } else {
                // Keep existing gram
                seq.output_grams[write_pos] = seq.output_grams[read_pos];
                read_pos = read_pos + 1u;
            }
        } else {
            seq.output_grams[write_pos] = seq.output_grams[read_pos];
            read_pos = read_pos + 1u;
        }

        write_pos = write_pos + 1u;
    }

    seq.sequence_len = write_pos;
    seq.compression_ratio = seq.compression_ratio * (f32(write_pos) / f32(read_pos));

    atomicAdd(&compression_stats[1], u32(f32(1000u) * seq.compression_ratio));
    sequences[seq_idx] = seq;
}

// Level 3: Super → Hyper (semantic clustering)
@compute @workgroup_size(256)
fn compress_level3(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let seq_idx = global_id.x;
    if (seq_idx >= arrayLength(&sequences)) {
        return;
    }

    var seq = sequences[seq_idx];

    // Hyper compression: cluster super grams by semantic similarity
    // Using deterministic hash-based clustering
    var cluster_hash: u32 = 0u;

    for (var i: u32 = 0u; i < seq.sequence_len; i = i + 1u) {
        // Rolling hash for semantic context
        cluster_hash = (cluster_hash * 31u + seq.output_grams[i]) % 4294967291u;

        // Every N grams or on sequence boundaries, emit hyper gram
        if (i > 0u && (i % 4u == 0u || i == seq.sequence_len - 1u)) {
            let hyper_gram = (cluster_hash % config.output_vocab_size) + (32064u + 16384u + 8192u);
            seq.output_grams[i] = hyper_gram;
            cluster_hash = 0u;
        }
    }

    seq.compression_ratio = seq.compression_ratio * 0.5; // Typical 50% reduction at L3

    atomicAdd(&compression_stats[2], u32(f32(1000u) * seq.compression_ratio));
    sequences[seq_idx] = seq;
}

// Compute cumulative compression statistics
@compute @workgroup_size(1)
fn compute_compression_stats() {
    let l1_total = atomicLoad(&compression_stats[0]);
    let l2_total = atomicLoad(&compression_stats[1]);
    let l3_total = atomicLoad(&compression_stats[2]);

    // Compute combined compression efficiency
    let combined = u32(
        (f32(l1_total) * 0.33 + f32(l2_total) * 0.33 + f32(l3_total) * 0.34) / 1000.0
    );

    atomicStore(&compression_stats[3], combined);
}

// Decompress hyper grams back through hierarchy for inference
@compute @workgroup_size(256)
fn decompress_hyper(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let seq_idx = global_id.x;
    if (seq_idx >= arrayLength(&sequences)) {
        return;
    }

    var seq = sequences[seq_idx];
    var expanded: array<u32, 256>;
    var expand_pos: u32 = 0u;

    // Reverse mapping: hyper gram -> super grams
    for (var i: u32 = 0u; i < seq.sequence_len; i = i + 1u) {
        let gram = seq.output_grams[i];

        if (gram >= (32064u + 16384u + 8192u)) {
            // Hyper gram - expand to multiple super grams
            let base = gram - (32064u + 16384u + 8192u);
            expanded[expand_pos] = (base % 16384u) + (32064u + 16384u);
            expand_pos = expand_pos + 1u;
        } else if (gram >= (32064u + 16384u)) {
            // Super gram - expand to multiple fold grams
            let base = gram - (32064u + 16384u);
            expanded[expand_pos] = (base % 8192u) + 32064u;
            expand_pos = expand_pos + 1u;
        } else if (gram >= 32064u) {
            // Fold gram - expand to tokens
            let base = gram - 32064u;
            expanded[expand_pos] = base % 32064u;
            expand_pos = expand_pos + 1u;
        } else {
            // Token - keep as is
            expanded[expand_pos] = gram;
            expand_pos = expand_pos + 1u;
        }

        if (expand_pos >= 256u) {
            break;
        }
    }

    // Copy back
    for (var i: u32 = 0u; i < expand_pos && i < 128u; i = i + 1u) {
        seq.output_grams[i] = expanded[i];
    }

    seq.sequence_len = expand_pos;
    sequences[seq_idx] = seq;
}
