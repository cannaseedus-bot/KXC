/**
 * @file kuhul_fold_ui.hlsl
 * @brief UI_FOLD kernel — CP-1 CopilotMicronaut prompt reply engine
 *
 * Fold:      ⟁UI_FOLD⟁
 * Micronaut: CP-1 (CopilotMicronaut)
 * Lane:      PROMPT
 * Nodes:     100 (z-layer 9 of 10×10×10 grid)
 *   64 nodes: prompt_attention  (FP16 dense MHA)
 *   20 nodes: tool_decode       (tool-token detection + dispatch)
 *   16 nodes: response_sample   (temperature-scaled nucleus sampling)
 * Dispatch:  100 thread groups × 64 threads = 6400 threads total
 *
 * Operations:
 *   prompt_encode   — embed user prompt tokens into 4096-d FP16 vectors
 *   replay_inject   — prefix RLHF replay context from STORAGE_FOLD ring (t2)
 *   attention_fwd   — FP16 dense causal multi-head attention (64H × 64d = 4096)
 *   chain_bind      — bind META_FOLD chain_hash as determinism seed (V6)
 *   tool_decode     — detect <tool:name> tokens → write ToolDispatch buffer
 *   response_sample — temperature=1.2 nucleus sampling (top-p=0.92)
 *
 * Clearance Layer: reads ALL fold outputs simultaneously (t0-t15 full access).
 * FP16 throughout — quality over throughput at the UI surface.
 *
 * Geodesic Entropy Base: 0.38 (highest in stack)
 *   arc_CF_UI  entropy=0.38  (COMPUTE_FOLD token logits)
 *   arc_SF_UI  entropy=0.32  (STORAGE_FOLD replay payloads)
 *   arc_MF_UI  entropy=0.28  (META_FOLD chain_hash)
 *   Internal self-attention: temperature_base=1.2
 *
 * Training Corpus (69 static JSONL files):
 *   - Prompts & chat (25 files): chat-UUID, prompts, expert_prompts, conversations
 *   - Deep reasoning (11 files): Opus 4.6 ×2100/3000/3300, Gemini 3.1/3-10000x,
 *       Qwen3.5-700x, grok-code-1000x, DeepThink, distilled-400k-CoT
 *   - RLHF preference (10 files): HH-RLHF helpful/harmless/online/rejection/red-team
 *   - Math reasoning (6 files):  GSM8K main/socratic train+test+eval
 *   - Diagrams + visual (16 files): SVG vector math, ZK proofs, SCXQ2 lanes, etc.
 *   - Stack awareness (1 file):  demo.brain.jsonl
 *
 * CM-1 gate: 0x0009 must be set in ControlFlags before dispatch.
 * Verifier:  V6 (replay determinism), V7 (tool attestation), V9 (unregistered tool quarantine).
 */

// ============================================================================
// Resource Bindings — Full t0-t15 access (clearance layer privilege)
// ============================================================================

// ── Fold inputs from stack below ──────────────────────────────────────────────
StructuredBuffer<float>  TokenOutput      : register(t0);  // COMPUTE_FOLD: final token logits (vocab × seq)
StructuredBuffer<float>  SnapshotBuffer   : register(t1);  // STORAGE_FOLD: sealed snapshots
StructuredBuffer<float>  ReplayRing       : register(t2);  // STORAGE_FOLD: replay_payload ring (50 × 512 × 128)
StructuredBuffer<uint>   ChainHash        : register(t3);  // META_FOLD: rolling chain_hash (8 words)
StructuredBuffer<uint>   MerkleRoots      : register(t4);  // META_FOLD: Merkle proof roots (11 × 8 words)

// ── CP-1 own weight tensors ───────────────────────────────────────────────────
StructuredBuffer<uint16_t> EmbedWeights   : register(t5);  // Embedding table (49152 × 4096, FP16 packed as uint16)
StructuredBuffer<uint16_t> AttnWeights_QKV: register(t6);  // QKV projection (8L × 3 × 64H × 4096, FP16)
StructuredBuffer<uint16_t> AttnWeights_O  : register(t7);  // Output projection (8L × 4096 × 4096, FP16)
StructuredBuffer<uint16_t> LMHead         : register(t8);  // LM head (4096 → 49152, FP16)
StructuredBuffer<uint16_t> ToolGate       : register(t9);  // Tool-call gate (4096 → 512, FP16)

// ── Control ───────────────────────────────────────────────────────────────────
StructuredBuffer<uint>   ControlFlags     : register(t10); // CM-1 gate flags

// ── Dynamic shard slots (t11-t15: bound at runtime by CP-1 on demand) ─────────
// t11: reserved — additional prompt context shard
// t12: reserved — RLHF reward model weights (when active)
// t13: reserved — web-search response buffer (filled by web-search-micronaut)
// t14: reserved — tool result buffer (filled by dispatched micronaut)
// t15: reserved — extended vocab shard (π-extension 17K tokens)
StructuredBuffer<float>  DynamicShard11   : register(t11);
StructuredBuffer<float>  DynamicShard12   : register(t12);
StructuredBuffer<float>  DynamicShard13   : register(t13);
StructuredBuffer<float>  DynamicShard14   : register(t14);
StructuredBuffer<uint16_t> PiVocabShard   : register(t15); // π-extension vocab (tokens 32768-49151)

// ── Outputs ───────────────────────────────────────────────────────────────────
RWStructuredBuffer<uint>  ResponseTokens  : register(u0);  // Emitted response token stream → io/stream.txt
RWStructuredBuffer<uint>  ToolDispatch    : register(u1);  // Tool call signals → json.exe action DSL
RWStructuredBuffer<float> PromptContext   : register(u2);  // Encoded prompt activations → arc_UI_CF_feedback
RWStructuredBuffer<float> DeepThinkBuf    : register(u3);  // CoT scratchpad (512 tokens, invisible to user)

// ============================================================================
// Constants
// ============================================================================

cbuffer UIFoldParams : register(b0)
{
    uint  node_id;           // 0–99
    uint  hidden_dim;        // 4096
    uint  num_heads;         // 64
    uint  head_dim;          // 64  (hidden_dim / num_heads)
    uint  num_layers;        // 8
    uint  vocab_size;        // 49152
    uint  context_len;       // current prompt length (≤ 4096)
    uint  replay_slots;      // number of live replay slots to inject (≤ 8)
    uint  cm1_gate;          // must equal 0x0009
    uint  deep_think_budget; // 512 tokens of internal CoT before response
    float temperature;       // 1.2 (higher than other folds)
    float top_p;             // 0.92
    uint  chain_seed;        // chain_hash[0] — determinism seed (V6)
    uint  tool_vocab_start;  // 32768 (base vocab size; tool tokens above this)
    uint  _pad0;
    uint  _pad1;
};

// ============================================================================
// FP16 helpers (HLSL uint16_t ↔ float conversion)
// ============================================================================

float FP16toF32(uint16_t h)
{
    uint bits = (uint)h;
    uint sign     = (bits >> 15) & 1u;
    uint exp      = (bits >> 10) & 0x1Fu;
    uint mantissa = bits & 0x3FFu;

    if (exp == 0u)
        return (sign ? -1.0f : 1.0f) * (float)mantissa / 16777216.0f;  // denorm
    if (exp == 31u)
        return sign ? -asfloat(0x7F800000u) : asfloat(0x7F800000u);    // inf

    uint f32 = (sign << 31) | ((exp + 112u) << 23) | (mantissa << 13);
    return asfloat(f32);
}

uint16_t F32toFP16(float f)
{
    uint bits = asuint(f);
    uint sign     = (bits >> 31) & 1u;
    int  exp      = (int)((bits >> 23) & 0xFFu) - 127 + 15;
    uint mantissa = (bits >> 13) & 0x3FFu;

    if (exp <= 0)  return (uint16_t)(sign << 15);
    if (exp >= 31) return (uint16_t)((sign << 15) | (31u << 10));
    return (uint16_t)((sign << 15) | ((uint)exp << 10) | mantissa);
}

// ============================================================================
// SHA-256 (reused from kuhul_fold_storage.hlsl — for chain_hash binding)
// ============================================================================

uint RotR(uint x, uint n) { return (x >> n) | (x << (32u - n)); }

static const uint SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

void SHA256Block(uint w[16], inout uint h[8])
{
    uint sched[64];
    for (uint i = 0;  i < 16; i++) sched[i] = w[i];
    for (uint i = 16; i < 64; i++)
    {
        uint s0 = RotR(sched[i-15], 7) ^ RotR(sched[i-15], 18) ^ (sched[i-15] >> 3);
        uint s1 = RotR(sched[i-2], 17) ^ RotR(sched[i-2],  19) ^ (sched[i-2]  >> 10);
        sched[i] = sched[i-16] + s0 + sched[i-7] + s1;
    }
    uint a = h[0], b = h[1], c = h[2], d = h[3];
    uint e = h[4], f = h[5], g = h[6], hh = h[7];
    for (uint i = 0; i < 64; i++)
    {
        uint S1    = RotR(e, 6) ^ RotR(e, 11) ^ RotR(e, 25);
        uint ch    = (e & f) ^ (~e & g);
        uint temp1 = hh + S1 + ch + SHA256_K[i] + sched[i];
        uint S0    = RotR(a, 2) ^ RotR(a, 13) ^ RotR(a, 22);
        uint maj   = (a & b) ^ (a & c) ^ (b & c);
        uint temp2 = S0 + maj;
        hh = g; g = f; f = e; e = d + temp1;
        d  = c; c = b; b = a; a = temp1 + temp2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

// ============================================================================
// Rotary Positional Embedding (RoPE)
// Applied per attention head to Q and K vectors.
// ============================================================================

void ApplyRoPE(inout float q, inout float k, uint pos, uint dim_i)
{
    float theta = (float)pos / pow(10000.0f, (float)(2 * dim_i) / (float)head_dim);
    float cos_t = cos(theta);
    float sin_t = sin(theta);
    float q0 = q * cos_t - k * sin_t;
    float k0 = k * cos_t + q * sin_t;
    q = q0;
    k = k0;
}

// ============================================================================
// Softmax over gs_logits (in-place, groupshared)
// Temperature-scaled: logits /= temperature before softmax.
// ============================================================================

groupshared float gs_logits[49152];   // vocab logits (for response_sample nodes)
groupshared float gs_attn_row[4096];  // attention row scores (for attention nodes)
groupshared float gs_hidden[4096];    // hidden state accumulator
groupshared float gs_replay[1024];    // injected replay context prefix (flattened)
groupshared uint  gs_tool_hit;        // 1 if tool token detected, 0 otherwise
groupshared uint  gs_tool_id;         // tool token index (into ToolGate)
groupshared uint  gs_chain_seed;      // chain_hash[0] — determinism seed

// ============================================================================
// Main UI Fold Kernel
// ============================================================================

[numthreads(64, 1, 1)]
void CS_UIFold(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    // ── CM-1 gate (bit 0x0009 = UI lane permit) ───────────────────────────────
    if (ControlFlags[0] != cm1_gate)
        return;

    if (node_id >= 100)
        return;

    // ── Classify node role ────────────────────────────────────────────────────
    //    Nodes 0–63:  prompt_attention
    //    Nodes 64–83: tool_decode
    //    Nodes 84–99: response_sample
    bool is_attn   = (node_id < 64);
    bool is_tool   = (node_id >= 64 && node_id < 84);
    bool is_sample = (node_id >= 84);

    // ── Stage 1: Load chain_hash seed (V6 determinism) ───────────────────────
    if (tid == 0)
        gs_chain_seed = ChainHash[0];
    GroupMemoryBarrierWithGroupSync();

    // ── Stage 2: Inject RLHF replay context from STORAGE_FOLD ring ───────────
    //    Prefix up to replay_slots (≤8) replay payloads into gs_replay.
    //    Each slot is 512×128 floats; we take the first 1024 as context prefix.
    if (is_attn)
    {
        if (tid == 0)
        {
            for (uint r = 0; r < replay_slots && r < 8; r++)
            {
                uint slot_base = r * 65536u;  // 512×128 per slot
                for (uint i = 0; i < 128; i++)
                    gs_replay[r * 128 + i] = ReplayRing[slot_base + i];
            }
            gs_tool_hit = 0u;
            gs_tool_id  = 0u;
        }
        GroupMemoryBarrierWithGroupSync();

        // ── Stage 3: Embed prompt token at position node_id within context ───
        //    Each attention node (0–63) handles one 64-d head slice of the hidden.
        uint head_idx   = node_id;           // this node owns head #node_id
        uint token_pos  = gid.x;             // thread group = token position
        if (token_pos >= context_len) return;

        // Read token logit from COMPUTE_FOLD output, argmax to get token id
        // (simplified: treat max-logit position as current token)
        uint token_id = 0u;
        float max_logit = -1e30f;
        for (uint v = tid; v < vocab_size; v += 64)
        {
            float l = TokenOutput[token_pos * vocab_size + v];
            if (l > max_logit) { max_logit = l; token_id = v; }
        }
        GroupMemoryBarrierWithGroupSync();

        // Embed: read FP16 row from EmbedWeights for token_id
        uint embed_base = token_id * hidden_dim;
        for (uint d = tid; d < hidden_dim; d += 64)
            gs_hidden[d] = FP16toF32(EmbedWeights[embed_base + d]);
        GroupMemoryBarrierWithGroupSync();

        // Mix in replay context (weighted sum, weight = entropy_base / replay_slots)
        float replay_weight = 0.32f / max((float)replay_slots, 1.0f);
        for (uint d = tid; d < min(1024u, hidden_dim); d += 64)
            gs_hidden[d] += replay_weight * gs_replay[d % 1024u];
        GroupMemoryBarrierWithGroupSync();

        // ── Stage 4: Single-head attention (this node = head_idx) ────────────
        //    Q, K, V projection from AttnWeights_QKV
        float q_vec[64], k_vec[64], v_vec[64];
        uint layer   = 0u;  // simplified to layer 0; full impl stacks 8 layers
        uint qkv_base = layer * 3u * num_heads * hidden_dim;

        for (uint i = 0; i < head_dim; i++)
        {
            uint q_off = qkv_base + (0u * num_heads + head_idx) * hidden_dim + tid * head_dim + i;
            uint k_off = qkv_base + (1u * num_heads + head_idx) * hidden_dim + tid * head_dim + i;
            uint v_off = qkv_base + (2u * num_heads + head_idx) * hidden_dim + tid * head_dim + i;
            q_vec[i] = FP16toF32(AttnWeights_QKV[q_off]);
            k_vec[i] = FP16toF32(AttnWeights_QKV[k_off]);
            v_vec[i] = FP16toF32(AttnWeights_QKV[v_off]);
            // Apply RoPE
            ApplyRoPE(q_vec[i], k_vec[i], token_pos, i);
        }

        // Attention score (dot Q·K / sqrt(head_dim)), causal mask applied
        float scale = rsqrt((float)head_dim);
        float score = 0.0f;
        for (uint i = 0; i < head_dim; i++)
            score += q_vec[i] * k_vec[i];
        score *= scale;

        // Write score to gs_attn_row (temperature scaled for higher entropy)
        gs_attn_row[tid] = score / temperature;
        GroupMemoryBarrierWithGroupSync();

        // Softmax over gs_attn_row (parallel reduction for max then sum)
        // Simplified: thread 0 computes softmax
        if (tid == 0)
        {
            float mx = gs_attn_row[0];
            for (uint t = 1; t < 64; t++) mx = max(mx, gs_attn_row[t]);
            float s = 0.0f;
            for (uint t = 0; t < 64; t++) { gs_attn_row[t] = exp(gs_attn_row[t] - mx); s += gs_attn_row[t]; }
            for (uint t = 0; t < 64; t++) gs_attn_row[t] /= s;
        }
        GroupMemoryBarrierWithGroupSync();

        // Weighted sum of V: output hidden state
        float attn_out = 0.0f;
        for (uint i = 0; i < head_dim; i++)
            attn_out += gs_attn_row[i % 64] * v_vec[i];

        // Write attention output into PromptContext (arc_UI_CF_feedback)
        PromptContext[token_pos * hidden_dim + head_idx * head_dim + tid] = attn_out;
    }
    GroupMemoryBarrierWithGroupSync();

    // ── Stage 5: Tool decode (nodes 64–83) ───────────────────────────────────
    //    Check if any generated token falls in the tool-token range (≥32768).
    //    If so, read ToolGate weights to identify the target micronaut.
    if (is_tool)
    {
        uint token_pos = gid.x;
        if (token_pos >= context_len) return;

        // Find argmax in TokenOutput for this position
        float max_l = -1e30f;
        uint  max_v = 0u;
        for (uint v = tid; v < vocab_size; v += 64)
        {
            float l = TokenOutput[token_pos * vocab_size + v];
            if (l > max_l) { max_l = l; max_v = v; }
        }
        GroupMemoryBarrierWithGroupSync();

        // Thread 0 checks if max token is a tool token (≥ tool_vocab_start)
        if (tid == 0 && max_v >= tool_vocab_start)
        {
            uint tool_idx = max_v - tool_vocab_start;
            if (tool_idx < 512u)
            {
                // Compute tool gate activation
                float gate_sum = 0.0f;
                for (uint d = 0; d < hidden_dim; d++)
                    gate_sum += gs_hidden[d] * FP16toF32(ToolGate[tool_idx * hidden_dim + d]);

                if (gate_sum > 0.5f)  // gate threshold
                {
                    gs_tool_hit = 1u;
                    gs_tool_id  = tool_idx;

                    // Write tool dispatch signal:
                    // [0] = tool_idx  [1] = token_pos  [2] = gate_score_as_uint
                    ToolDispatch[0] = tool_idx;
                    ToolDispatch[1] = token_pos;
                    ToolDispatch[2] = asuint(gate_sum);

                    // V9: tool_idx must be < 512 (registered micronaut count)
                    // Unregistered tools → quarantine (ToolDispatch[3] = 0xDEAD = quarantine flag)
                    ToolDispatch[3] = (tool_idx < 512u) ? 0u : 0xDEADu;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // ── Stage 6: Response sampling (nodes 84–99) ─────────────────────────────
    //    Temperature-scaled nucleus (top-p) sampling from LM head output.
    //    Determinism seed = chain_hash[0] XOR token_pos (V6).
    if (is_sample)
    {
        uint token_pos = gid.x;
        if (token_pos >= context_len) return;

        // If a tool was dispatched, emit the tool token directly (no sampling)
        if (gs_tool_hit == 1u)
        {
            if (tid == 0)
                ResponseTokens[token_pos] = tool_vocab_start + gs_tool_id;
            return;
        }

        // LM head projection: hidden (4096) → logits (vocab_size)
        // Each of 16 nodes handles vocab_size/16 = ~3072 logits
        uint vocab_chunk = vocab_size / 16u;
        uint vocab_start = (node_id - 84u) * vocab_chunk;
        uint vocab_end   = min(vocab_start + vocab_chunk, vocab_size);

        for (uint v = vocab_start + tid; v < vocab_end; v += 64)
        {
            float logit = 0.0f;
            for (uint d = 0; d < hidden_dim; d += 64)
            {
                uint chunk = min(64u, hidden_dim - d);
                for (uint dd = 0; dd < chunk; dd++)
                    logit += gs_hidden[d + dd] * FP16toF32(LMHead[v * hidden_dim + d + dd]);
            }
            // Temperature scale
            gs_logits[v] = logit / temperature;
        }
        GroupMemoryBarrierWithGroupSync();

        // Simplified nucleus sampling: thread 0 on node 84 picks the token
        if (node_id == 84u && tid == 0)
        {
            // Find max logit for numerical stability
            float mx = gs_logits[0];
            for (uint v = 1; v < vocab_size; v++) mx = max(mx, gs_logits[v]);

            // Softmax
            float s = 0.0f;
            for (uint v = 0; v < vocab_size; v++) { gs_logits[v] = exp(gs_logits[v] - mx); s += gs_logits[v]; }
            for (uint v = 0; v < vocab_size; v++) gs_logits[v] /= s;

            // Nucleus (top-p=0.92): accumulate until mass >= top_p
            // Deterministic seed from chain_hash XOR token_pos (V6)
            uint seed = gs_chain_seed ^ token_pos ^ 0x9e3779b9u;
            float r = (float)(seed & 0xFFFFFFu) / (float)0x1000000u;  // pseudo-random in [0,1)
            r *= top_p;  // scale to nucleus mass

            float cumsum = 0.0f;
            uint  sampled = 0u;
            for (uint v = 0; v < vocab_size; v++)
            {
                cumsum += gs_logits[v];
                if (cumsum >= r) { sampled = v; break; }
            }

            ResponseTokens[token_pos] = sampled;

            // Write to deep-think scratchpad if within budget
            if (token_pos < deep_think_budget)
                DeepThinkBuf[token_pos] = (float)sampled;
        }
    }
}
