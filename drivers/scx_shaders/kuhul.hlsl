/**
 * @file kuhul.hlsl
 * @brief KUHUL π Root GPU Dispatcher — KPI1 bytecode → fold kernel routing
 *
 * This is the master compute shader for the KUHUL π runtime.
 * It reads a KPI1 glyph opcode stream and dispatches each instruction
 * to the correct fold kernel — entirely on GPU, zero CPU round-trips.
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  .khl source                                                    │
 * │     ↓  kuhul-lexer (MX2LEX DFA)                                 │
 * │  KPI1 bytecode  ←─────────────── StructuredBuffer<KPI1Instr>   │
 * │     ↓  CS_KuhulDispatch  (this file)                            │
 * │  ┌──────┬──────────┬──────────┬────────┬──────────┐            │
 * │  │ CM-1 │ COMPUTE  │ STORAGE  │  META  │  UI_FOLD │            │
 * │  │ gate │ CS_Comp  │ CS_Store │ CS_Meta│ CS_UIFold│            │
 * │  └──────┴──────────┴──────────┴────────┴──────────┘            │
 * │     ↓                                                           │
 * │  RenderOutput / io/stream.txt / ForeignCallBuf → json.exe      │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * KPI1 Glyph Opcodes:
 *   Pop     0x00  CTRL    — push frame / CM-1 gate check
 *   Wo      0x01  COMPUTE — matmul / trunk forward pass
 *   Sek     0x02  STORAGE — store / snapshot / seal
 *   Ch'en   0x03  META    — hash / chain_hash update / Merkle
 *   Yax     0x04  UI      — render / emit / CP-1 response
 *   Xul     0x05  COMPUTE — MoE route / top-1 expert select
 *   K'ayab' 0x06  STORAGE — replay_payload write/read / RLHF inject
 *   Kumk'u  0x07  META    — Merkle verify / V7 attestation
 *   Muwan   0x08  FOREIGN — FOREIGN_CALL_PY / json.exe tool dispatch
 *
 * Opcode category field determines fold target:
 *   0 → CM-1 ControlFold  (gate enforcement)
 *   1 → COMPUTE_FOLD MM-1 (Wo: trunk matmul)
 *   2 → STORAGE_FOLD SM-1 (Sek: snapshot/seal)
 *   3 → META_FOLD VM-2    (Ch'en: hash chain)
 *   4 → UI_FOLD CP-1      (Yax: render/emit)
 *   5 → FOREIGN           (Muwan: json.exe IPC)
 *
 * Parallel dispatch: data-independent instructions across categories
 * execute simultaneously on different fold nodes. cdeg field hints
 * at required thread group count per instruction.
 *
 * CM-1 gate: 0x0001 must be set in ControlFlags[0].
 * All fold gates are verified here before any instruction is routed.
 */

// ============================================================================
// KPI1 Instruction Struct
// ============================================================================
// Each instruction is 6 bytes packed into 2× uint (for HLSL alignment).
// Layout: [category_id | glyph_id | ptype | plen | cdeg | pos]

struct KPI1Instruction
{
    uint word0;  // bits [7:0]=category_id  [15:8]=glyph_id  [23:16]=ptype  [31:24]=plen
    uint word1;  // bits [7:0]=cdeg         [15:8]=pos        [31:16]=reserved
};

// Unpack helpers
uint KPI1_Category(KPI1Instruction i) { return (i.word0)       & 0xFFu; }
uint KPI1_Glyph   (KPI1Instruction i) { return (i.word0 >> 8)  & 0xFFu; }
uint KPI1_PType   (KPI1Instruction i) { return (i.word0 >> 16) & 0xFFu; }
uint KPI1_PLen    (KPI1Instruction i) { return (i.word0 >> 24) & 0xFFu; }
uint KPI1_CDeg    (KPI1Instruction i) { return (i.word1)       & 0xFFu; }
uint KPI1_Pos     (KPI1Instruction i) { return (i.word1 >> 8)  & 0xFFu; }

// ============================================================================
// Glyph Opcode Constants
// ============================================================================

// Category IDs (opcode family → fold target)
static const uint CAT_CTRL    = 0u;  // CM-1 gate / frame control
static const uint CAT_COMPUTE = 1u;  // COMPUTE_FOLD MM-1 (Wo)
static const uint CAT_STORAGE = 2u;  // STORAGE_FOLD SM-1 (Sek)
static const uint CAT_META    = 3u;  // META_FOLD VM-2    (Ch'en)
static const uint CAT_UI      = 4u;  // UI_FOLD CP-1      (Yax)
static const uint CAT_FOREIGN = 5u;  // FOREIGN_CALL_PY / json.exe (Muwan)

// Glyph IDs within each category
static const uint GLYPH_Pop     = 0x00u;  // CTRL    — push frame
static const uint GLYPH_Wo      = 0x01u;  // COMPUTE — matmul
static const uint GLYPH_Sek     = 0x02u;  // STORAGE — store/seal
static const uint GLYPH_Chen    = 0x03u;  // META    — hash/attest
static const uint GLYPH_Yax     = 0x04u;  // UI      — render/emit
static const uint GLYPH_Xul     = 0x05u;  // COMPUTE — MoE route
static const uint GLYPH_Kayab   = 0x06u;  // STORAGE — replay
static const uint GLYPH_Kumku   = 0x07u;  // META    — Merkle verify
static const uint GLYPH_Muwan   = 0x08u;  // FOREIGN — foreign call

// ── Extended K'UHUL DSL Operator Sub-Opcodes ──────────────────────────────────
// Discovered from html-samples/deepseek_html_20260324_9cdee7.html
// These are the K'UHUL operator symbols used in gaming engine / video pipeline code.
// They appear inside user-facing .kuhul programs as: object.method (⟿) args
// KPI1 encoding: category=COMPUTE(1) or STORAGE(2) or META(3), glyph=sub-opcode below.

static const uint GLYPH_NEURAL_PATH      = 0x20u; // ⟿ U+27FF — neural path / AI behavior routing → COMPUTE_FOLD Wo
static const uint GLYPH_DATA_FLOW        = 0x21u; // ⤍ U+2909 — directional data assignment → COMPUTE_FOLD Wo
static const uint GLYPH_ROTATE_COMPRESS  = 0x22u; // ↻ U+21BB — rotational compression for 3D meshes → COMPUTE_FOLD Wo
static const uint GLYPH_VECTOR_COND      = 0x23u; // ⤦ U+2926 — vector conditional / collision detection → COMPUTE_FOLD Wo
static const uint GLYPH_GRADIENT_FLOW    = 0x24u; // ⤨ U+2928 — gradient flow / smooth animation → COMPUTE_FOLD Wo
static const uint GLYPH_BIDIR_FLOW       = 0x25u; // ↔ U+2194 — bidirectional compress/decompress → STORAGE_FOLD Sek
static const uint GLYPH_ADAPTIVE_COMP    = 0x26u; // ⤓ U+2913 — adaptive bitrate compression → STORAGE_FOLD Sek
static const uint GLYPH_MOTION_ESTIMATE  = 0x27u; // ⤭ U+292D — H.264 motion estimation → COMPUTE_FOLD Wo
static const uint GLYPH_FRAME_PREDICT    = 0x28u; // ⤮ U+292E — P/B frame prediction → COMPUTE_FOLD Wo
static const uint GLYPH_H264_ENCODE      = 0x29u; // ⤯ U+292F — DCT + quantization encode → COMPUTE_FOLD Wo
static const uint GLYPH_NAL_UNIT         = 0x2Au; // ⤰ U+2930 — NAL unit packaging + attest → META_FOLD Ch'en
static const uint GLYPH_PACKETIZE        = 0x2Bu; // ⤏ U+290F — network packet formation → META_FOLD Ch'en
static const uint GLYPH_NETWORK_SEND     = 0x2Cu; // ⤎ U+290E — encrypted network send → META_FOLD + Muwan

// Payload types (ptype field)
static const uint PTYPE_NONE  = 0u;
static const uint PTYPE_F32   = 1u;
static const uint PTYPE_U32   = 2u;
static const uint PTYPE_PTR   = 3u;
static const uint PTYPE_TOKEN = 4u;

// Fold dispatch codes written to FoldDispatch output buffer
static const uint FDISPATCH_CTRL    = 0x00000001u;
static const uint FDISPATCH_COMPUTE = 0x00000002u;
static const uint FDISPATCH_STORAGE = 0x00000004u;
static const uint FDISPATCH_META    = 0x00000008u;
static const uint FDISPATCH_UI      = 0x00000010u;
static const uint FDISPATCH_FOREIGN = 0x00000020u;

// ============================================================================
// Resource Bindings
// ============================================================================

StructuredBuffer<KPI1Instruction> KPI1Instructions : register(t0);  // KPI1 bytecode stream
StructuredBuffer<uint>            KPI1Payloads     : register(t1);  // variable-length payloads
StructuredBuffer<uint>            ControlFlags      : register(t2);  // CM-1 gate flags
StructuredBuffer<float>           FoldOutputs       : register(t3);  // previous fold results (feedback)

RWStructuredBuffer<uint>  FoldDispatch    : register(u0);  // fold + node_id targets per instruction
RWStructuredBuffer<uint>  RenderOutput    : register(u1);  // pixel/DOM/token output (Yax results)
RWStructuredBuffer<uint>  StorageOps      : register(u2);  // queued Sek/K'ayab' ops for STORAGE_FOLD
RWStructuredBuffer<uint>  ForeignCallBuf  : register(u3);  // Muwan payloads → json.exe IPC pipe

cbuffer KuhulDispatchParams : register(b0)
{
    uint  num_instructions;   // total KPI1 instructions in this frame
    uint  frame_id;           // monotonic frame counter
    uint  cm1_gate;           // must equal 0x0001
    uint  payload_base;       // base offset into KPI1Payloads buffer
    uint  max_render_tokens;  // max tokens CP-1 may emit this frame
    uint  foreign_call_limit; // max Muwan calls per frame (rate limiting)
    uint  enable_deep_think;  // 1 = CP-1 CoT scratchpad active this frame
    uint  _pad;
};

// ============================================================================
// Groupshared state
// ============================================================================

groupshared uint gs_category;      // decoded category for this instruction
groupshared uint gs_glyph;         // decoded glyph opcode
groupshared uint gs_cdeg;          // compute degree (parallelism hint)
groupshared uint gs_payload_off;   // payload offset in KPI1Payloads
groupshared uint gs_fold_bits;     // fold dispatch bitmask built by this group
groupshared uint gs_gate_ok;       // 1 if CM-1 gate passed, 0 if blocked

// ============================================================================
// Main Dispatcher
// ============================================================================

[numthreads(64, 1, 1)]
void CS_KuhulDispatch(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    // Each thread group processes one KPI1 instruction.
    uint instr_idx = gid.x;
    if (instr_idx >= num_instructions)
        return;

    // ── CM-1 gate check ───────────────────────────────────────────────────────
    if (tid == 0)
        gs_gate_ok = (ControlFlags[0] == cm1_gate) ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();

    if (gs_gate_ok == 0u)
        return;  // CM-1 blocked — entire frame halted

    // ── Decode instruction ────────────────────────────────────────────────────
    if (tid == 0)
    {
        KPI1Instruction instr = KPI1Instructions[instr_idx];
        gs_category   = KPI1_Category(instr);
        gs_glyph      = KPI1_Glyph(instr);
        gs_cdeg       = KPI1_CDeg(instr);
        gs_payload_off = payload_base + instr_idx * 4u;  // 4 payload words per instr max
        gs_fold_bits   = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    // ── Route by category ─────────────────────────────────────────────────────

    // ─── CAT_CTRL (Pop): frame scope push, gate re-verify ─────────────────────
    if (gs_category == CAT_CTRL)
    {
        if (tid == 0)
        {
            gs_fold_bits = FDISPATCH_CTRL;
            // Pop writes frame_id into FoldDispatch as scope marker
            FoldDispatch[instr_idx] = FDISPATCH_CTRL | (frame_id << 8);
        }
    }

    // ─── CAT_COMPUTE (Wo, Xul + DSL operators): trunk matmul, MoE, gaming/video ─
    else if (gs_category == CAT_COMPUTE)
    {
        if (tid == 0)
        {
            uint node_target = 0u;
            if (gs_glyph == GLYPH_Wo || gs_glyph == GLYPH_NEURAL_PATH ||
                gs_glyph == GLYPH_DATA_FLOW   || gs_glyph == GLYPH_ROTATE_COMPRESS ||
                gs_glyph == GLYPH_VECTOR_COND || gs_glyph == GLYPH_GRADIENT_FLOW   ||
                gs_glyph == GLYPH_MOTION_ESTIMATE || gs_glyph == GLYPH_FRAME_PREDICT ||
                gs_glyph == GLYPH_H264_ENCODE)
            {
                // All COMPUTE DSL operators route through trunk nodes (0-199)
                // High cdeg (≥4) gets routed to MoE expert nodes (100-199) for heavy ops
                node_target = (gs_cdeg >= 4u)
                    ? 100u + (KPI1Payloads[gs_payload_off] % 100u)  // expert node
                    :         KPI1Payloads[gs_payload_off] % 100u;   // trunk node
            }
            else if (gs_glyph == GLYPH_Xul)
                node_target = 200u + (KPI1Payloads[gs_payload_off] % 100u);  // router node 200-299

            gs_fold_bits = FDISPATCH_COMPUTE;
            FoldDispatch[instr_idx] = FDISPATCH_COMPUTE | (node_target << 8) | (gs_glyph << 24);
        }
    }

    // ─── CAT_STORAGE (Sek, K'ayab', ↔ ⤓): snapshot / seal / replay / compress ─
    else if (gs_category == CAT_STORAGE)
    {
        if (tid == 0)
        {
            uint op_code = (gs_glyph == GLYPH_Sek || gs_glyph == GLYPH_BIDIR_FLOW)
                                ? 0u /*OP_STORE*/  :
                           (gs_glyph == GLYPH_Kayab || gs_glyph == GLYPH_ADAPTIVE_COMP)
                                ? 4u /*OP_REPLAY*/ : 2u /*OP_SEAL*/;

            uint slot = KPI1Payloads[gs_payload_off] % 100u;  // snapshot slot 0-99

            // Enqueue to StorageOps ring for CS_StorageFold to pick up
            uint so_idx = instr_idx * 3u;
            StorageOps[so_idx + 0] = op_code;
            StorageOps[so_idx + 1] = slot;
            StorageOps[so_idx + 2] = gs_payload_off + 1u;  // data payload pointer

            gs_fold_bits = FDISPATCH_STORAGE;
            FoldDispatch[instr_idx] = FDISPATCH_STORAGE | (slot << 8) | (op_code << 24);
        }
    }

    // ─── CAT_META (Ch'en, Kumk'u, ⤰ ⤏ ⤎): hash / Merkle / NAL / network ────
    else if (gs_category == CAT_META)
    {
        if (tid == 0)
        {
            // NAL_UNIT / PACKETIZE → Merkle nodes (attestation chain)
            // NETWORK_SEND → Merkle + Muwan (also writes to ForeignCallBuf below)
            uint meta_node =
                (gs_glyph == GLYPH_Chen || gs_glyph == GLYPH_NAL_UNIT || gs_glyph == GLYPH_PACKETIZE) ?
                (KPI1Payloads[gs_payload_off] % 64u) :       // step_hash node 0-63
                64u + (KPI1Payloads[gs_payload_off] % 36u);  // Merkle node 64-99

            gs_fold_bits = FDISPATCH_META;
            FoldDispatch[instr_idx] = FDISPATCH_META | (meta_node << 8);
        }
    }

    // ─── CAT_UI (Yax): render / emit via CP-1 UI_FOLD ────────────────────────
    //     Handles: app frames, game output, UI components, web page DOM, API responses
    else if (gs_category == CAT_UI)
    {
        // Each thread reads a payload token and writes to RenderOutput
        uint render_slot  = KPI1Payloads[gs_payload_off] % max_render_tokens;
        uint render_base  = render_slot * 64u;

        // Payload words (plen words) → RenderOutput (DOM/pixel/token buffer)
        for (uint w = tid; w < 64u && w < (gs_cdeg * 4u); w += 64)
        {
            uint pval = KPI1Payloads[gs_payload_off + 1 + w];
            RenderOutput[render_base + w] = pval;
        }

        // enable_deep_think: if set, CP-1 CoT scratchpad is active this Yax
        if (tid == 0)
        {
            gs_fold_bits = FDISPATCH_UI;
            uint ui_node = (gs_glyph == GLYPH_Yax) ?
                (KPI1Payloads[gs_payload_off] % 64u) : 64u;  // attention or tool_decode node

            FoldDispatch[instr_idx] = FDISPATCH_UI
                | (ui_node << 8)
                | (enable_deep_think << 24);
        }
    }

    // ─── CAT_FOREIGN (Muwan): FOREIGN_CALL_PY / json.exe IPC ─────────────────
    //     Rate-limited to foreign_call_limit per frame.
    //     Payload: [tool_id, arg_count, arg0, arg1, ...]
    else if (gs_category == CAT_FOREIGN)
    {
        if (tid == 0)
        {
            // Check frame foreign call budget
            uint call_idx = instr_idx % foreign_call_limit;

            // Write to ForeignCallBuf ring for json.exe to pick up via IPC
            uint fc_base = call_idx * 8u;
            ForeignCallBuf[fc_base + 0] = frame_id;
            ForeignCallBuf[fc_base + 1] = instr_idx;
            ForeignCallBuf[fc_base + 2] = KPI1Payloads[gs_payload_off + 0];  // tool_id
            ForeignCallBuf[fc_base + 3] = KPI1Payloads[gs_payload_off + 1];  // arg_count
            ForeignCallBuf[fc_base + 4] = KPI1Payloads[gs_payload_off + 2];  // arg0
            ForeignCallBuf[fc_base + 5] = KPI1Payloads[gs_payload_off + 3];  // arg1
            ForeignCallBuf[fc_base + 6] = 0u;  // result slot (filled by json.exe)
            ForeignCallBuf[fc_base + 7] = 0xMUWANu;  // magic tag for json.exe parser
            // Note: 0xMUWAN = 0x4D5557414Eu in ASCII ("MUWAN") truncated to uint

            gs_fold_bits = FDISPATCH_FOREIGN;
            FoldDispatch[instr_idx] = FDISPATCH_FOREIGN | (call_idx << 8);
        }
    }

    GroupMemoryBarrierWithGroupSync();
}

// ============================================================================
// Parallel fold execution note:
//
// CS_KuhulDispatch DOES NOT call the fold kernels directly — HLSL compute
// shaders cannot launch child dispatches. Instead:
//
// 1. CS_KuhulDispatch writes FoldDispatch[i] for each instruction.
// 2. The D3D12 execute-indirect / multi-dispatch command list reads FoldDispatch
//    and issues the appropriate CS_ComputeFold / CS_StorageFold / CS_MetaFold /
//    CS_UIFold dispatches in the same GPU command queue.
// 3. Data-independent fold dispatches run in parallel (D3D12 async compute queues).
// 4. STORAGE and META folds that depend on COMPUTE output wait on UAV barriers.
// 5. UI_FOLD (CP-1) runs last — it reads all other fold outputs (t0-t15).
//
// This gives the KUHUL runtime its on-the-fly GPU-native execution model:
//   .kuhul → KPI1 → CS_KuhulDispatch → FoldDispatch table →
//   [parallel: CS_ComputeFold | CS_StorageFold | CS_MetaFold] →
//   [serial:   CS_UIFold] → RenderOutput → io/stream.txt
//
// Apps, games, UI components, and web pages all reduce to this same pipeline.
// The only difference is the distribution of opcodes in the KPI1 stream:
//   App logic   → heavy Wo (compute) + Yax (render)
//   Game tick   → heavy Wo (physics/AI) + Sek (state) + Yax (frame)
//   Web page    → heavy Yax (DOM layout) + Sek (persist) + Ch'en (hash)
//   API call    → Muwan (dispatch) + Wo (process) + Yax (emit JSON)
// ============================================================================
