# MICRONAUT-RUNTIME.md — Runtime version map

Three micronaut runtime generations under `dist/`. Each layer introduced
components that live on in the current stack.

---

## Lineage at a glance

```
V0 (Python folds + JS ROM + native D3D11)
  ↓  introduced: fold service contract, S7/tile model format, sidecar spec, GPT-2 HLSL shaders
cpp_runtime (C++ JSONL exe + attention library + C# bindings)
  ↓  introduced: JSONL protocol, glyph→op binding table, AttentionContext, XcfeRouter
v2 (.NET 9 binary + compiled HLSL shaders + native SCX runtime)
  ↓  introduced: MoE route shader, fold compute shaders, compiled .cso artifacts
kuhul-folds (.NET 8 DLL-reflected fold cycle + three-outcome Yax + MicronautScore reward)
  ↓  introduced: IFoldStage ABI, three-outcome admission (Strong/Weak/None), Chen reward loop, CycleIdentity
Current stack (json_runtime.exe + kuhul-server + GLSL + native.FOLD)
```

---

## `dist/Micronaut_V0/` — fold service platform (V0/V1)

The earliest complete runtime. Every fold is a separate Python Flask service.

### Fold services

15 folds, each has `folds/<FOLD>/fold.json` + a log/storage JSON + a Python service:

| fold | port | storage file | service |
|---|---|---|---|
| CODE_FOLD | 3202 | `storage.json` | `code_fold_service.py` |
| CONTROL_FOLD | 3204 | `control_log.json` | `control_fold_service.py` |
| COMPUTE_FOLD | 3217* | `compute_log.json` | `compute_fold_service.py` |
| DATA_FOLD | 3201 | `data_store.json` | `data_fold_service.py` |
| STORAGE_FOLD | 3203 | `storage.json` | `storage_fold_service.py` |
| AUTH_FOLD | — | `auth_log.json` | `auth_fold_service.py` |
| COMPILER | — | `compile_log.json` | `compiler_fold_service.py` |
| DB_FOLD | — | `db_events.json` | `db_fold_service.py` |
| EVENTS_FOLD | — | `events.json` | `events_fold_service.py` |
| NETWORK_FOLD | — | `network_routes.json` | `network_fold_service.py` |
| PATTERN_FOLD | — | `patterns.json` | `pattern_fold_service.py` |
| SPACE_FOLD | — | `space_index.json` | `space_fold_service.py` |
| STATE_FOLD | — | `state.json` | `state_fold_service.py` |
| TIME_FOLD | — | `ticks.json` | `time_fold_service.py` |
| UI | — | `renders.json` | `ui_fold_service.py` |

*COMPUTE_FOLD at 3217 is live in the current stack (runs `mock_cli.py execute_plan_dx12`).

### Sidecar contracts (LIVE — referenced by current stack)

```
sidecars/scxq2_gpu.sidecar.json   — @sidecar: "scxq2_gpu" v2.0
sidecars/glyph_compute.sidecar.json
```

`scxq2_gpu.sidecar.json` is the authoritative contract for the
`sidecar://scxq2_gpu/*` URIs in `programs/scxq2_tile_upload.json`:

| op | fn | description |
|---|---|---|
| `INIT_DEVICE` | `gpu_init` | Open SCXQ2 container, init D3D11, compile 3 shaders, alloc triple-buffer |
| `RESOLVE_HASH` | `gpu_resolve_hash` | layer/head/lane → SHA-256 hash string |
| `REQUEST_SHARD` | `gpu_request_shard` | disk read → slot C staging |
| `POLL_READY` | `gpu_poll_ready` | true when shard is on GPU |
| `DECODE_INT4` | `gpu_decode_int4` | dispatch `scxq2_int4_lane.hlsl` — packed uint → float |
| `RUN_ATTENTION` | `gpu_run_attention` | dispatch `tile_qk_softmax.hlsl` + `tile_v_mul.hlsl` |
| `READBACK` | `gpu_readback` | CopyResource → Map(READ) → CPU float[65536] |
| `ADVANCE` | `gpu_advance` | rotate triple-buffer A→recycle, B→A, C→B |

Lane map: `DICT=0, FIELD=1, LANE=2, EDGE=3`.
Alignment: shard_start=4096, tile_bytes=262144, scales_align=16, payload_align=256.

### GPU programs (V0 origin)

`programs/scxq2_tile_upload.json` — the 10-step triple-buffer GPU pipeline
documented in `PROGRAMS.md`. This is the authoritative source copy.

### Native layer

`native/` contains:
- `gpu_d3d11_engine.cpp/.h` — D3D11 device + compute dispatch
- `gpu_d3d12_engine.h` — D3D12 engine header
- `scxq2_loader.cpp/.h` — SCXQ2 container reader + triple-buffer
- `sc77_reader.h` — SC77 tile format reader

**GPT-2 training shaders** (`native/shaders/gpt2_*.hlsl`):

```
gpt2_embed_fwd / gpt2_embed_bwd      embedding forward + backward
gpt2_layernorm_fwd / _bwd            layer normalization
gpt2_attn_fwd / gpt2_attn_bwd        attention forward + backward
gpt2_gelu_fwd / gpt2_gelu_bwd        GELU activation
gpt2_matmul_fwd / gpt2_matmul_bwd    matmul forward + backward
gpt2_bias_bwd                         bias gradient
gpt2_lm_head                          language model head
gpt2_loss                             cross-entropy loss
gpt2_adam                             Adam optimizer step
```

Also: `kuhul_fold_compute.hlsl`, `kuhul_fold_meta.hlsl`, `kuhul_fold_storage.hlsl`,
`fused_scxq2_flash.hlsl`, `glyph_compute.hlsl`, `xvm_*.hlsl`.

### Model formats

`models/micro-model-1/`: tile-based binary (`.bin` tiles + `.deflate` compressed
+ `.bin.meta.json` per-tile metadata + `index.json`).

`models/micro-model-2,3/`: `.s7` binary format + `records.jsonl` training records.

### ROM tooling

```
kuhul-mkbootstrap.mjs  build .micronaut.bson ROM from JSON source
kuhul-verify.mjs       verify a .micronaut.bson against schema
kuhul-ctl.mjs swap     hot-swap: copy ROM to micronauts/ + update registry
```

BSON format uses `bson` + `ajv` npm modules. Fold name: `MICRONAUT_V1` (the
directory is called V0 but the tooling declares V1 internally).

### semantic_kernel_cpp

C++ SK implementation with DX12 executor (`dx12_executor.cpp`). Contains:
- `ir.h` / `ir.proto` — execution IR (protobuf schema)
- `compiler.cpp` — IR compiler
- `planner.cpp` — task planner
- `scheduler.cpp` — job scheduler
- `memory.cpp` — memory manager
- `quant.cpp` — INT8 quantization
- `simd_kernels.cpp` — SIMD compute kernels
- Built as VS solution `semantic_kernel_cpp.sln` (also CMake)

---

## `dist/micronaut_cpp_runtime/` — C++ JSONL micronaut runtime

A full C++ implementation of the micronaut subprocess protocol. Built
binary: `build-nnck/micronaut.exe`.

### JSONL protocol

```
stdin:  {"@kind":"micronaut.request.v1","op":"chat.generate","id":"r1","payload":{...}}
stdout: {"@kind":"micronaut.event.v1","id":"r1","type":"token","data":{"token":"...","index":0}}
        {"@kind":"micronaut.result.v1","ok":true,"id":"r1","op":"chat.generate","result":{...}}
```

Entry point: `main.cpp` — CLI (`--chat prompt`) or `run_stdin_loop()`.
Op path: stdin → `MicronautOp` → `scxq2_execute(op)` → `XcfeRouter::route()` → handler.

### XcfeRouter

`xcfe_router.h` — prefix-based op handler dispatch (mirrors XCFE primitive routing):

```cpp
void register_handler(const string& op_prefix, Handler handler);
MicronautResult route(const MicronautOp& op) const;
```

`register_builtin_handlers()` wires the default handler set.

### Attention library

20 attention variants registered in `micronauts.json` (`@kind: "micronauts.attention.v1"`),
executed via `attention_registry.cpp`:

| glyph | op | node range | fold | impl |
|---|---|---|---|---|
| ⨀ | MATMUL | 900–907 | TENSOR | `attention/matmul.cpp` |
| ◉ | SCALE | 908–915 | TENSOR | `attention/scale.cpp` |
| — | SOFTMAX | — | TENSOR | `attention/softmax.cpp` |
| — | ROTARY | — | TENSOR | `attention/rotary.cpp` (RoPE) |
| — | QKV_SPLIT | — | TENSOR | `attention/qkv_split.cpp` |
| — | KV_CACHE | — | MEMORY | `attention/kv_cache.cpp` |
| — | MHA | — | TENSOR | `attention/matmul.cpp` (base) |
| — | GQA | — | TENSOR | `attention/gqa.cpp` |
| — | MQA | — | TENSOR | `attention/mqa.cpp` |
| — | MLA | — | TENSOR | `attention/mla.cpp` |
| — | PAGED | — | MEMORY | `attention/paged_attn.cpp` |
| — | RING | — | KERNEL | `attention/ring_attn.cpp` |
| — | SLIDING | — | KERNEL | `attention/sliding.cpp` |
| — | SINK | — | MEMORY | `attention/sink_attn.cpp` |
| — | SPARSE | — | TENSOR | `attention/sparse_attn.cpp` |
| — | FLASH | — | KERNEL | `attention/flash_attn.cpp` |
| — | LINEAR | — | TENSOR | `attention/linear_attn.cpp` |
| — | CROSS | — | TENSOR | `attention/cross_attn.cpp` |
| — | LOCAL | — | KERNEL | `attention/local_attn.cpp` |
| — | HEAD_ROUTING | — | CONTROL | `attention/head_routing.cpp` |

Fold types here (`TENSOR / KERNEL / MEMORY / CONTROL`) are attention-layer
compute folds — distinct from the K'UHUL phase folds (Pop/Wo/Yax/Sek/Chen/Xul).

### AttentionContext

Mirrors `FoldCtx` at the tensor execution layer:

```cpp
struct AttentionContext {
    unordered_map<string, Tensor> tensors;   // named tensor store
    unordered_map<string, float>  scalars;   // head_dim, scale, etc.
    unordered_map<string, int64_t> ints;     // seq_len, num_heads, etc.
    vector<string> trace;                    // op trace (→ proof_trace analogue)
};
```

`Tensor` carries `name`, `shape`, and `vector<float> data`.
`OpResult` carries `ok`, `opcode`, `glyph`, `output`, `message`.

### C# integration layer

| file | role |
|---|---|
| `micronauts.cs` | C# hall monitor — active micronaut selection |
| `ScxqddsModelSource.cs` | SCXQDDS shard source adapter |
| `ScxqddsTieredSource.cs` | Tiered (hot/warm/cold) shard source |
| `Resolver.cs` | Micronaut capability resolver |
| `ExecutionProof.cs` | Proof chain record (→ Phase 7.6 ancestor) |
| `ToolHierarchy.cs` | Tool → micronaut hierarchy mapping |
| `MiniAdapter.cs` | Lightweight model adapter |

`gpu_queues.h` — GPU command queue management.

### SCXQ2 runtime (this layer)

`scxq2_runtime.cpp/.h` — JSONL → SCXQ2 frame → `XcfeRouter::route()`.
Not the same binary as `scxq2_runtime.exe` in the main stack.

---

## `dist/micronaut-v2/` — V2 .NET 9 + native SCX deployment

The V2 release binary. Two components:

### micronaut/micronaut.exe

.NET 9 (`net9.0`, `Microsoft.NETCore.App 9.0.0`) compiled binary.
Deps: `micronaut.deps.json`. No hostfxr bootstrapping needed —
self-contained .NET 9 deploy.

### scx_runtime/scx_runtime.exe

Native SCX runtime binary. Ships with:
- `dxcompiler.dll` — DirectX shader compiler (DXC)
- `dxil.dll` — DXIL signing library

**Compiled shaders (.cso):**

| cso | hlsl source | purpose |
|---|---|---|
| `int4_matmul.cso` | `int4_matmul.hlsl` | INT4 matrix multiply |
| `matmul_int4.cso` | `matmul_int4.hlsl` | INT4 matmul (alternate layout) |
| `moe_route.cso` | `moe_route.hlsl` | GPU top-2 MoE expert routing |
| `fabric_kernel_minimal.cso` | `fabric_kernel_minimal.hlsl` | Minimal fabric kernel |

**`moe_route.hlsl` — GPU MoE routing (top-2 over NUM_EXPERTS):**

```hlsl
cbuffer Params : register(b0) { uint NUM_EXPERTS; uint TOPK; };
StructuredBuffer<float> token  : register(t0);  // token[128]
StructuredBuffer<float> router : register(t1);  // router[NUM_EXPERTS × 128]
RWStructuredBuffer<uint> out_idx : register(u0);

[numthreads(32,1,1)] void main(...) {
    // dot product: token × router[e] for each expert
    // → best1 (id1) and best2 (id2)
    // out_idx[tid*2+0]=id1, out_idx[tid*2+1]=id2
}
```

This is the GPU analogue of the `native.FOLD` W·C·R Yax admission in the
current stack — both select the dominant expert; this one uses learned router
weights via dot product, the current stack uses declared W/C/R triples.

**Fold compute shaders (carried from V0):**

```
kuhul_fold_compute.hlsl   GPU fold state computation
kuhul_fold_meta.hlsl      fold metadata / phase tracking
kuhul_fold_storage.hlsl   fold storage ops
kuhul_fold_ui.hlsl        fold UI surface compute
kuhul.hlsl                main K'UHUL compute surface
kuhul.css.hlsl / kuhul.html.hlsl   CSS/HTML projection targets
```

---

## Cross-version connection map

| Concept | V0 | cpp_runtime | v2 | Current stack |
|---|---|---|---|---|
| **Fold services** | 15 Python Flask folds | TENSOR/KERNEL/MEMORY/CONTROL | kuhul_fold_*.hlsl | Pop/Wo/Yax/Sek/Chen/Xul via native.FOLD |
| **Model format** | .bin tiles + .s7 weights | — | — | GGUF + SafeTensors + SCXQDDS + S7L (Supernaut) |
| **MoE routing** | fold_registry routes | micronauts.json glyph→op | moe_route.cso (top-2 dot) | native.FOLD MuField W·C·R (declared) |
| **GPU backend** | D3D11 + HLSL | — | D3D11 + .cso | GLSL 430 (xcfe_gl_ops.dll) |
| **Sidecar contract** | scxq2_gpu.sidecar.json ← LIVE | — | — | sidecar://scxq2_gpu/* in programs |
| **SCXQ2 runtime** | scxq2_loader.cpp | scxq2_runtime.cpp | scx_runtime.exe | scxq2_runtime.exe |
| **SCX compiler** | — | — | scx_runtime + DXC | scxq2_runtime.exe |
| **Proof chain** | ExecutionProof.cs (ancestor) | AttentionContext.trace | — | merkle.cpp + FOLD_PROOF |
| **ROM tooling** | kuhul-mkbootstrap.mjs → .bson | — | — | JROM (CBOR) + JROM_EXEC op |
| **Attention** | xvm_fused_qkv_attention.hlsl | 20 cpp variants + glyph table | — | GLSL attention shader |
| **Runtime protocol** | HTTP fold dispatch | JSONL stdin/stdout | .NET 9 binary | HTTP JSON via kuhul-server:8764 |

---

## What's live vs historical

| artifact | status | used by |
|---|---|---|
| `Micronaut_V0/sidecars/scxq2_gpu.sidecar.json` | **LIVE contract** | `programs/scxq2_tile_upload.json` `sidecar://scxq2_gpu/*` |
| `Micronaut_V0/programs/scxq2_tile_upload.json` | **LIVE reference** | PROGRAMS.md 10-step pipeline |
| `Micronaut_V0/native/shaders/gpt2_*.hlsl` | Historical | Superseded by `trainer/` D3D11 trainer |
| `Micronaut_V0/models/micro-model-*/` | Historical | Format reference for S7L (Supernaut) |
| `micronaut_cpp_runtime/build-nnck/micronaut.exe` | Built binary | Local testing via `--chat` |
| `micronaut_cpp_runtime/micronauts.cs` | C# binding | Ancestor of `micronauts.cs` in main stack |
| `micronaut-v2/scx_runtime/shaders/moe_route.cso` | Compiled shader | Reference for GPU routing |
| `micronaut-v2/micronaut/micronaut.exe` | Deployed binary | V2 release |

---

## `dist/kuhul_folds/` — .NET 8 fold cycle runtime

The canonical C# phase machine. Each phase DLL implements `IFoldStage`; `FoldOrchestrator`
reflects them from `kuhul.fold-runtime.json` at startup (no concrete type references).

```
Kuhul.Folds.Contract/
  IFoldStage            — Execute(ref FoldContext c) → FoldResult
  FoldContext           — MuField, AdmissionOutcome, Reward, StrongThreshold, WeakThreshold, LearningRate
  MicronautScore        — Name, W (competence), C (confidence), R (relevance), Score = W×C×R
  AdmissionOutcome      — None | Weak | Strong

Pop/Wo/Yax/Sek/Chen/Xul — phase DLLs
Orchestrator/           — FoldOrchestrator.cs: traversal + CycleIdentity
kuhul.fold-runtime.json — manifest: cycle, admission block
```

### Three-outcome Yax admission

Yax scores `MuField` by `S = W×C×R`; all outcomes route to Sek — Chen and Xul always run:

| Outcome | Condition | Sek dispatch |
|---------|-----------|--------------|
| `STRONG` | `S ≥ 0.50` | execute V → output leases |
| `WEAK` | `0.15 ≤ S < 0.50` | `sidecar://micronaut-evolution/dispatch` |
| `NONE` | `S < 0.15` or empty field | `sidecar://micronaut-factory/dispatch` |

### Chen reward loop

```
reward = projectionScore + arcScore
  projectionScore: 0.6 if Outcome==Strong AND Result non-empty
  arcScore:        min(0.4, ArcState × 0.4)

Strong only: C ← C + lr × (reward − C)   [per-execution EMA, immediate]
W update deferred: Xul logs (node, W, C, R, reward, outcome) → MX-2 IDB across folds
```

W·C·R maps to QKV: W = V magnitude (competence), C = K signal (confidence), R = Q·K dot
(relevance). Yax is the attention head; `argmax(S)` = hard attention gate.

### CycleIdentity

`SHA256(SHA256(Pop.dll) ‖ SHA256(Wo.dll) ‖ … ‖ SHA256(Xul.dll))` — binary identity,
input-independent. Stable across runs with identical binaries.

---

## `dist/Quantum/build/` — Quantum Trinity executables

Five JSON-RPC C++ bots registered in `micronauts/registry.json` under
`sidecar://quantum/dispatch`. All smoke-tested (2026-08-25):

| Executable | Operations | Role |
|---|---|---|
| `quantum_hybrid.exe` | `process / analyze_code / extract_relations / extract_patterns / get_history` | CHEESE code-edge emitter (Roslyn parser + Regex + ELIZA + ADAM12) |
| `quantum_trinity.exe` | `research / analyze_ngrams / translate_notation / store_memory / retrieve_memory / get_metrics` | DuckDuckGo Instant API + Wikipedia API web research |
| `quantum_microagents.exe` | `process / get_agents / get_history / get_context / get_config_paths` | Candidate-only sidecar — 6 agent templates, `authority_boundary: "candidate_only"` |
| `quantum_personality.exe` | `interact / get_profile / get_personas` | Adaptive personality engine with CognitiveState (arousal/dominance/valence/trust/rapport/quantum_coherence) |
| `quantum_grammar.exe` | `parse / get_grammar / get_parse_tree` | Grammar parser |

**Authority constraint**: `quantum_microagents` emits candidate JSON structures only — it
never creates, updates, merges, or promotes micronauts. Every response carries
`authority_boundary: "candidate_only"`.

Source: `dist/Quantum/src/quantum_*.cpp`. Built with VS2022 `cl` / vcvars64.

---

## Related docs
- `MICRONAUTS.md` — Moµ field, W·C·R scoring, FoldContext, native.FOLD, Supernaut.
- `W-C-R.md` — W·C·R admission law, MuScore/MuField, GPU shader, Merkle chain.
- `PROGRAMS.md` — `native.FOLD` primitive, `scxq2_tile_upload.json` 10-step pipeline, sidecar://scxq2_gpu/*.
- `FOLDS.md` — Fold/Node + SCXQDDS shard format.
- `GPU.md` — provider inventory, DirectML chain.
