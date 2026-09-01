# MICRONAUTS.md — the Moµ (Mixture of Micronauts) system

A **micronaut** is a small, explicitly-declared semantic specialist. The Moµ
mixture routes work across a field of micronauts — the semantic analogue of a
Mixture-of-Experts (MoE) router, but over **declared capabilities** rather than
learned sub-networks.

## The core law

```
Micronaut identity  ≠  Micronaut implementation
```

A micronaut is a **semantic contract** with a **resolvable implementation**.
The same micronaut can be realized in C#, C++, or GLSL depending on capability
and hardware.

```
Micronaut = semantic contract + resolvable implementation
```

## The layers

```
micronaut.json     WHAT the micronaut is (capability / bounds / W / inputs / outputs)
      ↓
micronauts.cs      WHICH micronaut becomes active (Pop → Wo → Yax → Sek → Ch'en → Xul)
      ↓
XCFE               WHICH provider can realize it
      ↓
micronaut.glsl/.hlsl   HOW this micronaut executes on this GPU
```

- `micronaut.json` — the semantic identity (declarative).
- `micronauts.cs` — the hall monitor: decides which micronaut may act.
- `XCFE` — binds the micronaut to a provider (D3D11 / GLSL / CPU).
- `micronaut.glsl/.hlsl` — the GPU-resident realization (W·C·R field scoring).

## The fold cycle

Micronauts move through the K'uhul fold cycle:

```
Pop → Wo → Yax → Sek → Ch'en → Xul
```

Each fold is a phase of the micronaut's residency and authority.

## W·C·R scoring

The candidate field is scored by competence, confidence, and relevance:

```
S_µ = W_µ · C_µ · R_µ
```

The dominant/admissible micronaut is the one that wins the field competition —
performed on the GPU by `micronaut.glsl` / `micronaut.hlsl`:

```glsl
S = weight * confidence * relevance;
if (S >= 0.5) admitted++;
// workgroup reduction -> dominant index
```

## The symmetry with GPT-OSS

```
GPT-OSS MoE:   router field → 32 expert scores → top-4 → expert GPU kernels
Micronaut Moµ: semantic field → N micronaut scores → dominant/admitted → micronaut GPU kernel
```

Same shape — but GPT-OSS expert identity is **learned**; micronaut identity is
**explicitly declared**.

## The coder micronaut family (grammar registry)

Each grammar type (language) maps to its own coder micronaut:

```
c → c-coder-µ · cpp → cpp-coder-µ · python → py-coder-µ
typescript → ts-coder-µ · javascript → js-coder-µ · rust → rust-coder-µ · go → go-coder-µ
```

### Open grammar — novel grammars are evolved, not rejected

The registry is a **seed, not a closed set**. A novel grammar (e.g. XJSON) is
handled by a two-stage lifecycle:

```
user provides XJSON code
      ↓
grammar_distance → nearest known coder (bootstrap)
      ↓
FACTORY (micronaut_factory_core.dll) CREATES  xjson-grammar-µ
      ↓
EVOLUTION (micronaut_evolution.dll) UPDATES it as more XJSON code is seen
      ↓
registered → registry grows
```

The grammar micronaut is **created once, then refined** — factory is the birth,
evolution is the growth.

## DDS addresses micronauts too

DDS doesn't only address neural weights — it addresses the micronaut resource
space:

```
DDS Fold: MICRONAUTS
├─ Node CODE-µ    → code_micronaut.glsl
├─ Node ARC-µ     → arc_micronaut.glsl
└─ Node ROUTE-µ   → routing_micronaut.glsl
```

## The native builds

- `micronaut-factory/` — `micronaut_factory_core.dll` (creates grammar
  micronauts) + `micronaut_evolution.dll` (updates them).
- `micronaut-coder/` — `micronaut_coder.exe`, `micronaut_code_reviewer.exe`,
  and the `micronaut_lang.y` grammar registry.
- `micronaut_cpp_runtime/` — the C# XCFE runtime (`micronauts.cs`,
  `ScxqddsModelSource.cs`, `Resolver.cs`, `ExecutionProof.cs`,
  `ToolHierarchy.cs`, `MiniAdapter.cs`, `gpu_queues.h`).

## FoldContext — the state carrier

`native.FOLD` carries all inter-stage state in a `FoldContext` struct:

```cpp
struct FoldCtx {
    string fold;          // current phase name
    string node_name;     // active node
    string result;        // accumulated result text
    string residency;     // "WARM" | "COLD"
    vector<MuScore> mu_field;     // the live µ-field
    vector<string> proof_trace;   // per-stage hash chain (→ Merkle root)
    vector<string> provider_req;  // WARM/D3D11 requirements from Wo
};
```

`mu_field` is the list of `MuScore` candidates:

```cpp
struct MuScore {
    string name;
    double W = 0.5;   // weight / competence
    double C = 0.5;   // confidence
    double R = 0.5;   // relevance
    double score() const { return W * C * R; }
};
```

Pop seeds the field. Yax reads it and admits the dominant micronaut (highest
`score()`). Sek dispatches to that micronaut. Chen records the result in
`proof_trace`. Xul commits the final result.

## native.FOLD — the in-process C++ primitive

Added to `xcfe.cpp` as the `"native.FOLD"` primitive. The six fold stages are
implemented directly in C++ — no subprocess, no CLR call (yet):

| fn | description |
|---|---|
| `fn=cycle` | Run Pop→Wo→Yax→Sek→Chen→Xul N times (default 2). Returns `{fold, node, result, residency, proof_trace, providers, cycle_identity, rounds, ok}`. |
| `fn=stage` | Run a single named stage. Accepts `@ctx` to continue an existing FoldContext. |
| `fn=proof` | Phase 7.6 — SHA-256 Merkle root over `@trace` or `@ctx.proof_trace`. Returns `{root, leaf_count, ok}`. |

`cycle_identity` is SHA-256 of the concatenated SHA-256 hashes of each stage
name byte string (`Pop`, `Wo`, `Yax`, `Sek`, `Chen`, `Xul`), uppercased. This
is deterministic and reproducible without the fold DLL files.

### JSON program usage

```json
{ "@op": "FOLD_CYCLE", "@rounds": 3, "@out": "fold_result" }
{ "@op": "FOLD_STAGE", "@stage": "Yax", "@ctx": "$fold_result", "@out": "yax_ctx" }
{ "@op": "FOLD_PROOF", "@ctx": "$fold_result", "@out": "proof" }
```

## Supernaut — the fold cycle at the inference layer

`kuhul_supernaut` (static lib built from `dist/json-runtime/supernaut/`) is the
fold cycle realized at the model inference layer. The mapping is exact:

| Supernaut | Fold cycle |
|---|---|
| `NumaticManifoldEngine::resolve_geodesic` layer L1 | Pop |
| Layer L2 | Wo |
| Layer L3 | Yax |
| Layer L4 | Sek |
| Layer L5 | Chen |
| Layer L6 | Xul |
| `DaemonSession` | FoldContext |
| `DaemonSession::replay_lanes` | `proof_trace` |
| `DaemonSession::session_hash` | `cycle_identity` |
| `SemanticGenome` field | µ-field (MuScore list) |

Supernaut adds: SIMD MATMUL via DirectXMath, bimodal manifold fusion
(A_flat ⊗ A_curved), S7Mini transformer inference (INT8 quantized, loaded from
`.s7l` files), and the SHA-256 Merkle proof chain in `s7l/merkle.cpp`.

## Phase 7.6 — Proof chain

`supernaut/s7l/merkle.cpp` implements a complete self-contained SHA-256 Merkle
tree (no external deps). Leaf sources:

- `DaemonSession::replay_lanes` — per-round lane hashes
- `SCXQDDS` shard SHA-256 hashes — shard integrity leaves
- `FoldCtx::proof_trace` — per-stage result hashes from `native.FOLD`

`FOLD_PROOF` op and `fn=proof` in `native.FOLD` both use `merkle_root_from_strings()`
to produce the root. `verify_merkle_root()` checks inclusion.

## Phase 7.8 — CLR fold host (scaffold)

`supernaut/src/clr_fold_host.cpp` hosts .NET 8 fold assemblies
(Pop/Wo/Yax/Sek/Chen/Xul DLLs) **in-process** via `hostfxr` (NOT `mscoree` —
the fold DLLs target `net8.0`, not .NET Framework 4.8).

API surface:
```cpp
bool clr_fold_host_init(const wstring& runtime_config_path);
FoldJsonBridgeFn clr_fold_load_stage(const wstring& assembly_path,
                                      const wstring& type_name);
void clr_fold_host_shutdown();
```

Gap: each fold assembly needs a `[UnmanagedCallersOnly]` static `FoldEntry`
bridge method exposing `int FoldEntry(IntPtr ctxJson, IntPtr resultBuf, int bufLen)`.
When present, `clr_fold_load_stage` returns a live `FoldJsonBridgeFn` and the
`fn=stage` dispatch can route to the managed implementation.

## Registry (`micronauts/registry.json`)

Current micronaut categories and fold assignments:

| name | fold | category | notes |
|---|---|---|---|
| `tool_call` | — | system | structured tool output, T=0.1 |
| `chat` | — | system | free-text, T=0.8 |
| `memory` | Pop | specialist | long-context retrieval |
| `coder` | Sek | specialist | deterministic code gen, T=0.15 |
| `eliza` | Ch'en | specialist | deep thinking, quality tier |
| `librarian` | Yax | specialist | knowledge, quality tier |
| `ui` | Xul | specialist | concise UI output |
| `khanary` | Wo | persona | main orchestration persona |
| `factory` | Wo | meta | spawns/evolves new micronauts |
| `evolution` | — | meta | high entropy discovery, T=1.0 |
| `default` | — | system | neutral baseline |
| `stack_doc` | Pop | stack | stack documentation guide |
| `primeos_guide` | Xul | stack | PRIMEOS desktop app guide |
| `scx_guide` | Sek | stack | SCXQ2/SCX bytecode guide |
| `asx_guide` | Wo | stack | ASX physics/routing guide |
| `distillation_guide` | Ch'en | stack | LoRA distillation guide |
| `compiled_model` | Pop | stack | compiled micronaut data model |
| `pop/wo/yax/sek/chen/xul` | (self) | fold | fold-phase specialists |
| `ts_coder` | Sek | specialist | TypeScript fine-tune, T=0.1 |
| `programs` | Wo | meta | JSON program dispatcher |

## Related docs
- `W-C-R.md` — complete W·C·R scoring law, MuScore/MuField types, GPU shader,
  physics scalars, cycle identity, Merkle proof chain.
- `NNC-K.md` — NNC = hardware-independent neural execution semantics; the
  micronaut is one specialist, NNC is the whole environment's GPU vocabulary.
- `GLSL.md` — the GLSL path (HLSL→GLSL map, GLSL_server).
- `GPU.md` — provider inventory.
- `FOLDS.md` — Fold/Node + SCXQDDS.
- `PROGRAMS.md` — `native.FOLD` primitive, FOLD_CYCLE/STAGE/PROOF ops.
