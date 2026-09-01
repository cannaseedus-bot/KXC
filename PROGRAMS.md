# PROGRAMS.md — json-runtime JSON Program System

> Authoritative location: `programs/`  (repo root)
> Dist copy (subset):    `dist/v3.5.0-WebX/bin/json-runtime/programs/`
> Runtime:               `dist/json-runtime/build/Release/json_runtime.exe`
> Runtime port:          `9000` (bootstrap.json / micronaut-ui-chat-app.ps1)
> API prefix:            `/api`

---

## Core stack law

```
micronauts   — the semantic field: W·C·R capability nodes, shared K-Cube geometry, lane specialists
programs     — declarative JSON control graphs (this layer)
json_runtime — C++ executor: XCFE evaluator, 9 native primitive families, sidecar dispatch
stdlib       — the 9 primitive namespaces (native.READ/WRITE/EVAL/HASH/CALL/PHASE/STACK/PROGRAM/FOLD)
```

These four are the complete executable stack. Everything else is built on top:
- **Sugar** — KAST/KHL/kuhul-es compiler chain, MCP gateway, driver contracts
- **Models** — GGUF/SafeTensors/SCXQDDS weight files consumed via sidecar or kuhul_engine
- **UI** — PRIMEOS shell, WebView2 bridge, studio-dist

## What it is

A **JSON control graph** — a declarative program that json_runtime.exe executes
natively (C++, no Node/JS). Programs are the data layer of ASX OS:
not scripts, not eval strings — typed op-graphs that route through declared
primitives. No JSON ever touches the GPU; it orchestrates the path to native dispatch.

```
@program  —  program identity / name
@state    —  initial mutable state (key/value store)
@ops      —  op registry (compose new ops from primitives or other @ops)
@control  —  ordered execution steps (the program body)
```

---

## Runtime schema

```json
{
  "@program": "name",
  "@state":   { "key": "initial_value" },
  "@ops":     { "MY_OP": { "@type": "composed", "@exec": { "@op": "ADD" } } },
  "@control": [
    { "@op": "ADD", "@in": ["a", "b"], "@out": "c" },
    { "@op": "IF",  "@cond": "$c_big", "@then": {...}, "@else": {...} }
  ]
}
```

`$key` references resolve from `@state`. `@in` / `@args` mix literals and `$key` refs.
`@out` writes the result back into `@state`.

---

## Native primitives

### Root stdlib (`programs/stdlib.json`) — 9 primitives

```
native.READ    GET / EXISTS / KEYS / VALUES / SIZE
native.WRITE   SET / STORE / DELETE
native.EVAL    math, logic, flow, IO, meta, GPU, tensors
native.HASH    HASH / SHA256
native.CALL    external dispatch: sidecar://, sco://, arbitrary URI
native.PHASE   K'UHUL phase law — FOLD_ENTER / FOLD_EXIT / FOLD / PHASE / TRANSITION
native.STACK   call stack — PUSH / POP / PEEK / STACK_DEPTH
native.PROGRAM program loader — EXEC / LOAD_PROGRAM / JROM_EXEC / JROM_DECODE
native.FOLD    K'UHUL fold cycle in-process — Pop→Wo→Yax→Sek→Chen→Xul native C++
```

All 9 primitives are now present in the root `programs/stdlib.json`
(formerly only 5; PHASE/STACK/PROGRAM were dist-only; FOLD is new in this session).

### `native.FOLD` — in-process fold cycle (added 2026-08-18)

Routes to the `"native.FOLD"` C++ primitive in `src/xcfe.cpp`. No subprocess,
no CLR call in the current implementation — all six stages run as pure C++ data
transforms over `FoldCtx`.

| op | fn | description |
|---|---|---|
| `FOLD_CYCLE` | `cycle` | Full Pop→Wo→Yax→Sek→Chen→Xul, N rounds (default 2). Returns `{fold,node,result,residency,proof_trace,providers,cycle_identity,rounds,ok}`. |
| `FOLD_STAGE` | `stage` | Run one named stage. Accepts `@ctx` for FoldContext continuity. |
| `FOLD_PROOF` | `proof` | Phase 7.6 — SHA-256 Merkle root over `@trace` or `@ctx.proof_trace`. Writes `io\proof\<det_id>.proof.json`. Returns `{root,leaf_count,artifact,ok}`. |

`FOLD_CYCLE` was previously routed to `native.EXE` calling `FoldOrchestrator.exe`;
it now routes natively, removing the subprocess round-trip.

---

## Op families (stdlib.json)

| Family | Ops |
|---|---|
| **State** | GET SET STORE EXISTS DELETE KEYS VALUES SIZE |
| **Math** | ADD SUB MUL DIV MOD POW MIN MAX ABS FLOOR CEIL SQRT CLAMP |
| **Logic** | EQ NEQ GT LT GTE LTE AND OR NOT XOR |
| **Bitwise** | BIT_AND BIT_OR BIT_XOR BIT_SHL BIT_SHR |
| **Flow** | IF LOOP MAP FILTER REDUCE |
| **IO** | LOG PRINT |
| **Meta** | DEFINE_OP DELETE_OP LIST_OPS EVAL NEGOTIATE |
| **Hash** | HASH SHA256 |
| **Tensor / GPU** | TENSOR_ALLOC TENSOR_COPY TENSOR_REGISTER TENSOR_GET TENSOR_EXISTS TENSOR_REMOVE TENSOR_LIST MATMUL GEMM RELU SOFTMAX `GPU_DISPATCH` |
| **Phase (dist only)** | PHASE FOLD TRANSITION PHASE_MANIFOLD |
| **Stack (dist only)** | PUSH POP |
| **Program (dist only)** | EXEC |
| **External** | CALL LOAD_SIDECAR SCO_LOAD SCO_HASH |

`GPU_DISPATCH` → `native.EVAL @fn:dispatch` → `GPUComputeEngine::dispatchShader()`
→ `MicronautDispatch.cpp` D3D11 path when shader contains "micronaut".

---

## Programs inventory

### `manifest.json` — model manifest (default: `micronaut-coder`)

| id | type | runtime | notes |
|---|---|---|---|
| `scx-expert-8` | moe_gguf | json-runtime | 8-expert MoE |
| `micronaut-coder` | coder | native-cpp | parse/compile/generate/todos pipeline |
| `mini` | mini | json-runtime | GPT-2, HOT residency (128MB VRAM) |
| `dolphin` | gguf | llama-server | Dolphin 3B Q5_K_S, tool-caller |
| `gpt-oss` | moe | scxqdds | GPT-OSS 20B planner, DDS address space |

### `bootstrap.json` — UI runtime config
- Port `9000`, host `127.0.0.1`, API prefix `/api`
- Used by `micronaut-ui-chat-app.ps1`
- Default model: `micronaut-coder`

### `actions.manifest.json` — action → executable map

| domain | action | exe |
|---|---|---|
| coder | parse / generate | `dist/micronaut-coder/micronaut_coder_pipeline.exe` |
| coder | todos | `dist/micronaut-coder/code-micronaut/micronaut_coder_helpers.exe` |
| factory | create | `dist/micronaut-factory/test_runtime_flow.exe` |
| runtime | run | `dist/json-runtime/build/Release/json_runtime.exe` |
| runtime | health | `GET /api/health` |
| gpu | matvec | `scratch/cl_moe_matvec.exe` (OpenCL) |
| gpu | harness | `scratch/cl_harness.exe` (OpenCL 1.2) |

### `kuhul_dispatch.json` — stub (needs expansion)
Currently: LOG → SET result=$input → LOG. Placeholder for the K'UHUL XCFE dispatch
path. The intended full-cycle body:

```
PHASE → Pop           observe field
FOLD  → Wo            schedule candidates
FOLD  → Yax           branch
GPU_DISPATCH → Sek    micronaut field competition (physics-gated via cbuffer b0)
FOLD  → Chen          verify
FOLD  → Xul           emit result + update @state
```

### `fold_node.json` — Yax admission branch
Introduces `FOLD_ENTER` / `FOLD_EXIT` ops (not yet in root stdlib):

```json
FOLD_ENTER → Yax / admission-node
GT(candidate, threshold) → admitted
IF admitted → FOLD_EXIT(Sek, "admitted")
           → FOLD_EXIT(Xul, "rejected")
```

Maps directly to the W·C·R admission gate: `candidate` = `S_µ`, `threshold` = dynamic gate.

### `scxq2_tile_upload.json` — full GPU pipeline (10 steps)
Orchestrates SCXQ2 container → triple-buffer → iGPU INT4 attention. Nothing JSON
touches the GPU — json_runtime.exe calls `sidecar://scxq2_gpu/*` which drives
D3D11 compute shaders:

| step | label | op | GPU? |
|---|---|---|---|
| 1 | init_gpu | `sidecar://scxq2_gpu/INIT_DEVICE` | sets up D3D11 device |
| 2 | resolve_hash | `RESOLVE_HASH` | SHA-256 shard lookup |
| 3 | request_shard | `REQUEST_SHARD` | triple-buffer: disk→staging→upload |
| 4 | wait_ready | `POLL_READY` | retry ≤100×, interval 1ms |
| 5 | decode_int4 | `DECODE_INT4` | `scxq2_int4_lane.hlsl` |
| 6 | run_attention | `RUN_ATTENTION` | `tile_qk_softmax.hlsl` + `tile_v_mul.hlsl` |
| 7 | readback | `READBACK` | CPU float[65536] |
| 8 | advance_loader | `ADVANCE` | rotate triple-buffer |
| 9 | advance_head | `native.EVAL` | head/layer counter |
| 10 | loop_or_done | `native.EVAL` | goto 2 or halt |

Never on GPU: JSON, BSON, hex strings, SCXQ2 metadata.

### `main.json` (dist copy) — smoke test
ADD/MUL/SUB/SQRT/GT/IF/DEFINE_OP/LOG demo. Canonical json_runtime.exe smoke run.

### `stdlib.json` — base op table
Not executable. Loaded by json_runtime.exe at start. Programs extend with inline `@ops`.

---

## GPU_DISPATCH ↔ Micronaut physics (wired 2026-08-16)

After wiring `MicronautDispatch.cpp`, `GPU_DISPATCH` on `micronaut.cso` now:

1. `Micronaut::execute()` calls `gpu_->setPhysics(PhysicsState::fromField(*field))`
2. `GPUComputeEngine::dispatchShader()` routes `micronaut.*` → `dispatchMicronautD3D11()`
3. `dispatchMicronautD3D11()` maps `PhysicsCBuffer` into `cbuffer(b0)`:
   - `gravity_gate` — lowers admission threshold when > 1 (gravity well active)
   - `entropy` — raises threshold + dampens W_eff (chaotic field)
   - `attention` — lowers threshold slightly (focused context)
   - `pressure` — amplifies W_eff (sharpens competence signal)
4. `micronaut.cso` computes `W_eff * C * R` with physics-scaled W and dynamic gate

A program can trigger this with:
```json
{ "@op": "GPU_DISPATCH", "@args": ["micronaut.cso"], "@out": "dominant_micronaut" }
```

`GPU_DISPATCH micronaut.cso` targets the **shared K-Cube** (`[6, 1024, 1024, 4]`) — not
per-micronaut tensors. Every micronaut is a node (x,y) in the FieldGraph lattice; the dispatch
evolves all 1024×1024 nodes on all 6 faces simultaneously. `dominant_micronaut` = `argmax` of
SH coefficients projected from that shared field. See `K-CUBE.md`.

---

## stdlib/ — K'UHUL semantic library

`stdlib/*.kuhul` — the K'UHUL semantic standard library. Each module is
structured as a fold-phase cycle (Pop=bind, Wo=bind/config, Yax=resolve,
Sek=dispatch, Ch'en=collect_status, Xul=commit). The fold cycle is the module's
own compilation unit — not just the runtime loop.

Compile: `python tools/khlc.py --provider <name> stdlib/<name>.kuhul` → `stdlib/<name>.kson`
Validate: `python tools/kson_validate.py stdlib/<name>.kson`

| Module | Depends on | Contents |
|--------|-----------|----------|
| `core` | — | folds, lanes, types, glyphs, opcodes, contracts, providers |
| `constants` | core | π, e, ϕ, physical constants |
| `functions` | core | higher-order: map, filter, reduce, compose |
| `vectors` | core | vec2/vec3/vec4, dot, cross, normalize |
| `matrices` | vectors | mat2/3/4, inverse, transpose, determinant |
| `tensors` | matrices | tensor_0d–4d, reshape, dtype, strides |
| `geometry` | vectors | plane, sphere, AABB, ray, intersection |
| `gravity` | vectors | gravity_gate formula, arc weights, geodesic flow |
| `pi` | constants | K'UHUL π-phase math: θ=0..5, fold angles |
| `hlsl` | core | HLSL cs_5_0 dispatch helpers, cbuffer layouts |
| `glsl` | core | GLSL ES 300 / WebGL2 kernel helpers |
| `semantic_cube` | tensors, hlsl | cube face ops, SH projection, face energy |
| `audio` | core | audio node dispatch (web audio) |
| `colors` | core | HSL/RGB/hex conversion, theme tokens |
| `image` | core | image load/decode/encode ops |
| `time` | core | timestamp, delta-t, tick counter |
| `random` | core | PRNG, Gaussian noise, uniform sample |
| `statistics` | tensors | mean, variance, softmax, argmax |
| `fibonacci` | core | Fibonacci sequence, golden ratio spiral |

**Two stdlib layers — not the same:**
- `programs/stdlib.json` = json_runtime **primitive table** (9 `native.*` families — the C++ execution surface)
- `stdlib/*.kuhul` = K'UHUL **semantic library** (20 modules — vocabulary compiled to KAST JSON)

json_runtime loads `programs/stdlib.json` at startup. Programs that need the K'UHUL semantic vocabulary `include` or reference the compiled `.kson` files.

---

## WebGL2 / browser graphics programs

For PWA apps and games, the GPU compute path runs in the browser via WebGL2.
These programs in `programs/` drive that path:

| Program | Backend | Purpose |
|---------|---------|---------|
| `webgl2.fieldgraph.json` | webgl2 | 100d field graph rendered as fragment shader; transport: WebView2 postMessage or WASM |
| `webgl2.kernel.json` | webgl2 | General WebGL2 compute kernel dispatch |
| `webgl2.fieldgraph.json` `@shader` | `drivers/webgl2/fieldgraph_100d.frag` | Fragment shader — field graph 100d float16 texture |
| `opencl.geodesics.json` | opencl | Geodesic flow on OpenCL 1.2 |
| `opencl.kernel.json` | opencl | OpenCL kernel dispatch |

`webgl2.fieldgraph.json` is the canonical pattern for PWA app GPU rendering:
- Source: `.s7` file (100-dim field graph AST)
- Transport: WebView2 postMessage → JS → WebGL2 texture upload → fragment shader
- Operations: `upload_texture`, `lookup_record`, `project_scalar`
- Apps import this via `@bridge: "drivers/webgl2/fieldgraph_100d.mjs"`

---

## programs/micronauts/ — program-layer micronaut adapters

Distinct from `micronauts/` (instance definitions). This subdirectory holds
runtime adapters, cluster node specs, and JROM adapters used by programs:

| File | Role |
|------|------|
| `control-micronaut-1.json` | CM-1 brain profile — coordinator for all phase micronauts |
| `cluster_node.node.json` | Cluster node spec for distributed execution |
| `cm1-runtime.mjs` | CM-1 runtime bridge (JS) |
| `jrom-research-gpt2-q8.json` | JROM adapter: research micronaut → GPT-2 Q8 |
| `jrom-template-adapter.json` | JROM adapter template |
| `mx2-brain-jrom-adapter.json` | MX-2 IDB → JROM bridge |
| `neural_pipeline.pipeline.json` | Neural pipeline spec |
| `MICRONAUTS.ebnf` | Micronaut grammar (EBNF) |

---

## Supernaut build artifacts

`kuhul_supernaut` is the fold cycle at the inference layer. Built as a static
lib by `dist/json-runtime/CMakeLists.txt`:

```
kuhul_supernaut.lib       supernaut + DaemonSession + S7Mini + Merkle proof chain
supernaut.exe             Numatic Manifold inference entry point
scxq2_packer_standalone   SCXQ2 packer — inline SHA-256, no deps, always builds
scxq2_packer              SCXQ2 packer — OpenSSL SHA-256 (conditional, requires OpenSSL)
```

Key sources added in this session:

| file | phase | role |
|---|---|---|
| `supernaut/s7l/merkle.cpp` | 7.6 | Self-contained SHA-256 Merkle tree |
| `supernaut/s7l/merkle.h` | 7.6 | Public API: `merkle_root`, `verify_merkle_root`, `merkle_proof` |
| `supernaut/src/clr_fold_host.cpp` | 7.8 | .NET 8 in-process CLR host for fold DLLs via hostfxr |

---

## Key gaps

| Gap | File | Status | Impact |
|---|---|---|---|
| `FOLD_ENTER`/`FOLD_EXIT` not in stdlib | `stdlib.json` | **CLOSED** — now under `native.PHASE` | `fold_node.json` now runnable |
| Root stdlib missing STACK/PROGRAM/PHASE | `stdlib.json` | **CLOSED** — all 9 primitives present | — |
| `kuhul_dispatch.json` stub | `kuhul_dispatch.json` | **CLOSED** — full 6-phase program: Pop(observe)→Wo(schedule_candidates)→Yax(select_best)→Sek(GPU_DISPATCH micronaut.cso)→Chen(verify_result)→Xul(emit) | `schedule_candidates`/`select_best`/`execute_selected`/`verify_result` still need C++ impl |
| `sidecar://scxq2_gpu/*` not yet a real sidecar | — | Open | `scxq2_tile_upload.json` can't run end-to-end |
| Phase 7.6 proof artifacts to io/proof/ | `xcfe.cpp fn=proof` | **CLOSED** | `io\proof\<det_id>.proof.json` written on every FOLD_PROOF call; deterministic_id = first 16 chars of Merkle root |
| Phase 7.8 CLR fold host gap | `clr_fold_host.cpp` | **CLOSED** — `FoldJsonBridge.cs` added to Contract; each stage has `[UnmanagedCallersOnly] FoldEntry`. C++ resolves via `"Kuhul.Folds.XxxStage, Xxx"` type name + `"Kuhul.Folds.FoldEntryDelegate, Kuhul.Folds.Contract"` delegate. | — |
| MX-2 IDB W update | `FoldOrchestrator.cs` | **CLOSED** — `Mx2Bridge.Persist()` parses `Xul:mx2_record` lines after each fold cycle and appends JSON records to `mx2_W_updates.jsonl`. idb_memory.py reads this file to evolve W across folds. | idb_memory.py W-update reader not yet wired |
| SCXQDDS shard hashes not fed to Merkle | `merkle.cpp` | Open | Shard integrity not in proof chain yet |

---

## Related
- `dist/v3.5.0-WebX/native/MicronautDispatch.cpp` — D3D11 cbuffer + dispatch
- `dist/v3.5.0-WebX/shaders/micronaut.hlsl` — W·C·R physics-gated shader
- `dist/v3.5.0-WebX/shaders/cs5/micronaut.cso` — compiled cs_5_0 binary
- `dist/v3.5.0-WebX/native/webx_compute.h` — PhysicsState, GPUComputeEngine, setPhysics()
- MICRONAUTS.md — W·C·R scoring law, MoMU field, coder micronaut family
- SEMANTIC_ENGINE.md — gravity_gate formula, arc weights, physics scalars source; shared K-Cube field architecture
- K-CUBE.md — shared field tensor `[6,1024,1024,4]`; GPU_DISPATCH targets this field; SH projection → dominant_micronaut
- SCXQ2.md / SCX.md — SCXQ2 container format, shard addressing
