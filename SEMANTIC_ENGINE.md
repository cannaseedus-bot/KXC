# SEMANTIC_ENGINE.md — ASX Runtime / FieldExecutionEngine (v0.7)

> Location: `C:\Users\canna\_khanary_inspect\desktop\semantic_engine\`
> See also: GPU.md, SCXQ2.md, SCX.md

---

## What it is

The `semantic_engine` is the **ASX Runtime Loop** — a C++ execution engine that sits above XCFE/SCXQ2. It implements the K'UHUL physics laws, MoE expert routing, geodesic flow computation, and shared memory telemetry. It uses D3D12 directly (not via DirectML high-level API) for tiled shard residency.

It is separate from json_runtime / XCFE. The two engines are connected via HTTP (kuhul_engine endpoint) and shared memory (`Local\KuhulGeometricState`).

---

## FieldExecutionEngine — runtime components

```
FieldExecutionEngine
  ├── DdsShardLoader       — D3D12 tiled residency: DDS shards + XShard specialists
  ├── EvolutionBot         — fitness-based weight mutation loop
  ├── ReplayLaneManager    — execution trace recording (V6 replay determinism)
  ├── LegalityVerifier     — mutation legality check (Law E)
  ├── ProjectionExpert     — project tensor state → CSS + shared memory
  ├── SharedMemoryBridge   — "Local\KuhulGeometricState" cross-process telemetry
  ├── GeodesicFlowLayer    — 1024-dim manifold flow computation
  ├── MetricNormalization  — metric tensor normalization (1024-dim)
  └── MoE Experts
        ├── AgentCoder   (CODING specialization)
        └── AgentFactory (FACTORY specialization)

D3D12 Hardware Context:
  ID3D12Device, ID3D12CommandQueue (compute), ID3D12GraphicsCommandList,
  ID3D12Fence + HANDLE fence_event, fence_value counter
```

---

## Runtime physics state (per engine instance)

These are the live physics scalars that drive gravity wells, arc weights, and routing:

```cpp
m_current_entropy    = 0.14f   // thermodynamic entropy of the field
m_current_attention  = 0.72f   // attention pressure (how focused)
m_current_pressure   = 0.34f   // manifold pressure
m_current_gravity    = 9.80665f // Earth gravity constant (base) — scaled per tick
m_current_arc_weights[1024]     // π-nary arc weights, init 1/√1024
m_current_metric_tensor[1024]   // manifold metric tensor, init 0.1
```

---

## Gravity well formula (per tick)

```
gravity_gate = clamp(
    1.0 + 0.35·pressure - 0.25·entropy + 0.15·attention + 0.10·affinity,
    0.1, 4.0
)
gravity = 9.80665 × gravity_gate
```

**This is the gravity well** — each fold's replay affinity (`affinity`) pulls the gravity gate up. Higher pressure → more gravity. Higher entropy → less gravity. The bone cluster's attention bias in `cs_fold_kernel_compute_.hlsl` (`gravity_scale * 2.0 * overlap`) is the GPU-side mirror of this same gravity concept.

---

## Arc weight formula (per tick)

```
arc_bias[i] = 1.0 + 0.10·attention - 0.08·entropy + 0.06·pressure + 0.04·affinity
arc_weight[i] = clamp((1/√1024) × arc_bias[i], 0.01, 2.0)
```

The arc weights are the π-nary weights that scale the 1024-dim metric tensor during geodesic flow. These propagate into `m_flow_layer->set_arc_weights(arc_weights)` → `GeodesicFlowLayer::flow(metric_tensor, velocity)`.

---

## Execution cycle (one tick = `run_end_to_end_step`)

```
TICK N

1. PERCEPTION / LAW B  → get_replay_affinity(fold_id)
2. LAW C  (ROUTING)    → update_fold_residency(fold_id, true)  — D3D12 tiled page-in
3. COMPUTE             → MoE routing:
     query contains "code"/"refactor"  → AgentCoder.execute_task("refactor", query)
     query contains "create"/"new"     → AgentFactory.execute_task("instantiate", query)
     default                           → call_gguf_inference(query)  → HTTP or GGUF_ENDPOINT
4. GRAVITY GATE        → compute gravity + arc_weights from physics state
5. GEODESIC FLOW       → flow_layer.flow(metric_tensor, velocity)
     velocity[i] = 0.001 × (attention - entropy) × (1 + i%7)
6. REPLAY/EVAL         → evolver.update_fitness(0, fitness)
7. META / LAW E        → evolver.evolve_step(metric_tensor)
                        → verifier.verify_mutation(delta, entropy)
                           LAWFUL  → replay_manager.record_trace(trace)
                           REJECT  → log warning
8. PROJECTION          → projection_expert.project_to_css(state)
                        → shm.update_state(tick, fold_id, entropy, attention, pressure)
```

**MoE fitness scores**:
- Specialist activated: `fitness = 0.941`
- General (GGUF call): `fitness = 0.30 + 0.69 × (hash(response) % 1000) / 1000`

The **kuhul-folds runtime** (`dist/kuhul_folds/`) implements a more precise reward signal
at the **Chen** stage: `reward = projectionScore + arcScore`, where `projectionScore = 0.6`
for a Strong-admitted result and `arcScore = min(0.4, ArcState × 0.4)` — both are honest
in-fold observables, not heuristic constants. C (per-execution confidence) is updated there
immediately; W (competence) is evolved by MX-2 IDB across folds. See `KUHUL.md §kuhul-folds`.

---

## GGUF inference bridge

When no MoE specialist matches, `call_gguf_inference(query)` POSTs to the model server:

```
Endpoint: GGUF_ENDPOINT env var  OR  http://127.0.0.1:5000/v1/chat/completions
Method:   curl -s -X POST (spawned via _popen)
Payload:  {"messages":[{"role":"user","content":"..."}],"max_tokens":24,"temperature":0.7}
```

Response is hashed → fitness score. Empty response → 0.5 fitness, warning logged.

**Note**: port 5000 is the GGUF fallback default. kuhul_engine runs on port 17480.

---

## Manifest loading (`load_manifest`)

JSON manifest format:

```json
{
  "base_model": {
    "shards": [
      { "path": "...", "size": 1234567, "fold": "3" }
    ]
  },
  "shards": [
    { "id": 7, "path": "expert_7.xshard", "name": "CodeExpert" }
  ],
  "experts": [...]
}
```

- `base_model.shards` → `DdsShardLoader.register_shard(fold_id, DdsShardMetadata)`
  - `DXGI_FORMAT_R32_FLOAT`, width=4096, height computed from size
  - `initialize_reserved_resource(total_size)` — D3D12 reserved resource for full model
- `shards` with `.xshard` extension → `register_xshard(id, XShardMetadata)` — specialist shards

---

## IR (Intermediate Representation)

From `ir.h`:

```cpp
enum class XCFEPhase : uint8_t { Pop=0, Wo, Sek, Chen, Xul, Unknown };

enum class OpCode : uint16_t {
    NOP = 0, ADD_I64 = 1, MUL_I64 = 2,
    FUSED_ADD_MUL_I64 = 3,
    BIMODAL_ATTENTION = 4,   // GPU-targeted, CPU fallback in BatchExecutor
    GEODESIC_FLOW = 5        // GPU-targeted, CPU fallback in BatchExecutor
};

struct IRNode {
    XCFEPhase phase;
    OpCode opcode;
    uint32_t a_idx, b_idx, c_idx, out_idx;  // indices into Buffers
};

struct Buffers {
    std::vector<int64_t> i64;
    std::vector<double>  f64;
    std::vector<float>   f32;  // for matmul / FP32 tensors
};
```

`BatchMap` groups `IRNode*` by `OpCode` for batched execution. `BatchExecutor::executePhase` runs batches in deterministic numeric OpCode order. `BIMODAL_ATTENTION` and `GEODESIC_FLOW` log a trace and emit a deterministic CPU fallback — GPU dispatch path is via `dx12_executor.h`.

**Phase note**: `XCFEPhase` here has 5 phases (Pop, Wo, Sek, Chen, Xul) — `Yax` is absent. The xcfe.cpp phase system has 6 (Pop, Wo, Yax, Sek, Ch'en, Xul). Discrepancy to be aware of when bridging the two systems.

---

## DX12 executor (`dx12_executor.h`)

```cpp
bool executePlanDX12(ExecutionPlan &plan, CompiledProgram &prog,
                     const std::string &schedulePath, bool enableGpu=false);
```

- Executes a compiled plan on D3D12
- Outputs a JSON dispatch schedule to `schedulePath` (for external runner)
- `enableGpu=false` by default — safe-mode: generate schedule only

`DispatchParams` packs per-batch arrays (A, B, C for fused, OUT) for D3D12 dispatch.

---

## Shared memory telemetry

`SharedMemoryBridge` opens/creates named shared memory at `Local\KuhulGeometricState`. Updated every tick:

```cpp
shm.update_state(tick, fold_id, entropy, attention, pressure);
```

Any process that opens `Local\KuhulGeometricState` gets live physics state. PRIMEOS and kuhul_engine can both read this for real-time monitoring.

---

## Shared field architecture — one cube, N micronauts

There is **one semantic cube** (`[6, 1024, 1024, 4]` float32, D3D11 `RWTexture2DArray`) shared across the entire brain. Micronauts are **nodes (x, y) within the FieldGraph lattice** — not owners of separate cubes.

```
K-CUBE  [6, 1024, 1024, 4]
         │   └──────────┘
         │   FieldGraph lattice — all micronauts coexist as (x,y) node addresses
         │
         └── 6 faces: Phi (Pop) · Fold (Wo) · Gram (Yax) · Geodesic (Sek)
                      Projection (Chen) · Entropy (Xul)
```

`SemanticCubeShader` dispatches `Dispatch(64, 64, 6)` and evolves **every node on every face simultaneously** as a coupled PDE system. The Gram face evolves all candidate micronaut nodes in parallel — `gram_coupling` drifts each node's score toward the current fold gate in a single threadgroup pass, no per-micronaut loop.

```
SemanticCubeShader
  [numthreads(16,16,1)] × Dispatch(64, 64, 6)
  id.z = face     id.x, id.y = FieldGraph node (x, y)
  → all 6·1024·1024 = 6,291,456 nodes evolved in one dispatch
```

SH projection then compresses the result to a winner:

```
1,048,576 nodes per face
    ↓  DirectX::SHProjectCubeMap (order 2–6, max 36 coefficients)
    ↓  argmax over SH_R coefficients
    =  dominant micronaut node   ← O(1) regardless of micronaut count
```

The `FoldContext.MuField` W·C·R triples in the kuhul-folds C# runtime are **discrete readbacks** from this field — projections of the cube geometry into scalars, not independent tensors. The cube is the geometry; W·C·R is the readback. `readFaceEnergy(ctx, out_energy[6])` is the cheap staging path (1×1 center texel per face) that feeds those scalars.

The MoE routing in `run_end_to_end_step` (keyword match → AgentCoder/AgentFactory/GGUF) is the **CPU-side approximation** of this field resolution — a string-match shortcut used when the full D3D11 dispatch path is not active. The canonical resolution is the GPU field evolution + SH projection.

---

## include/ directory inventory

| Header | Role |
|--------|------|
| `ir.h` | IRNode, OpCode, Buffers, LaneBatch, BatchExecutor |
| `executor.h` | ExecutionNode, OpRegistry, ThreadPool, ParallelXCFEWalker, BatchExecutor |
| `field_execution_engine.h` | FieldExecutionEngine — complete ASX runtime loop (all includes) |
| `dx12_executor.h` | executePlanDX12 — D3D12 dispatch + JSON schedule output |
| `dx12_device_factory.h` | DeviceContext factory: device, compute_queue, copy_queue, fence |
| `compiler.h` | CompiledProgram — compiles ExecutionNode tree to batched IRNodes |
| `planner.h` | ExecutionPlan — scheduling structure |
| `dds_shard_loader.h` | DdsShardLoader — D3D12 tiled residency for DDS + XShard |
| `geodesic_flow_layer.h` | GeodesicFlowLayer — 1024-dim manifold flow |
| `metric_normalization.h` | MetricNormalization — arc-weighted metric normalization |
| `domain_expert.h` | DomainExpert base + ExpertSpecialization enum |
| `projection_expert.h` | ProjectionExpert — project_to_css(ProjectionState) |
| `evolution_bot.h` | EvolutionBot — fitness tracking + evolve_step + get_best_adapter |
| `replay_lane_manager.h` | ReplayLaneManager — trace recording + replay_affinity |
| `legality_verifier.h` | LegalityVerifier — verify_mutation → LegalityVerdict::LAWFUL/REJECT |
| `memory.h` | Memory management helpers |
| `memory_compressor.h` | Memory compression (likely for SCXQDDS-style INT8 packing) |
| `kv_cache.h` | KV cache management |
| `shared_memory_bridge.h` | SharedMemoryBridge — Local\KuhulGeometricState |
| `simd_kernels.h` | SIMD math kernels (used by BatchExecutor ADD_I64 lane) |
| `tokenizer.h` | Tokenizer |
| `semantic_kernel.h` | Semantic kernel ops |
| `semantic_reader.h` | Semantic field reader |
| `knobs.h` | Runtime tuning knobs |
| `quant.h` | Quantization helpers |
| `scheduler.h` | Execution scheduler |
| `runtime_loop.h` | Main runtime loop integration |
| `scx/io_helpers.hpp` | SCX I/O helpers — connects semantic_engine to SCX ecosystem |
| `nlohmann/` | JSON library (embedded) |

---

## Connection to the rest of the stack

```
PRIMEOS (desktop app)
  │
  ├── FieldExecutionEngine
  │     ├── D3D12 device  (owns compute + copy queues)
  │     ├── DdsShardLoader  (tiled model residency)
  │     ├── GeodesicFlowLayer  (1024-dim manifold)
  │     ├── SharedMemoryBridge → "Local\KuhulGeometricState"
  │     └── call_gguf_inference() → HTTP → kuhul_engine:17480
  │
  ├── XCFE / json_runtime (port 8787)  [separate process]
  │     └── SCXQ2 ops → DirectML dml_gemm.dll
  │
  └── kuhul_engine (port 17480)  [separate process]
        └── llama-server / GPT-OSS model
```

The gravity well scalars (entropy, attention, pressure, gravity) computed by `FieldExecutionEngine` are the same physics quantities that:
- Drive `cs_fold_kernel_compute_.hlsl` (`gravity_scale * 2.0 * overlap`)
- Drive XVM manifold opcodes (`PRESSURE_PROPAGATE`, `ENTROPY_GRADIENT`)
- Set the KuhulPhysics antigravity scale in the GPT-2 trainer

`SharedMemoryBridge` at `Local\KuhulGeometricState` is how these values flow between processes without serialization overhead.

---

## Related docs

- `K-CUBE.md` — full semantic cube reference: face layout, HLSL evolution physics, SH projection, shared field architecture, SemanticCubeMap API, file map
- `KUHUL.md §kuhul-folds` — C# IFoldStage runtime; three-outcome Yax; Chen reward; W·C·R readback from field
- `FOLDS.md §8` — micronauts as FieldGraph nodes; W·C·R=QKV; three-outcome admission
- `NNC-K.md` — hardware-independent semantic runtime; driver binder
- `GPU.md` — provider inventory; D3D11/D3D12 chain; KLSL
