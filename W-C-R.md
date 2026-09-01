# W-C-R.md — Micronaut Admission Scoring

The **W·C·R** triple is the admission formula for the Moµ (Mixture of Micronauts)
field. It decides which micronaut may act in a given fold round.

```
S_µ = W_µ · C_µ · R_µ
```

| component | meaning | range | default |
|---|---|---|---|
| **W** | Weight / competence | 0–1 | 0.5 |
| **C** | Confidence | 0–1 | 0.5 |
| **R** | Relevance | 0–1 | 0.5 |
| **S** | Admission score | 0–1 | 0.125 |

A micronaut is **admitted** when `S ≥ threshold` (default 0.5). The micronaut
with the highest `S` is the **dominant** and executes in Sek.

---

## Types (C++ — `src/xcfe.cpp` `native.FOLD` primitive)

```cpp
struct MuScore {
    string name;
    double W = 0.5;   // weight / competence in this context
    double C = 0.5;   // model or system confidence in the candidate
    double R = 0.5;   // relevance to current task/fold
    double score() const { return W * C * R; }
};

struct FoldCtx {
    string fold;                    // current phase (Pop/Wo/Yax/Sek/Chen/Xul)
    string node_name;               // active node label
    string result;                  // accumulated output
    string residency;               // "WARM" | "COLD"
    vector<MuScore> mu_field;       // the live field of candidates
    vector<string> proof_trace;     // per-stage SHA-256 hashes → Merkle root
    vector<string> provider_req;    // GPU/D3D11 requirements declared in Wo
};
```

`mu_field` is the **µ-field** — the semantic field of candidate micronauts
scored against the current task. It is seeded in **Pop** and resolved in **Yax**.

---

## Fold-cycle integration

### Pop — seed the µ-field

Pop reads the current context and populates `mu_field` with candidate
`MuScore` entries. Each entry gets initial W/C/R values based on the task:

```
Pop observes → produces mu_field candidates
         ↓
         FoldCtx.mu_field = [ {coder, W=0.8, C=0.85, R=0.9}, ... ]
```

### Wo — schedule; declare provider requirements

Wo reads the µ-field and appends provider requirements to `provider_req`
(e.g. `"WARM"`, `"D3D11"`). It does not alter W/C/R directly.

### Yax — admission gate (dominant selection)

Yax iterates `mu_field`, computes `score()` for each entry, finds the dominant
micronaut (highest score), and writes it to `FoldCtx.node_name`:

```cpp
auto dom = max_element(mu_field, [](auto& a, auto& b){ return a.score() < b.score(); });
ctx.node_name = dom->name;
ctx.result += "[Yax] admitted: " + dom->name + " S=" + to_string(dom->score());
```

Any micronaut with `S ≥ 0.5` is admitted; the dominant one is selected for Sek.
This mirrors the GPU admission in `micronaut.hlsl` (see below).

### Sek — dispatch to dominant micronaut

Sek calls the admitted micronaut and stores the result in `FoldCtx.result`.
Appends a SHA-256 hash of the result to `proof_trace`.

### Chen — observe and record

Chen computes the observation hash, appends it to `proof_trace`, and verifies
the result is non-empty.

### Xul — commit; residency decision

Xul reads `provider_req`. If any provider required `"WARM"`, the result stays
resident (WARM). Otherwise it transitions to COLD. Appends a final hash to
`proof_trace`.

---

## Cycle identity

```
cycle_identity = SHA256( SHA256("Pop") || SHA256("Wo") || SHA256("Yax") ||
                          SHA256("Sek") || SHA256("Chen") || SHA256("Xul") )
```

Uppercase hex string. Deterministic and reproducible without the fold DLL files.
Mirrors `FoldOrchestrator.CycleIdentity` name-bytes fallback path.

---

## GPU shader — `micronaut.hlsl` / `micronaut.cso`

The GPU path implements the same formula in HLSL (cs_5_0):

```hlsl
cbuffer PhysicsCBuffer : register(b0) {
    float gravity_gate;   // lowers admission threshold when > 1
    float entropy;        // raises threshold + dampens W_eff
    float attention;      // lowers threshold slightly
    float pressure;       // amplifies W_eff (sharpens competence)
};

float W_eff = W * (1.0 + pressure * 0.1) / (1.0 + entropy * 0.5);
float gate  = 0.5 * (1.0 - attention * 0.1) * (gravity_gate > 1.0 ? 0.8 : 1.0);
float S     = W_eff * C * R;
if (S >= gate) admitted++;
// workgroup reduction → dominant index (highest S)
```

Physics scalars are written via `PhysicsState::fromField()` →
`GPUComputeEngine::setPhysics()` → `dispatchMicronautD3D11()` (MicronautDispatch.cpp).

### JSON program trigger

```json
{ "@op": "GPU_DISPATCH", "@args": ["micronaut.cso"], "@out": "dominant_micronaut" }
```

---

## Physics scalars (SemanticEngine / ASX)

These come from the live ASX physics state (shared memory `Local\KuhulGeometricState`):

| scalar | effect on admission |
|---|---|
| `gravity_gate > 1` | Lowers threshold → easier admission (gravity well active) |
| `entropy ↑` | Raises threshold + dampens `W_eff` → harder admission (chaotic field) |
| `attention ↑` | Lowers threshold slightly → focused context aids admission |
| `pressure ↑` | Amplifies `W_eff` → sharpens the competence signal |

The CPU path (native.FOLD `fn=cycle`) does not read live physics; it uses the
default W/C/R values seeded in Pop. The GPU path (`GPU_DISPATCH` on
`micronaut.cso`) reads the live cbuffer.

---

## Supernaut connection

`NumaticManifoldEngine::resolve_geodesic` (6 L-layers) is the Supernaut realization
of the same W·C·R admission at model inference time:

```
L1 (Pop)  → embed input; seed bimodal manifold A_flat ⊗ A_curved
L2 (Wo)   → schedule geodesic candidates from SemanticGenomes
L3 (Yax)  → select dominant genome by manifold score (W·C·R analogue)
L4 (Sek)  → SIMD MATMUL via DirectXMath; run S7Mini inference
L5 (Chen) → observe result; update DaemonSession.replay_lanes
L6 (Xul)  → commit to session_hash; emit token
```

`DaemonSession.replay_lanes` → `proof_trace`  
`DaemonSession.session_hash` → `cycle_identity`

---

## Phase 7.6 — Merkle proof chain

`FOLD_PROOF` op (`native.FOLD fn=proof`) computes the SHA-256 Merkle root over
`proof_trace` entries. Leaf layout:

```
leaf[0] = SHA256("Sek result text")    ← from Chen stage
leaf[1] = SHA256("Chen observation")   ← from Chen stage
leaf[2] = SHA256("Xul commit")         ← from Xul stage
...
root = merkle_root_from_strings(proof_trace)
```

Binary reduction: odd layers duplicate the last node (standard / Bitcoin-style).
`verify_merkle_root(lane_hashes, expected_root)` recomputes and compares.
`merkle_proof(leaves, leaf_idx)` builds an inclusion proof path.

Implementation: `supernaut/s7l/merkle.cpp` (self-contained SHA-256, no deps).

---

## Symmetry table

| Layer | W·C·R concept | Moµ | Supernaut | GPU shader |
|---|---|---|---|---|
| Seed | Candidate field | `mu_field` | `SemanticGenome` list | W/C/R SSBO |
| Score | `S = W·C·R` | `MuScore::score()` | Manifold geodesic score | `W_eff * C * R` |
| Gate | `S ≥ threshold` | `S ≥ 0.5` (default) | L3 dominant select | `S >= gate` |
| Winner | Dominant micronaut | `FoldCtx.node_name` | Selected `SemanticGenome` | `dominant_idx` |
| Proof | Per-round hash chain | `proof_trace` | `replay_lanes` | — |
| Identity | Round fingerprint | `cycle_identity` | `session_hash` | — |

---

## Related docs
- `MICRONAUTS.md` — Moµ architecture, FoldContext, CLR fold host, registry.
- `PROGRAMS.md` — `native.FOLD` primitive, `FOLD_CYCLE`/`FOLD_STAGE`/`FOLD_PROOF` ops.
- `FOLDS.md` — Fold/Node phase law, SCXQDDS.
- `GPU.md` — DirectML + D3D11 provider inventory.
- `SEMANTIC_ENGINE.md` — gravity_gate formula, arc weights, physics scalars source.
