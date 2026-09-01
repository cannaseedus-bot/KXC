# GEO-WEIGHTS — Geometric Weight + DDS Tile Streaming

The model's weights don't need to live entirely in memory. They stream as
**DDS tiles** through **SCXQ2** (a reversible encoding), and the runtime keeps
only a small **HOT residency** for the active model. The shape of that streaming
is governed by a **geometric weight** — a structural descriptor that is
analogous in both the DirectX skin-mesh grammar and the K'UHUL KSON grammar.

```
DDS tiles (all weights, up to 100 GB total, SCXQ2-encoded)
   ↓ stream via SCXQ2 (FP16 canonical reversible; INT4 projected)
runtime keeps ~10 tiles HOT (resident) for the active model
   ↓ structural frame (the "geometric weight")
KSON driver (scxq2 req, Sek/Ch'en/Xul phase hooks) + X-skin geometric adapter
```

---

## 1. SCXQ2 — reversible streaming transport

| Encoding | Round-trip error | Domain |
|---|---|---|
| **FP16/BF16** | `~2e-04` (fp16 eps); half→half identity holds | **canonical reversible** — `SCXQ2₀ ≡ SCXQ2₁` ✅ |
| **INT4** | ~3% rel | collapsed/projected (lossy by design) |

Verified on real weights. SCXQ2 is **model-size agnostic** — any tensor or LoRA
A/B adapter streams through it losslessly at FP16. Tooling:
`tools/scxq2_adapter.py` (encode/decode/verify) and `tools/scxq2_tiles.py`
(DDS tile residency + streaming demo).

## 2. DDS tile residency

- The full model's tensors are split into **DDS tiles** (one per tensor, or
  grouped). Total could be 100 GB across all tiles.
- The runtime keeps **~10 tiles HOT** (resident) — the active model's working
  set. The rest are **STREAMING** (on disk, SCXQ2-encoded).
- A cold tile **streams in on demand** and decodes back reversibly (SCXQ2 LAW).

## 3. The geometric weight — two grammars, one frame

The structural descriptor that shapes streaming exists in two equivalent forms:

| `smgm16_base.adapter` (DirectX X skin) | `driver_v2.kson` (K'UHUL KAST) |
|---|---|
| `XSkinMeshHeader` (bones, skin weights) | `@driver` (provider, capabilities, hash) |
| `SkinWeights{…}` × 36 — per-bone geometric weights | `@requires.scxq2 >= 2.0` — streaming transport |
| mesh geometry = the frame data attaches to | `@phase_hooks`: Sek=dispatch, Ch'en=collect, Xul=**commit_tensor_state** |
| 1 Frame / bone hierarchy | `@admission` resource limits (memory, dispatch, workgroup) |

The KSON is the **runtime streaming contract**:
- `@requires.scxq2` declares the transport.
- `@phase_hooks` are the streaming lifecycle:
  - **Sek → dispatch** (send a tile to compute)
  - **Ch'en → collect_status** (read result back)
  - **Xul → commit_tensor_state** (commit the streamed/updated weight)

So the **geometric weight = the KSON driver contract + the geometric frame**:
the KSON tells the runtime *how* to stream (phases, resources, scxq2); the
geometric adapter provides the *structural weights* that the streamed DDS tiles
attach to.

---

## 4. Mini tile pool (worked example)

`tools/scxq2_tiles.py` splits the mini GPT-2 (`gpt2_small_lite_tool`, 148
tensors) into tiles, keeps the first `--hot` resident, streams the rest from
DDS via SCXQ2, and verifies a streamed-in cold tile round-trips (SCXQ2₀ ≡ SCXQ2₁).

```bash
python tools/scxq2_tiles.py \
  E:/models/GPT2/mini-GPT/gpt2_small_lite_tool.safetensors \
  E:/models/GPT2/mini-GPT/scxq2_tiles --hot 10
```

See the mini `driver_v2.kson` manifest (generated) for the declarative form.

## 5. ARC weights + kuhul physics

The manifest also links the **compiled weight** (the ARC artifact) and declares
the **K'UHUL physics** that shapes streaming:

- **`@arc.compiled_weight`** — the compiled artifact the tile pool is sharded
  from (path, format, size, `sha256`, `encoding: scxq2_f16`, `linked: true`).
  It's the specifically-linked compiled weight, covered by the manifest `@hash`.
- **`@physics`** — the gravity gate
  `clamp(1.0 + 0.35·pressure - 0.25·entropy + 0.15·attention + 0.10·affinity, 0.1, 4.0)`
  with per-tick params (pressure/entropy/attention/affinity) and
  `stream_priority: hot-first`. Physics determines which tiles stream/resident
  at each gravity tick over the pool.

Generator: `tools/build_mini_tile_kson.py` → emits `mini_driver_v2.kson`.

```bash
python tools/build_mini_tile_kson.py \
  E:/models/GPT2/mini-GPT/gpt2_small_lite_tool.safetensors \
  E:/models/GPT2/mini-GPT/mini_driver_v2.kson --hot 10
```

## 6. ARC constrains entropy; Xul collapses it

ARC is a geometric **constraint on state transition** (not entropy itself).

```
ENTROPY = uncertainty of state
ARC     = constraint on state transition
XUL     = collapse under ARC constraints
```

**ΔH(arc)** — the local entropy cost of moving along an arc:

```
ΔH(arc) = k_l·arc_len + k_c·|curvature| + k_b·bounds_penalty
H_next  = H_cur + ΔH(arc)          admissible iff H_next <= H_budget
```

The folds map onto the geometry (`@arc.geometry.fold_map`):

```
Pop   → establish geometric field
Wo    → establish state/constraints
Yax   → expose candidate arcs
Sek   → evaluate arc geometry (ΔH)
Ch'en → measure resulting field/state
Xul   → collapse onto admissible (lowest-entropy) arc → Pop
```

SCXQ2 integration: FP16/BF16 is the canonical reversible geometry; **INT4 is an
ARC-bounded projection** — local information may be thrown away only within the
arc's deviation budget; reconstruction returns to FP16 iff the state stays on
the permitted geometry. Runtime: `tools/arc_physics.py`.

## 7. Fold/Node — the model's own geometry

Two scales of one primitive — **Fold = bounded semantic container (DDS shard)**,
**Node = addressable tensor/range**:

```
MODEL
  ├─ Fold (DDS shard)        e.g. layer_07.dds
  │    ├─ Node (tensor)      attention.q_proj
  │    ├─ Node               mlp.up_proj
  │    └─ Node ...           (offset, length, shape, dtype, scxq2, arc, residency)
  ├─ Fold ...
  └─ Fold ...
```

Nodes describe **ranges/lanes into the shard** (no duplicated payload). Folds
can **unfold** — a node may reference another fold, so paging becomes **semantic
graph traversal**, not file paging. ARC entropy bounds the unfolding:

```
required nodes → which folds contain them → ΔH(candidate) → H_next ≤ budget → Xul → residency
```

Same primitive across the stack:

```
STRUCTURE  Fold=container  Node=element
STORAGE    Fold=DDS shard  Node=tensor/range
EXECUTION  Fold=Pop..Xul   Node=op contribution
TRANSPORT  Fold=SCXQ2 nhbd  Node=SCXQ2 tensor
PHYSICS    ARC constrains unfolding; Xul collapses it
```

Mini manifest (`mini_driver_v2.kson`) groups the model into **14 folds**
(embedding, `layer_00`–`layer_11`, output), each with its nodes. The loader
`tools/mini_tile_loader.py` runs the closed fold tick at fold level and streams
the winning fold via SCXQ2.

```bash
python tools/build_mini_tile_kson.py <model.safetensors> mini_driver_v2.kson --hot 10
python tools/mini_tile_loader.py mini_driver_v2.kson
```
