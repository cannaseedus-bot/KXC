# SMCA — Semantic Microcoded Architecture

SMCA is the **root constitution** for sovereign microcoded systems. It defines
**what kinds of machines may exist**, not how they run.

Two compilers produce SMCA-compliant output:

| Repo | Compiler | Domain |
|------|----------|--------|
| https://github.com/cannaseedus-bot/KXC | KXC (`kxc.exe`) | compute shader kernels → HLSL / WGSL / SMCA |
| https://github.com/cannaseedus-bot/KHLC-PY | KHLC (`khlc.py`) | semantic programs → KAST / KSON |
| https://github.com/cannaseedus-bot/SMCA | SMCA registry | architecture spec (this repo) |

| Compiler | Input | Output | Domain |
|----------|-------|--------|--------|
| **KXC** (`kxc.exe`) | `.kuhul` kernel descriptor | `.smca.json`, HLSL, WGSL, CPU C++ | compute shaders |
| **KHLC** (`khlc.py`) | `.kuhul` / `.khl` semantic source | KAST + KSON | programs / driver contracts |

---

## KXC — kernel compiler

Writing a compute shader kernel is three lines:

```kuhul
[Pop FusedAttention]
  [Muwan dispatch 64 1 1]
  [Sek needsSoftmax true]
  [Sek needsMatMul true]
  [Xul]
```

Run it:

```powershell
.\kxc.exe fused_attention.kuhul
# emitted: FusedAttention.hlsl  FusedAttention.wgsl  FusedAttention.cpu.cpp
#          FusedAttention.smca.json  FusedAttention.ir.json
```

The `.smca.json` output is the compiled kernel contract — SMCA-tagged, registry-matched,
ready for runtime dispatch:

```json
{
  "kernel": "FusedAttention",
  "threads": [64, 1, 1],
  "smca": {
    "kernelClass": "tensor_attention_fused",
    "collapseClass": "attention.fused",
    "lawful": true,
    "registryMatched": true,
    "layers": ["MATRIX", "SCXQ2", "SCXQ7", "SCO/1", "IDB"]
  }
}
```

KXC grammar tokens: `[Pop KernelName]`, `[Muwan dispatch X Y Z]`, `[Sek property value]`, `[Xul]`.
Full reference: [`KXC.md`](../KXC.md).

---

## KHLC — semantic compiler

Writing a semantic program is phase blocks:

```kuhul
⟁ Pop ⟁
  bind π = 3.141592653589793
  probe geometry

⟁ Sek ⟁
  dispatch provider(area)

⟁ Xul ⟁
  commit result
```

Run it:

```powershell
python tools/khlc.py stdlib/pi.kuhul
# emitted: pi.kson  (KAST document — protocol kast/1)
```

Driver contracts use the glyph form (`.khl`):

```khl
glyph opengl::dispatch(TENSOR_IN) →
  gl::upload(TENSOR_IN)      → GPU_BUF
  gl::dispatch(GPU_BUF, PROG) → OUT_BUF
  yield OUT_BUF
```

Full reference: [`KUHUL.md`](../KUHUL.md).

---

## What SMCA is

- A machine architecture specification
- A semantic ISA (instruction *meaning*, not execution)
- A collapse-based computational ontology
- A role partitioning system for clusters
- A lawful separation of imagination vs reality

## What SMCA is not

- A runtime — A VM — A kernel — A programming language — A scheduler

> **SMCA defines what kinds of machines may exist — not how they run.**

---

## Core layers

| Layer | Meaning |
|-------|---------|
| MATRIX | Authoring / proposal space |
| SCXQ2 | Semantic μ-op encoding |
| SCXQ7 | Firmware + legality governor |
| SCO/1 | Sovereign compute object |
| CM-1 | Pre-semantic control gate |
| IDB | Indexed causal memory |

## Authority gradient

```
MATRIX (imagination)
   ↓
CM-1 (gate)
   ↓
SCXQ7 (law)
   ↓
SCXQ2 (microcode)
   ↓
SCO/1 (execution)
   ↓
IDB (memory)
```

Authority flows downward only. No layer may escalate, skip, or collapse without law.

---

## Registry

| File | Contents |
|------|----------|
| `registry/kernel-classes/v1.json` | kernel → collapse class, requires, forbids |
| `registry/kernels/<name>.json` | per-kernel description and frozen status |
| `registry/collapse-geometry/v1.json` | collapse geometry descriptors |
| `registry/cluster-roles/v1.json` | cluster role definitions |
| `schemas/kernel-tag.schema.json` | kernel tag schema |
| `schemas/idb.schema.json` | IDB schema |
| `axioms/architecture.json` | architecture axioms |
| `axioms/collapse-geometry.json` | collapse geometry axioms |

## Repository layout

```
smca/
├─ README.md
├─ axioms/
├─ schemas/
├─ registry/
│  ├─ collapse-geometry/v1.json
│  ├─ kernel-classes/v1.json
│  ├─ kernels/
│  │  ├─ binary_split.json
│  │  ├─ neural_layer_kuhul.json
│  │  └─ tensor_attention_fused.json
│  └─ cluster-roles/v1.json
├─ conformance/
└─ hashes/
```

---

> **SMCA defines a sovereign microcoded machine in which imagination is external,
> execution is internal, and reality changes only by lawful collapse.**
