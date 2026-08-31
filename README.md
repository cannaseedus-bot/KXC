# KXC — K'UHUL Kernel Compiler

**GitHub:** https://github.com/cannaseedus-bot/KXC
**KHLC (semantic compiler):** https://github.com/cannaseedus-bot/KHLC-PY
**SMCA registry:** https://github.com/cannaseedus-bot/SMCA

KXC compiles `.kuhul` kernel descriptors into executable GPU and CPU artifacts.

---

## Writing a kernel takes three lines

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
```

Output:

```
FusedAttention.hlsl        HLSL compute shader  (D3D12 / DirectML)
FusedAttention.wgsl        WGSL compute shader  (WebGPU)
FusedAttention.cpu.cpp     CPU fallback
FusedAttention.smca.json   compiled kernel contract
FusedAttention.ir.json     SCXQ2-lowered IR
```

---

## The compiled kernel contract

`.smca.json` is what the runtime inspects before dispatch — kernel identity,
thread shape, hardware caps, classification, and legality in one document:

```json
{
  "kernel": "FusedAttention",
  "threads": [64, 1, 1],
  "caps": { "waveOps": false, "heapTier": 1, "bindingTier": 1, "uma": true },
  "smca": {
    "kernelClass": "tensor_attention_fused",
    "collapseClass": "attention.fused",
    "lawful": true,
    "registryMatched": true,
    "layers": ["MATRIX", "SCXQ2", "SCXQ7", "SCO/1", "IDB"],
    "requires": ["deterministic_join", "bounded_reduction"],
    "forbids": ["side_effects", "order_dependence"]
  }
}
```

---

## Grammar

| Token | Role |
|-------|------|
| `[Pop KernelName]` | open kernel block — sets the kernel name |
| `[Muwan dispatch X Y Z]` | thread group dimensions |
| `[Sek property value]` | kernel property flag (bool / int / string) |
| `[Xul]` | close block and compile |

Known `[Sek]` properties: `needsDecompress`, `needsSoftmax`, `needsMatMul`,
`kvInt4`, `needsMoERoute`, `needsMoEExpertFFN`, `needsMoECombine`, `needsPhaseMatch`

---

## Pipeline layers

| Layer | Meaning |
|-------|---------|
| MATRIX | source parsed into AST |
| SCXQ2 | lowered to backend-neutral IR |
| SCXQ7 | legality + caps-aware optimization |
| SCO/1 | backend emitters produce artifacts |
| IDB | sidecar metadata for external verification |

---

## Registry

Kernel classification is driven by three files in `registry/`:

- **`kernel-aliases.json`** — maps intermediate class names to canonical names
- **`kernel-classes.json`** — canonical class definitions (requires / forbids / backend / layers)
- **`kernel-extras.json`** — caps hints and fallback backends

The SMCA registry (architecture spec) lives at https://github.com/cannaseedus-bot/SMCA

---

## Verified kernels

```
examples/fused_attention_full.kuhul      kernelClass: tensor_attention_fused
examples/fused_attention_simple.kuhul    kernelClass: tensor_attention_fused
examples/binary_split_test.kuhul         kernelClass: generic-compute
examples/neural_layer_kuhul_test.kuhul   kernelClass: generic-compute
```

All four: `registryMatched: true`, exit 0.

---

## Relationship to KHLC

KXC and KHLC are two compilers for the same language, different domains:

```
.kuhul kernel descriptor  →  KXC   →  HLSL / WGSL / SMCA JSON   (compute shaders)
.kuhul / .khl source      →  KHLC  →  KAST / KSON               (programs)
```
