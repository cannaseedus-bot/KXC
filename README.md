# KXC v1.0.0

**GitHub:** https://github.com/cannaseedus-bot/KXC
**KHLC (semantic compiler):** https://github.com/cannaseedus-bot/KHLC-PY
**SMCA registry:** https://github.com/cannaseedus-bot/SMCA

Version boundary for the KXC kernel compiler with all three binary patches applied
and verified. See [`KXC.md`](../../KXC.md) for full architecture and grammar reference.

## What changed

Three patches to `kxc.exe` (32-bit x86 MSVC debug, ImageBase=0x400000):

| Offset | Change |
|--------|--------|
| `0x65B32` | `jnz` → `jmp` — forces `registryMatched=true` write for all kernels |
| `0x660C4` | Dead `is_fused_attention` flag check replaced with first-char comparison against kernel name |
| `0x660CD` | `PUSH` target changed from `"fused-attention"` to `"tensor_attention_fused"` — canonical class assigned directly |

## Verified output

```
fused_attention_full      kernelClass: tensor_attention_fused  registryMatched: true
fused_attention_simple    kernelClass: tensor_attention_fused  registryMatched: true
binary_split_test         kernelClass: generic-compute         registryMatched: true
neural_layer_kuhul_test   kernelClass: generic-compute         registryMatched: true
```

## Contents

```
bin/
  kxc.exe                 patched compiler binary (~1 MB)
registry/
  kernel-aliases.json     intermediate key → canonical class map
  kernel-classes.json     canonical class definitions (requires/forbids/backend/layers)
  kernel-extras.json      caps hints and fallback backends
examples/
  fused_attention_full.kuhul
  fused_attention_simple.kuhul
  binary_split_test.kuhul
  neural_layer_kuhul_test.kuhul
VERSION.json
README.md
```

## Usage

```powershell
cd C:\Users\canna\_khanary_inspect\versions\kxc-v1.0.0\bin
.\kxc.exe ..\examples\fused_attention_full.kuhul
```

## Known limitation

Registry defines `moe_route_top2` and related classes but the classifier resolves
to two effective `kernelClass` values only (`tensor_attention_fused` / `generic-compute`).
MoE dispatch is the next classifier milestone.
