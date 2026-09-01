# SCXQ2.md — K'uhul Symbolic Compute eXecution (Quantized, v2)

> Single source of truth for the SCXQ2 instruction set and its XCFE runtime executor.
> Status: **zero errors, working as defined** — not scaffolding, not magic numbers.

---

## What SCXQ2 is in this stack

**SCXQ2 is the instruction set specification.** It defines opcodes, mode bits, and region types.
**XCFE is the runtime executor.** It interprets SCXQ2 programs encoded as XJSON.
**XJSON is the encoding.** Programs are JSON objects with `@op` nodes — not binary, but intentional and fully enforced.

These three are one system:

```
SCXQ2 spec (scxq2.hpp)
  └── opcode groups: Tensor, Control, Mesh, RegionKind
  └── Mode bits: CPU=0b00  GPU=0b01  HASH=0b10  META=0b11

XCFE executor (xcfe.cpp)
  └── interprets XJSON @op programs
  └── primitives: READ, WRITE, EVAL, HASH, CALL, PHASE, BOT, STACK, PROGRAM
  └── dispatch: sco://, sidecar://, compile_gpu_kernel(), tensor_runtime()

XJSON encoding
  └── every instruction = JSON object { "@op": "...", "@fn": "...", "@args": [...] }
  └── @control = sequence of steps (the program body)
  └── @state   = initial variable bindings
  └── @ops     = inline op definitions
```

### XVM D3D12 runtime driver packaging

The `dist/xvm-d3d12` runtime projection now packages the executor as a driver DLL:

- `dist/xvm-d3d12/drivers/xvm_d12.dll` — SCXQ2/SCXG runtime driver core
- `dist/xvm-d3d12/drivers/xvm_d12_host.exe` — optional CLI host that loads the DLL

The driver path enforces the same SCXQ/SCXQ-DDS loader law as the executable path:
profile+magic+version gate, SCXQ checksum verification, and optional Atomic DOM /
micronaut compatibility gates before backend dispatch.

---

## Mode bits (per-instruction routing)

Every SCXQ2 instruction carries a 2-bit `Mode` field in the spec:

| Mode | Bits | Routes to |
|------|------|-----------|
| `CPU` | `0b00` | XVM 32-fiber thread cluster (`xvm_run_cpu_ticks_mt`) |
| `GPU` | `0b01` | DirectML / OpenCL / D3D11 cs_5_0 |
| `HASH` | `0b10` | SCO SHA-256 content-addressed path |
| `META` | `0b11` | Compile-time metadata only |

In `xcfe.cpp` the routing is via URI scheme on `native.CALL`:
- `sco://alias` → `Mode::HASH` path (SCORegistry lookup by SHA-256)
- `sidecar://name/op` → SidecarLoader plugin dispatch
- `@fn:"dispatch"` → `Mode::GPU` path (D3DCompile → cs_5_0 bytecode)
- `@fn:"matmul"` → `Mode::GPU` path (dml_gemm_bt_f32 via DirectML)
- XVM manifold ops → `Mode::CPU` path (fiber cluster)

---

## Opcode groups (from `scxq2.hpp`)

### Tensor subops (0x14–0x17)

**SCXQ2 Tensor ops are compute primitives — not storage formats.**
They operate on XJSON tensors at runtime (`{"shape":[M,K],"data":[...],"dtype":"f32"}`).
Weight files on disk are SafeTensors (HuggingFace). LoRA adapters are also SafeTensors.
PyTorch exists only in the Python tooling — never in the C++ runtime.
See `GPU.md § Tensor layers` for the full four-layer breakdown.

| Subop | Name | Implemented in xcfe/tensor_runtime? |
|-------|------|--------------------------------------|
| 0 | MATMUL | yes — XJSON tensor → `dml_gemm_bt_f32()` → DirectML, CPU fallback |
| 1 | DOT | no |
| 2 | TRANSPOSE | no |
| 3 | SOFTMAX | yes — CPU, XJSON tensor |
| 4 | NORMALIZE | no |
| 5 | ATTENTION | no (shader exists: `xvm_fused_qkv_attention.cso`) |
| 6 | FLASH_ATTN | no |
| 7 | KV_STORE | no |
| 8 | KV_LOAD | no |
| 9 | DISPATCH | yes — `compile_gpu_kernel()` → D3DCompile bytecode |

#### XJSON tensor format (in-flight)

```json
{
  "shape":   [M, K],
  "data":    [1.0, 2.0, ...],
  "dtype":   "f32",
  "backend": "khanary-directml"   // or "cpu-fallback"
}
```

Named tensors persist across calls via `tensor_register` / `tensor_get` registry in `tensor_runtime.cpp`. Gap: JSON float serialization is the bottleneck for large tensors — replace with binary transfer for the hybrid trainer (gap #5).

#### Weight pipeline (how SafeTensors → GPU)

```
.safetensors on disk
  → load via safetensors.torch (Python tools) OR direct binary parse (C++)
  → float buffer in CPU memory
  → dml_gemm.dll upload to D3D12 heap
  → GPU-resident; stays there (per-shape resource cache in dml_gemm.dll)

LoRA flow:
  oss_distillation.py (PyTorch) → from_zero_v0.6_lora.safetensors
  → SLERP merge tool → merged SafeTensors checkpoint
  → load as above
```

### Control opcodes

| Name | Status |
|------|--------|
| `PARALLEL` | spec'd; sequential in xcfe.cpp executor (needs thread pool) |
| `SYNC` | spec'd; barrier join |
| `MESH.MESH_EXEC` | spec'd; no handler yet |

### Region types

| RegionKind | Meaning |
|---|---|
| `FOLD` | Fold tensor region — dispatches `kuhul_fold_compute.cso` / `cs_fold_kernel_compute_.hlsl` |

---

## XCFE primitives (all implemented and working)

| Primitive | `@fn` values | What it does |
|-----------|-------------|--------------|
| `native.READ` | `read, exists, keys, values, size` | Scope + global_state lookup |
| `native.WRITE` | `write, delete` | Write to scope AND global_state (persistent across steps) |
| `native.STACK` | `push, pop` | Runtime stack via `__stack` in global_state |
| `native.PROGRAM` | `exec` | Execute a named function from `__program_functions` |
| `native.EVAL` | math, logic, bitwise, flow, define, dispatch, tensor | Full expression + GPU dispatch surface |
| `native.HASH` | *(none)* | SHA-256 of serialised value → hex string |
| `native.CALL` | *(none)* | External dispatch: `sco://`, `sidecar://`, `http://`, `discovered://` |
| `native.PHASE` | `phase, transition, fold, manifold` | K'uhul phase state machine |
| `native.BOT` | *(none)* | Build BotRuntime payload for bot helper service |

### native.EVAL `@fn` surface

**Math**: `add sub mul div mod pow min max abs floor ceil sqrt clamp`
**Logic**: `eq neq gt lt gte lte and or not xor`
**Bitwise**: `band bor bxor shl shr`
**Flow**: `if loop map filter reduce`
**IO**: `log`
**Meta**: `define undef list eval`
**GPU**: `dispatch` → `compile_gpu_kernel()`
**Tensor**: `alloc matmul relu softmax copy tensor_register tensor_get tensor_exists tensor_remove tensor_list`

---

## Expression evaluator (infix, built into xcfe.cpp)

Full recursive-descent parser. Operator precedence (high to low):

```
unary:   !  -
power:   **
mul:     *  /  %
add:     +  -
compare: <  >  <=  >=  ==  !=
logical: &&  ||
```

`$varname` resolves from scope → global_state. Parentheses supported. Used by `@expr` field on any node.

---

## Phase system (native.PHASE)

Six phases forming a complete unit circle at π/3 intervals:

| Phase | Angle | Glyph | Operation |
|-------|-------|-------|-----------|
| Pop | 0 | perceive.circle | perceive_load |
| Wo | π/3 | represent.hex | represent_build |
| Yax | 2π/3 | plan.triangle | plan_schedule |
| Sek | π | execute.star | execute_compute |
| Ch'en | 4π/3 | project.diamond | project_output |
| Xul | 5π/3 | consolidate.hex | consolidate_replay |

Transitions are **legally enforced**: only `Pop→Wo→Yax→Sek→Ch'en→Xul→Pop`. Illegal jumps throw a runtime error. `PHASE_TRANSITION` (XVM opcode 0x46) enforces the same rule at the CPU fiber level.

Phase state is stored in `__kuhul_phase` (global_state), history in `__kuhul_phase_history`.

`@fn:"fold"` executes a named sub-sequence of operations in the current phase, records result in `__kuhul_fold` and `fold`, and writes the fold descriptor into the phase state.

`@fn:"manifold"` returns the full phase manifold object (`@kind: "kuhul.phase.manifold.v1"`).

---

## SHA-256 implementation (xcfe.cpp, lines 20–83)

Implemented from scratch, no external deps. The K256 round constants (`0x428a2f98, 0x71374491, ...`) are the exact NIST SHA-256 standard constants — not magic numbers. Correct bit-length padding (append `0x80`, pad to 56 mod 64, append 64-bit big-endian bit count). Full 64-round compression with correct `rotr32`. Used by `native.HASH` and the SCO content-addressed store.

---

## SCO — Symbolic Cache Object

SHA-256 registry for GPU programs. Alias → bytecode object. `native.CALL` with `sco://alias` dispatches through SCO:

```
sco://alias
  → SCORegistry::resolve(alias, obj)
      hit  → if obj has @control: run it in a child XCFE instance
           → else: return obj.payload
      miss → error "sco_not_found"
```

`D3DSCache.dll` is available for persisting compiled shader bytecode, but the SCO↔D3DSCache wiring is not yet connected. Currently `compile_gpu_kernel()` recompiles on every call.

---

## XJSON program format

```json
{
  "name": "my_program",
  "@ops": { "my_op": { "@type": "primitive", "@impl": "native.EVAL", "@fn": "add" } },
  "@state": { "x": 10, "y": 20 },
  "@control": [
    { "@op": "native.EVAL", "@fn": "add", "@args": ["$x", "$y"], "@out": "result" },
    { "@op": "native.EVAL", "@fn": "log", "@args": ["$result"] }
  ],
  "functions": {
    "my_fn": [
      { "@op": "native.EVAL", "@fn": "mul", "@args": ["$a", "$b"], "@out": "product" }
    ]
  }
}
```

Key fields:
- `@op` — operation name (primitive or defined op)
- `@fn` — function within that primitive
- `@args` / `@in` — arguments (can be `$var` references or literals)
- `@out` — write result to this scope key
- `@cond` / `@then` / `@else` — conditional
- `@control` — main program sequence
- `@state` — initial bindings
- `@ops` — inline op definitions (extend at program load time)
- `functions` — named function bodies for `native.PROGRAM exec`

---

## Sidecar dispatch

`native.CALL` with `sidecar://name/op` routes to `SidecarLoader::call(name, op_name, scope)`. Scope is passed through. Used for plugin/extension ops that live outside the core XCFE primitives.

---

## Delta stream (apply_delta)

`XCFE::apply_delta(program, delta_ops)` patches a running program in place:
- `UPDATE @state.key` — update a state variable
- `INSERT @control idx` — insert a step at index
- `APPEND @control` — append a step
- `DELETE @control idx` — remove a step

Used for live program mutation during execution.

---

## What is NOT yet wired (gap list)

1. **Mode bit dispatch in tensor_runtime.cpp** — `tensor_runtime()` ignores `Mode::CPU`; always tries DirectML → CPU fallback. Should route `Mode::CPU=0b00` directly to XVM fiber cluster.
2. **TENSOR.ATTENTION / FLASH_ATTN** — shaders exist (`xvm_fused_qkv_attention.cso`) but no C++ dispatch path from tensor_runtime.
3. **CONTROL.PARALLEL** — runs sequentially in xcfe.cpp; needs thread pool for true GPU+CPU fan-out.
4. **`@gpu.buffer.write/read`** — declared in XCFE stdlib gpu capability, not yet in C++.
5. **SCO ↔ D3DSCache** — shader bytecode cache not wired; recompiles on every `@fn:"dispatch"`.
6. **KV_STORE / KV_LOAD** — no C++ path yet.

---

## File locations

| File | Role |
|------|------|
| `bin/json-runtime/src/scxq2.hpp` | Mode bits, opcode enums, RegionKind::FOLD |
| `bin/json-runtime/src/xcfe.cpp` | Full XCFE executor (~1237 lines) |
| `bin/json-runtime/src/xcfe.hpp` | XCFE class declaration |
| `bin/json-runtime/src/tensor_runtime.cpp` | Tensor ops + DirectML dispatch |
| `bin/json-runtime/src/gpu_dispatch.cpp` | `compile_gpu_kernel()` → D3DCompile |
| `bin/json-runtime/src/sco.hpp` | SCO SHA-256 registry |
| `bin/json-runtime/src/sidecar.hpp` | SidecarLoader plugin system |
| `bin/json-runtime/src/bots.hpp` | BotRuntime payload builder |
| `bin/json-runtime/folds.manifest.json` | Fold registry manifest |
| `build-llama/bin/Release/kuhul_engine.exe` | Main engine binary |
| `build-llama/bin/Release/xcfe_probe.exe` | XCFE probe / diagnostic tool |
| `build-llama/bin/Release/xcfe_matmul_test.exe` | DirectML matmul smoke test |

All paths relative to `C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\`.

## Current XCFE bytecode boundary

The repository's canonical portable semantic bytecode is **SCXQ2 v2**. It is
not the illustrative fixed 8-bit opcode struct sometimes used in design notes.
SCXQ2 v2 uses:

`SCXQ` magic → version/flags → string dictionary → instruction count →
variable-length instructions.

Each instruction carries a compact opcode group, sub-operation, execution mode
(`CPU`, `GPU`, `HASH`, or `META`), arguments, and an optional output reference.
This keeps XCFE semantic programs backend-neutral. The selected runtime backend
then maps tensor/field operations to CPU, DirectML/D3D11, OpenCL 1.2, WebGL2,
or another registered adapter.

The compiler CLI is built from `dist/json-runtime/src/scxq2_main.cpp`:

```powershell
& .\dist\json-runtime\build\Release\scxq2_runtime.exe --compile `
    programs\batches.compute.json scratch\batches.compute.scxq2
& .\dist\json-runtime\build\Release\scxq2_runtime.exe --info `
    scratch\batches.compute.scxq2
& .\dist\json-runtime\build\Release\scxq2_runtime.exe --decompile `
    scratch\batches.compute.scxq2 scratch\batches.compute.decompiled.json
```

OpenCL/WebGL2/D3D11 shader binaries remain execution adapters; compiling XCFE
bytecode does not itself execute a GPU kernel. A runtime dispatch step must
resolve the semantic operation against the active backend registry.
