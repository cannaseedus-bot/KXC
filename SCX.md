# SCX.md — Symbolic Compute eXecution: Full Stack

> See also: SCXQ2.md (instruction set + XCFE executor detail), GPU.md (compute paths)

---

## The relationship

```
       SCX (namespace / compiler / CLI)
            │
            │  compile(XJSON) → binary
            ▼
         SCXQ2                    XCFE / XJSON
    (binary format)  ◄────────►  (human-readable format)
    .scxq2 file      decompile()  @op JSON programs
                     ──────────►
                                  │
                                  ▼
                             XCFE executor
                          (xcfe.cpp runs it)
```

User notation: **`SCXQ2 => XCFE/XJSON <= SCX`**

Both paths converge on XCFE/XJSON as the execution pivot:
- **SCX** compiles XCFE/XJSON programs → SCXQ2 binary (for storage, transmission, distribution)
- **SCXQ2** binary decompiles back → XCFE/XJSON (round-trip lossless)
- **XCFE** executes XCFE/XJSON directly — binary step is optional at runtime

---

## SCX — the C++ namespace

`namespace SCX` in `scxq2.cpp` owns:
- `compile(json program) → vector<uint8_t>` — XCFE/XJSON → SCXQ2 binary
- `decompile(vector<uint8_t>) → json` — SCXQ2 binary → XCFE/XJSON
- `lookup_op(name, OpcodeEntry&)` — name → group+subop lookup
- `op_name(group, subop) → const char*` — reverse lookup
- `SCXQ2Codec` — extensible instance: `register_op()`, `extend(op_table)`, `compile()`, `decompile()`

`SCXQ2Codec::extend(json op_table)` lets callers register custom ops at runtime. Custom ops auto-assign into the META group user-defined range (`next_user_subop_`). Extension table takes priority over the static table in all lookups.

---

## SCXQ2 binary format

Magic: `SCXQ2_MAGIC` (4 bytes) — file identifier  
Version: 1 byte (`SCXQ2_VERSION`)  
Flags: 1 byte (reserved, currently 0x00)

```
[MAGIC 4B][VERSION 1B][FLAGS 1B]
[StringDict section]
[Instruction count — 4 bytes LE]
[Instructions — variable length]
```

### StringDict

All variable names, output keys, and unknown op names are interned into a `StringDict`. Max 255 entries (uint8_t index). Encoded as:

```
[count — 2 bytes LE]
for each string: [length 1B][chars ...]
```

Used by args of type `DICT_REF` and by `out_ref`.

### Instruction encoding

Minimum 2 bytes. Layout:

```
Byte 0:  OPCODE(5) | MODE(2) | ARGC_hi(1)
Byte 1:  ARGC_lo(2) | SUBOP(5) | HAS_OUT(1)
[out_ref — 1 byte, only if HAS_OUT]
[args — variable, one per argc]
```

**Mode bits** in every instruction:
```
0b00 = CPU    → XVM fiber cluster
0b01 = GPU    → DirectML / OpenCL / cs_5_0
0b10 = HASH   → SCO SHA-256 path
0b11 = META   → compile-time metadata
```

### Arg types (top 2 bits of arg byte)

| Type | Bits | Payload (6 bits) |
|------|------|-----------------|
| `SMALL_INT` | 0b00 | signed 6-bit immediate (int8_t, range –32..+31 usable) |
| `DICT_REF` | 0b01 | index into StringDict (variable name, string literal) |
| `REGISTER` | 0b10 | register index r0–r63 |
| `ADDRESS` | 0b11 | addr_mode(2) + addr_offset(4+4 across 2 bytes) |

---

## Opcode groups (static table in `scxq2.cpp`)

| Group | ID | Opcodes |
|-------|----|---------|
| MEMORY | 0x00 | READ, WRITE, DELETE, APPEND, MOVE, COPY, EXISTS, SIZE |
| STRUCT | 0x04 | MERGE, DIFF, PATCH, KEYS, VALUES, SLICE, INDEX, FLATTEN, EXPAND |
| CONTROL | 0x08 | IF, SWITCH, LOOP, MAP, FILTER, REDUCE, PARALLEL, SYNC, WAIT, SIGNAL |
| MATH | 0x0C | ADD, SUB, MUL, DIV, MOD, POW, MIN, MAX, CLAMP, ABS, FLOOR, CEIL, SQRT |
| LOGIC | 0x10 | EQ, NEQ, GT, LT, GTE, LTE, AND, OR, NOT, XOR |
| BIT | 0x11 | BIT_AND, BIT_OR, BIT_XOR, BIT_SHL, BIT_SHR |
| TENSOR | 0x14 | MATMUL, DOT, TRANSPOSE, SOFTMAX, NORMALIZE, ATTENTION, FLASH_ATTN, KV_STORE, KV_LOAD, GPU_DISPATCH |
| MESH | 0x18 | LOAD_SIDECAR, UNLOAD, CALL, SPAWN, SEND, RECEIVE, BROADCAST, ROUTE, MESH_EXEC, NEGOTIATE |
| META | 0x1C | DEFINE_OP, UPDATE_OP, DELETE_OP, EVAL, COMPILE, LIST_OPS |
| I/O | 0x1D | PRINT, LOG |

Unknown ops encode as `META group + subop 0x1F` with the name as a `DICT_REF` arg — enabling forward compatibility with custom op extensions.

**TENSOR group note**: TENSOR opcodes (MATMUL, SOFTMAX, etc.) are **compute primitives**, not storage. At runtime they operate on XJSON tensors (`{"shape":[...],"data":[...],"dtype":"f32"}`). Weight files are SafeTensors (HuggingFace format). LoRA adapters are also SafeTensors. PyTorch is only in Python tooling. See `GPU.md § Tensor layers` for the full breakdown.

---

## scxq2_runtime.exe — the CLI

Built from `scxq2_main.cpp`. Full command surface:

```
scxq2_runtime.exe --compile   <in.json>   [out.scxq2]      XCFE/XJSON → binary
scxq2_runtime.exe --decompile <in.scxq2>  [out.json]       binary → XCFE/XJSON
scxq2_runtime.exe --roundtrip <in.json>                    compile+decompile, verify lossless
scxq2_runtime.exe --info      <in.scxq2>                   header + dict + instruction count
scxq2_runtime.exe --extend    <ops.json> --compile <in.json> [out.scxq2]   compile with custom ops
scxq2_runtime.exe --list-ops                               list all built-in opcodes
```

`--roundtrip` is the correctness gate: compiles a program to binary and decompiles it back, then `original.dump() == restored.dump()` must hold. Any instruction encoding/decoding regression surfaces here.

`--info` reads the magic, version, and dict without full decode — useful for inspecting a .scxq2 artifact without executing it.

`--extend <ops.json>` loads custom op definitions before compile, registering them into the META user-defined range.

---

## XCFE/XJSON ↔ SCXQ2 translation

`node_to_instr(json node, StringDict& dict)` — XCFE `@op` node → `Instruction`:
- `@op` name → `lookup_op()` → group + subop
- Unknown `@op` → META group, name interned as DICT_REF arg
- `@out` → `has_out=true`, key interned in dict
- `@in` array → args: `$var` strings → DICT_REF, integers → SMALL_INT, floats → SMALL_INT (truncated), other → DICT_REF of `json.dump()`

`instr_to_node(Instruction, StringDict)` — reverse:
- group+subop → `op_name()` → `@op`
- `out_ref` → `@out` via dict lookup
- Args → `@in` array: SMALL_INT → number, DICT_REF → `"$" + dict[ref]`, REGISTER → `"$r<n>"`, ADDRESS → `"@addr:<offset>"`

---

## How XCFE executes the program

XCFE does not require binary. It runs XCFE/XJSON directly:

```
XCFE::execute(program)
  ├── load_ops(@ops)              — register inline op definitions
  ├── seed scope from @state
  └── for each step in @control:
        exec_step(step, scope)
          └── eval(step, scope)
                ├── look up @op in op_defs  → composed/primitive dispatch
                └── look up @op in primitives → direct native dispatch
```

When a SCXQ2 binary is available, `decompile()` reconstructs the `@control` sequence and `XCFE::execute()` runs it identically. The binary is a transport/cache format, not a different execution model.

---

## Round-trip guarantee

`compile(program)` → `decompile(bytes)` produces identical JSON (verified by `--roundtrip`). This means:
- Any XCFE/XJSON program can be serialised to binary and back without semantic loss
- The StringDict preserves all variable names and custom op names
- SMALL_INT truncation is the only lossy case (float args with fractional parts — real programs use DICT_REF for floats)

---

## SCX utilization in kuhul_engine

`kuhul_engine.exe` uses SCX through its `--emit` command:

```
kuhul_engine --emit scx    <document>   → Kuhul::Source::semanticPackage(document)
kuhul_engine --emit scxq2  <document>   → Kuhul::Source::semanticPackage(document)
```

Both `"scx"` and `"scxq2"` are accepted emit targets alongside `"cpp"`, `"hlsl"`, `"wgsl"`, `"opencl"`, `"svg3d"`. The engine validates the target list and calls `semanticPackage()` for the SCX path — this is the codegen surface, not a direct call to `SCX::compile()`.

The raw `SCX::compile()` / `SCXQ2Codec` path is exposed separately by **`scxq2_runtime.exe`** (`--compile`, `--decompile`, `--roundtrip`, etc.).

Two separate entry points — same underlying format:
```
kuhul_engine --emit scxq2     ← high-level: document → semanticPackage() → SCX output
scxq2_runtime.exe --compile   ← low-level:  XCFE/XJSON @op program → binary .scxq2
```

## Micronauts directory

`C:\Users\canna\_khanary_inspect\micronauts\` — exists and populated:

```
Phase micronauts:  pop.json  wo.json  yax.json  sek.json  chen.json  xul.json
Chat/task:         chat.json  coder.json  default.json  eliza.json
                   factory.json  khanary.json  librarian.json  memory.json
                   registry.json  tool_call.json  ui.json  evolution.json
Subdirectory:      semantic/   (semantic variant micronauts)
```

Phase micronauts match the K'UHUL cycle (Pop→Wo→Yax→Sek→Ch'en→Xul). Each JSON defines a micronaut's sampling contract, system prompt, and fold assignment. The `FoldRegistry` (`native/runtime/fold_registry.h`) loads these via `folds.manifest.json`.

## File locations

| File | Role |
|------|------|
| `bin/json-runtime/src/scxq2.hpp` | Types: Mode, ArgType, Arg, Instruction, StringDict, OpcodeEntry, SCXQ2Codec |
| `bin/json-runtime/src/scxq2.cpp` | SCX namespace: static op table, compile/decompile, codec impl |
| `bin/json-runtime/src/scxq2_main.cpp` | `scxq2_runtime.exe` CLI entry point |
| `bin/json-runtime/src/xcfe.cpp` | XCFE executor (~1237 lines) |
| `bin/json-runtime/src/xcfe.hpp` | XCFE class |
| `build-llama/bin/Release/xcfe_probe.exe` | XCFE diagnostic probe |
| `build-llama/bin/Release/xcfe_matmul_test.exe` | DirectML matmul smoke test |

All paths relative to `C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\`.

Authoritative docs: `SCX.md` (this file), `SCXQ2.md` (executor + instruction set detail), `GPU.md` (compute paths).
