# JROM — JSON Runtime Object Model

JROM is the binary ROM format for AtomicDOM programs.

JROM is also the compiled transport for Micronaut template adapters.  The
adapter pipeline has one canonical semantic form (KXML), while rendering is
performed in the target model's native dialect: Jinja, ChatML, or an explicit
KXML surface.  A stock model's existing chat-template block tokens are never
replaced by KXML markers; the adapter selects and verifies the correct native
block map before committing the AtomicDOM program to JROM.

`https://kuhul.dev/` is the KUHUL vocabulary namespace used in manifests.  It
is not a schema URL and the runtime does not fetch it.  Validation is local and
explicit through the repository's tools and runtime contracts.

**AtomicDOM** is the program language — the structured instruction format
already in use across the stack (`atomic_block.json`, `programs/main.json`,
`morrowind-entry.json`). Its fields are `@kind`, `@program`, `@import`,
`@state`, `@control`, `@folds`, `@rom`, `@run`.

**JROM** is the binary encoding of AtomicDOM: an AtomicDOM document
CBOR-encoded and wrapped in a 5-byte header, producing a portable `.rom`
binary that the runtime executes directly without text parsing.

```
AtomicDOM document (.json)
  ─── CBOR encode ──→  JROM binary (.rom)
─── decode_rom() ──→  AtomicDOM document (round-trip)
```

Template adapter flow:

```
Jinja / ChatML / role messages → KXML canonical blocks
KXML + native block map       → AtomicDOM adapter
AtomicDOM adapter             → JROM + CBOR
```

The declarative adapter contract is in
`programs/micronauts/jrom-template-adapter.json`; its compiler is
`tools/jrom_template_adapter.py`.  Actual stock-model rendering remains in
`tools/kxml_stock_adapter.py`, which reads the model's native Jinja template
and tool convention.

The grammar (`jrom.grammar.abnf`) and schema (`jrom.schema.json`) formally
specify the AtomicDOM language. The existing `atomic_block.json` in `programs/`
is the reference AtomicDOM implementation — everything in it is valid JROM.

---

## 1. Binary Format

```
┌────────────────────────────────────────────────────────────┐
│ magic   │ 4 bytes │ ASCII "JROM"  (0x4A 0x52 0x4F 0x4D)  │
│ version │ 1 byte  │ 0x01                                  │
│ payload │ N bytes │ CBOR-encoded JSON program (RFC 7049)  │
└────────────────────────────────────────────────────────────┘
```

CBOR (Concise Binary Object Representation, RFC 7049 / RFC 8949) is binary JSON.
A JSON object encodes to CBOR with full round-trip fidelity and typically
30–50% smaller than its text equivalent — no parsing overhead at runtime.

**Grammar:** `dist/json-runtime/jrom.grammar.abnf`  
**Schema:** `dist/json-runtime/jrom.schema.json` (JSON Schema draft-07)

---

## 2. Program Structure

After `decode_rom()` the CBOR payload becomes a JSON program object.
Required fields: `@program`, `@control`. All others are optional.

```json
{
  "@program": "my-program",

  "@import": [
    { "@from": "stdlib/audio.kuhul", "@use": ["fold_tone"] },
    { "@from": "../../programs/api.manifest.json" },
    { "@from": "../../programs/cache.manifest.json" }
  ],

  "@state": {
    "a": 5,
    "b": 7,
    "result": null
  },

  "@control": [
    { "@op": "MUL",  "@in": ["a", "b"],  "@out": "result" },
    { "@op": "GT",   "@args": ["$result", 30], "@out": "big" },
    { "@op": "IF",
      "@cond": "$big",
      "@then": { "@op": "LOG", "@args": ["result is big"] },
      "@else": { "@op": "LOG", "@args": ["result is small"] }
    }
  ]
}
```

`@import` paths are resolved relative to the manifest's `@paths.root`.  
Stdlib modules (`stdlib/*.kuhul`) load K'UHUL semantic functions.  
Shared manifests (`../../programs/*.manifest.json`) load capability declarations;  
if a matching `.cpp` extension exists in `dist/json-runtime/src/`, it is loaded as a native op set.

---

## 3. ROM Lifecycle

### 3.1 Compile — JSON program → JROM binary

```bash
# via API
curl -X POST http://localhost:8905/api/rom/compile \
  -H "Content-Type: application/json" \
  -d @programs/main.json \
  --output main.rom

# via tool
python tools/jrom_to_tensor.py --compile programs/main.json --out main.rom
```

### 3.2 Write — compile and write in one call

```bash
curl -X POST http://localhost:8905/api/rom/write \
  -H "Content-Type: application/json" \
  -d '{ "program": { "@program": "demo", "@control": [...] }, "path": "dist/demo.rom" }'
```

### 3.3 Execute — run a ROM binary

```bash
# Send JROM binary directly (Content-Type: application/rom)
curl -X POST http://localhost:8905/api/rom/exec \
  -H "Content-Type: application/rom" \
  --data-binary @main.rom

# Or send the JSON program (Content-Type: application/json)
curl -X POST http://localhost:8905/api/rom/exec \
  -H "Content-Type: application/json" \
  -d @programs/main.json
```

### 3.4 Parse — inspect without executing

```bash
# Decode JROM binary back to human-readable JSON
curl -X POST http://localhost:8905/api/rom/parse \
  -H "Content-Type: application/rom" \
  --data-binary @main.rom
```

---

## 4. API Reference

All routes registered by `src/api.cpp`, declared in `programs/api.manifest.json`.

| Method | Route               | Body                             | Returns            |
|--------|---------------------|----------------------------------|--------------------|
| POST   | `/api/rom/compile`  | JSON program                     | `application/rom` binary |
| POST   | `/api/rom/exec`     | `application/rom` or JSON program | `{ result: {...} }` |
| POST   | `/api/rom/parse`    | `application/rom` binary         | JSON program |
| POST   | `/api/rom/write`    | `{ program, path }`              | `{ written, bytes }` |
| GET    | `/api/rom/health`   | —                                | `{ status, rom_version, format }` |

Inline JSON programs also work on the existing `/api/run` route (no compile step).
JROM's `/api/rom/exec` skips JSON text parsing for faster repeated execution.

---

## 5. CBOR → Tensor

JSON programs are structured instruction streams.  
Encoding them as tensors lets models read, analyze, and learn from programs directly.  
This is not a training loop — it is a data translation layer.

### 5.1 Raw byte tensor (uint8)

The simplest form: treat the CBOR binary payload as a flat array of bytes.

```python
import cbor2, torch, numpy as np

data = open("main.rom", "rb").read()
cbor_payload = data[5:]                          # strip JROM header
tensor = torch.from_numpy(
    np.frombuffer(cbor_payload, dtype=np.uint8).copy()
)
# shape: (N,)  dtype: torch.uint8
```

**Use for:** byte-level anomaly detection, pattern matching on raw program binaries.

### 5.2 Op-token embedding (int32)

Walk `@control`, map each `@op` name to an integer token ID.
The result is a sequence tensor — one integer per instruction.

```python
import cbor2, torch, numpy as np

OP_VOCAB = { "ADD":1, "MUL":2, "IF":3, "LOG":4, ... }   # see jrom_to_tensor.py

program = cbor2.loads(open("main.rom","rb").read()[5:])
op_ids  = [OP_VOCAB.get(n["@op"], 0) for n in program["@control"]]
tensor  = torch.tensor(op_ids, dtype=torch.int32)
# shape: (n_ops,)
```

**Use for:** program-similarity search, sequence model input, embedding programs
into a vector space alongside natural language prompts.

### 5.3 Via CLI tool

```bash
# embed mode (int32 op-token sequence, default)
python tools/jrom_to_tensor.py --rom main.rom

# raw mode (uint8 byte tensor)
python tools/jrom_to_tensor.py --rom main.rom --mode raw

# from a JSON file (auto-encodes to JROM first)
python tools/jrom_to_tensor.py --json programs/main.json
```

### 5.4 Connection to the kuhul stack

The op-token tensor format feeds naturally into the kuhul inference chain:

```
JROM binary
  → decode_rom() → JSON program
  → embed_tensor() → int32 sequence tensor
  → SCXQ2 INT8 quantisation → shard tile
  → DirectML kernel (d3d11_infer.dll) → model output
```

Models in the stack (micronaut-coder, gpt-oss) can receive program tensors as
context, enabling program-aware generation and semantic reasoning over bytecode.

---

## 6. The `.rom.html` Pattern

Files named `*.rom.html` are ROMs in HTML form — the HTML IS the program.
The runtime reads the file, the browser executes it, and the json_runtime API
provides the backend.  The JROM binary format is the compiled equivalent:
same program, binary encoding, zero parsing cost, direct tensor conversion.

```
morrowind-asx.rom.html  ← ROM as HTML (browser executes)
programs/main.json       ← ROM as JSON (json_runtime executes)
dist/demo.rom            ← ROM as JROM binary (exec without parsing)
```

All three are the same abstraction at different encoding layers.

---

## 7. Files

| File | Purpose |
|---|---|
| `dist/json-runtime/src/api.cpp` | ROM route implementation (`/api/rom/*`) |
| `dist/json-runtime/src/api.hpp` | `API::encode_rom`, `API::decode_rom`, `API::register_routes` |
| `dist/json-runtime/jrom.grammar.abnf` | ABNF grammar for binary format + program structure |
| `dist/json-runtime/jrom.schema.json` | JSON Schema draft-07 for decoded programs |
| `programs/api.manifest.json` | API surface declaration (`@import` target for programs) |
| `tools/jrom_to_tensor.py` | CLI: compile, raw tensor, embed tensor |
