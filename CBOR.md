# CBOR.md — JROM Binary Format

JROM (JSON Runtime Object Model) is the binary encoding for compiled micronaut
AtomicDOM programs. Each `.rom` file is a deterministic binary artifact produced
by `tools/micronaut_to_jrom.py` from `micronauts/registry.json`.

---

## Wire format

```
[4 bytes]  magic    = "JROM"  (0x4A 0x52 0x4F 0x4D)
[1 byte]   version  = 0x01
[n bytes]  payload  = CBOR encoding of the AtomicDOM program object
```

The payload is a standard RFC 8949 CBOR map. No custom tags. Encoded by the
Python `cbor2` library (`pip install cbor2`). The AtomicDOM object structure
is identical to the corresponding `.atomicdom.json` file — CBOR is just the
compact binary serialization of the same data.

### Why CBOR

- Compact (no string key repetition overhead vs JSON)
- Byte-exact round-trip: `cbor2.loads(cbor2.dumps(obj)) == obj` for all
  types used here (maps, lists, strings, floats, ints, booleans)
- Deterministic under `cbor2.dumps` for dict inputs in Python 3.7+ (insertion order)
- No external schema needed — the AtomicDOM structure is self-describing

---

## AtomicDOM program structure

Every micronaut ROM encodes this fixed shape:

```json
{
  "@program": "micronaut.<name>",
  "@kind":    "kuhul.micronaut.program.v1",
  "@import": [
    { "@from": "../../programs/api.manifest.json" },
    { "@from": "stdlib/pi.kuhul", "@use": ["pi_value"] }
  ],
  "@state": {
    "name":           "<name>",
    "fold":           "<Pop|Wo|Yax|Sek|Chen|Xul>",
    "category":       "<category>",
    "category_id":    <int>,
    "confidence":     <float>,
    "quant_tier":     "<none|fast|standard|quality>",
    "quant_tier_id":  <int>,
    "temperature":    <float>,
    "repeat_penalty": <float>,
    "repeat_last_n":  <int>
  },
  "@control": [
    { "@op": "FOLD_ENTER", "@fold": "<fold>", "@node": "micronaut.<name>" },
    { "@op": "GT",  "@args": ["$confidence", 0.7], "@out": "high_conf" },
    { "@op": "IF",  "@cond": "$high_conf",
      "@then": { "@op": "SET", "@key": "dispatch_priority", "@args": [2] },
      "@else": { "@op": "SET", "@key": "dispatch_priority", "@args": [1] } },
    { "@op": "CALL", "@fn": "dispatch_program",
      "@args": ["$name", "$quant_tier_id", "$dispatch_priority"],
      "@out": "result" },
    { "@op": "LOG",  "@args": ["[micronaut.<name>] fold=<fold> conf=$confidence"] },
    { "@op": "FOLD_EXIT", "@fold": "Chen", "@node": "micronaut.<name>",
      "@args": ["$result"] }
  ]
}
```

All micronauts share this 6-op control graph. `dispatch_program` resolves the
actual execution path from the registry metadata (llama.cpp inference for model
micronauts, `sidecar_uri` for native adapters).

---

## Lookup tables

### Category IDs (category_id in @state, column 5 of state_vec)

| category   | id |
|---|---|
| system     | 10 |
| specialist | 11 |
| persona    | 12 |
| meta       | 13 |
| stack      | 14 |
| fold       | 15 |
| native     | 16 |

### Quant tier IDs (quant_tier_id in @state, column 4 of state_vec)

| quant_tier | id | meaning |
|---|---|---|
| none       | 0  | no LLM quantization — native adapter |
| fast       | 1  | INT4 / Q4 — low latency |
| standard   | 2  | Q8 / INT8 — balanced |
| quality    | 3  | F16 / full — high fidelity |

### Op vocab (for tensor embedding via jrom_to_tensor.py)

| op          | id | | op         | id |
|---|---|---|---|---|
| ADD         | 1  | | IF         | 16 |
| SUB         | 2  | | LOOP       | 17 |
| MUL         | 3  | | BREAK      | 18 |
| DIV         | 4  | | RETURN     | 19 |
| SQRT        | 5  | | CALL       | 20 |
| MOD         | 6  | | IMPORT     | 21 |
| POW         | 7  | | DEFINE_OP  | 22 |
| ABS         | 8  | | LOG        | 23 |
| GT          | 9  | | SET        | 24 |
| LT          | 10 | | FOLD_ENTER | 25 |
| EQ          | 11 | | FOLD_EXIT  | 26 |
| NEQ         | 12 | | GPU_DISPATCH| 27 |
| AND         | 13 | | PHASE_TICK | 28 |
| OR          | 14 | | HASH       | 29 |
| NOT         | 15 | | READ       | 30 |
|             |    | | WRITE      | 31 |
|             |    | | EVAL       | 32 |

Standard micronaut op sequence (8 ops): `[25, 9, 16, 24, 24, 20, 23, 26]`  
= FOLD_ENTER, GT, IF, SET, SET, CALL, LOG, FOLD_EXIT

---

## State vector

Each registry entry in `dist/micronauts/jrom/registry.json` carries:

```
state_vec = [confidence, temperature, repeat_penalty, quant_tier_id, category_id]
```

This 5-float vector is the projection used by the tensor layer
(`jrom_to_tensor.py --state-tensor`) and by the SCXQ2 dispatch layer for
W·C·R scoring.

---

## Native adapter entries

Micronauts with `category = "native"` have `quant_tier = "none"` (id=0) and
additional fields in `micronauts/registry.json` that `dispatch_program` uses
to route to the sidecar instead of llama.cpp:

| name            | sidecar_uri                        | backing binary |
|---|---|---|
| native_factory  | sidecar://micronaut-factory/dispatch | micronaut_factory_core.dll + micronaut_evolution.dll |
| nnck            | sidecar://nnck/parse               | NeuralGrammar.Core.dll (.NET 8) |
| quantum         | sidecar://quantum/dispatch         | quantum_trinity / personality / grammar / microagents / asx_ram_v2 / asx_gemm |

Native adapters use `temperature=0.0`, `repeat_penalty=1.0`, `repeat_last_n=0`
(sampling is irrelevant — the DLL/exe handles its own computation).

---

## Build

```
python tools/micronaut_to_jrom.py
```

Optional flags:
- `--no-torch`     — skip `tensors.pt` stacked tensor output (requires torch + numpy)
- `--state-tensor` — also emit a float32 state tensor alongside the op tensor
- `--registry <path>` — alternate registry (default: `micronauts/registry.json`)
- `--out <dir>`    — alternate output dir (default: `dist/micronauts/jrom/`)

Outputs:
- `dist/micronauts/jrom/<name>.rom` — JROM binary (CBOR payload)
- `dist/micronauts/jrom/<name>.atomicdom.json` — human-readable AtomicDOM source
- `dist/micronauts/jrom/registry.json` — compiled index (bytes, n_ops, op_ids, state_vec)
- `dist/micronauts/jrom/tensors.pt` — stacked op-id tensor (n_micronauts × max_seq)

---

## Related

- `tools/micronaut_to_jrom.py` — compiler (registry → ROM + atomicdom.json + registry index)
- `tools/jrom_to_tensor.py` — ROM → tensor embeddings for model training
- `tools/micronaut_fieldgraph.py` — field graph builder over compiled ROMs
- `micronauts/registry.json` — authoritative source registry (28 entries)
- `dist/micronauts/jrom/registry.json` — compiled index (rebuilt on every run)
- JROM.md — AtomicDOM program system and @op semantics
- PROGRAMS.md — json_runtime stdlib and native primitive routing
