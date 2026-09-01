# XSHARD/1 — Streaming Tensor Shard Format

A fold-tagged, streaming-native binary weight format. Each shard is a
self-contained training unit — no need to load or slice a full tensor
at runtime. The streaming pattern is baked into the format rather than
bolted on top of SafeTensors at inference time.

---

## Design goals

| Goal | How |
|------|-----|
| Streaming-native | Shards stored contiguously; reader seeks directly to `data_start + shard.offset` |
| Fold-tagged | Every shard carries a K'UHUL phase and angle — training routes semantically |
| In-place update | State block mutable without rewriting manifest; shard data writable at declared offset |
| Sub-tensor sharding | One tensor → N shards along any axis; VRAM ceiling never exceeded |
| Format-portable | Same sweep command as safetensors; just change `--output` extension |
| Integrity | SHA-256 per shard; footer n_shards cross-check |

---

## Binary layout

```
Offset    Size         Field
────────  ───────────  ──────────────────────────────────────────────
0         4            Magic: bytes 0x58 0x53 0x48 0x44  ("XSHD")
4         2            Version: u16 LE = 1
6         2            Flags: u16 LE  (see Flags table)
8         8            manifest_len: u64 LE
16        manifest_len Manifest JSON (UTF-8)
─         pad          Pad to next 64-byte boundary
state_start  n_shards  State block: one byte per shard ordered by seq
─         pad          Pad to next 64-byte boundary
data_start   variable  Shard data blocks (each 64-byte aligned)
EOF-8     4            Footer: u32 LE = n_shards
EOF-4     4            Footer magic: u32 LE = 0x44485358  (same "XSHD" bytes as header magic)
```

`state_start` and `data_start` are stored in the manifest so readers
do not need to compute padding offsets.

### Flags (u16, bit field)

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | FOLD_TAGGED | All shards have `fold` + `phase_angle` set |
| 1 | SUB_SHARDED | At least one tensor split across multiple shards |
| 2 | STATE_MUTABLE | State block is valid and may be updated in-place |
| 3–15 | — | Reserved, must be 0 |

### State block

One byte per shard, indexed by `shard.seq`. Can be updated in-place
(seek to `state_start + seq`, write one byte) without touching the manifest
or shard data.

| Value | Status |
|-------|--------|
| 0x00 | pending — not yet trained |
| 0x01 | trained — pass complete |
| 0x02 | error — pass failed |
| 0xFF | locked — reserved, do not train |

---

## Manifest JSON

Stored at bytes 16–(16+manifest_len). Fixed after file creation —
do not rewrite. All offsets in `shards[].offset` are relative to `data_start`.

```json
{
  "@kind": "xshard/1",
  "version": 1,
  "created": "2026-08-16T00:00:00Z",
  "model": "from_zero_v0.6",
  "arch": "gpt2",
  "n_layer": 12,
  "n_embd": 768,
  "n_head": 12,
  "vocab_size": 50270,
  "dtype_default": "F32",
  "alignment": 64,
  "n_shards": 48,
  "state_start": 4096,
  "data_start": 4224,
  "shards": [ ...ShardRecord... ]
}
```

### ShardRecord fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | ✓ | Unique shard ID within file (e.g. `h.0.attn.c_attn`) |
| `seq` | integer ≥ 0 | ✓ | Sequential index into state block |
| `tensor_name` | string | ✓ | Original tensor key (e.g. `transformer.h.0.attn.c_attn.weight`) |
| `fold` | PhaseName | ✓ (if FOLD_TAGGED) | K'UHUL phase owning this shard |
| `phase_angle` | float | ✓ (if FOLD_TAGGED) | Phase angle in radians (see table) |
| `shape` | int[] | ✓ | Full shape of this shard's data |
| `dtype` | DtypeName | ✓ | Data type (see table) |
| `stride` | int[] | — | Row-major strides; omit for C-contiguous |
| `offset` | integer ≥ 0 | ✓ | Byte offset from `data_start` |
| `nbytes` | integer > 0 | ✓ | Byte count of raw shard data |
| `sha256` | string | ✓ | SHA-256 hex of raw bytes at creation time |
| `shard_of` | string | — | Parent tensor name (if sub-sharded) |
| `shard_index` | integer ≥ 0 | — | This shard's index within the parent tensor |
| `shard_count` | integer ≥ 1 | — | Total shards for the parent tensor |
| `shard_axis` | integer ≥ 0 | — | Axis the parent tensor is split along |
| `shard_slice` | [int, int] | — | `[start, end)` in shard_axis dimension |
| `training_steps` | integer ≥ 0 | — | Steps applied so far (updated in .meta.json) |
| `best_loss` | float \| null | — | Best loss seen during training |

### DtypeName

`"F32"` · `"BF16"` · `"F16"` · `"INT8"` · `"Q8_0"` · `"Q4_K"`

---

## Fold assignment

Deterministic from tensor name — applied at file creation time.

| Pattern | Fold | Phase angle | Semantic role |
|---------|------|-------------|---------------|
| `wte`, `wpe`, `embed` | Pop | 0.0 | Input embedding — observe |
| `mlp.c_fc`, `mlp.fc_1`, `mlp.gate` | Wo | π/3 ≈ 1.0472 | MLP expansion — schedule |
| `mlp.c_proj`, `mlp.fc_2`, `mlp.down` | Yax | 2π/3 ≈ 2.0944 | MLP projection — branch |
| `attn.c_attn`, `attn.q`, `attn.k`, `attn.v`, `attn.c_proj` | Sek | π ≈ 3.1416 | Attention — execute |
| `ln_1`, `ln_2`, `ln_f`, `layernorm`, `norm`, `bias` | Chen | 4π/3 ≈ 4.1888 | Normalization — verify |
| `lm_head`, `output.weight` | Xul | 5π/3 ≈ 5.2360 | Logit projection — emit |
| (no match) | Pop | 0.0 | Fallback — observe |

---

## Sub-tensor sharding

When a tensor exceeds the VRAM ceiling, split it into N shards along one axis.
All sub-shards share the same `shard_of` and `shard_count`; differ in `shard_index`
and `shard_slice`. Readers reconstruct the full tensor by concatenating shards in
`shard_index` order along `shard_axis`.

Example — `transformer.h.0.attn.c_attn.weight` shape `[768, 2304]`
split into 3 shards along axis 1:

```json
{ "id": "h.0.attn.c_attn.s0", "shard_of": "transformer.h.0.attn.c_attn.weight",
  "shard_index": 0, "shard_count": 3, "shard_axis": 1,
  "shard_slice": [0, 768], "shape": [768, 768], "fold": "Sek", ... },
{ "id": "h.0.attn.c_attn.s1", ..., "shard_index": 1,
  "shard_slice": [768, 1536], "shape": [768, 768], ... },
{ "id": "h.0.attn.c_attn.s2", ..., "shard_index": 2,
  "shard_slice": [1536, 2304], "shape": [768, 768], ... }
```

---

## Streaming training flow

```
1. Read bytes 0–15  → magic, version, flags, manifest_len
2. Read manifest    → n_shards, state_start, data_start, shards[]
3. For each shard (in seq order, or filtered by fold):
   a. Read state_start + shard.seq  → skip if 0x01 (trained)
   b. Seek to data_start + shard.offset
   c. Read shard.nbytes → GPU buffer
   d. Run WebGL2 training pass (token_bin, configured steps/lr)
   e. Write adapted weights back to data_start + shard.offset  (in-place)
   f. Seek to state_start + shard.seq, write 0x01
   g. Append entry to .xshard.meta.json
4. Verify footer n_shards == manifest.n_shards
```

Fold-filtered training — adapt only Sek (attention) shards:

```powershell
kuhul-es train-webgl2-sweep --input model.xshard --output model.xshard `
  --fold-filter Sek --token-bin tokens.bin `
  --train-dim 768 --steps-per-shard 64 --batch 16 --lr 0.00025 `
  --browser auto
```

---

## Sidecar files

| File | Role |
|------|------|
| `model.xshard` | Primary binary — immutable manifest + mutable state + shard data |
| `model.xshard.meta.json` | Mutable training log: per-shard steps/loss/sha256/timestamp |
| `model.xshard.lock` | Advisory write lock (presence = another process is writing) |

### `.xshard.meta.json` structure

```json
{
  "@kind": "xshard.meta/1",
  "model": "from_zero_v0.6",
  "passes": [
    { "seq": 0, "id": "h.0.attn.c_attn", "steps": 64, "best_loss": 0.421,
      "sha256": "...", "ts": "2026-08-16T01:00:00Z" }
  ]
}
```

---

## Creation — SafeTensors → XSHARD

```powershell
python tools/safetensors_to_xshard.py `
  --input  model.safetensors `
  --output model.xshard `
  [--max-shard-mb 256]   # sub-shard tensors larger than this (default: 256)
  [--shard-axis 0]       # split axis; axis 0 keeps data contiguous (default)
  [--arch gpt2]          # architecture hint for fold assignment
  [--model <name>]       # model identifier written into manifest
  [--dry-run]            # print plan without writing
```

The tool:
1. Reads SafeTensors header → tensor names, shapes, dtypes, data offsets
2. Assigns fold by name pattern table (deterministic, no data loaded)
3. Plans shard layout — sub-shards tensors over `--max-shard-mb` along `--shard-axis`
4. Resolves `state_start` / `data_start` (convergence loop, ≤2 iterations)
5. Streams shard data from input → output (no full-model memory residency)
6. SHA-256 each shard during write; patches manifest records in memory
7. Writes `.xshard.meta.json` stub for training log

---

## Relation to other formats

| Format | Streaming | Fold-tagged | Mutable state | Quantized |
|--------|-----------|-------------|---------------|-----------|
| SafeTensors | external bolt-on | no | no | no |
| XSHARD/1 | native | yes | yes (state block) | no (use dtype) |
| SCXQDDS | native | no | no | yes (INT8/Q8) |
| SCXQDDS+XSHARD | native | yes | yes | yes |

SCXQDDS/DDS is the quantized analogue. An `.xshard` file with `dtype: "Q8_0"`
or `dtype: "Q4_K"` shards bridges both worlds.

---

## Schema

`dist/kuhul-es/schemas/xshard-1.json` — JSON Schema for the manifest.
`dist/kuhul-es/grammars/xshard-1.ebnf` — EBNF for the manifest grammar.
