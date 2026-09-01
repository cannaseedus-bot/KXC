#!/usr/bin/env python3
"""
gguf_to_xshard.py -- Convert a .gguf model directly to XSHARD/1 format.

Reads tensors from GGUF via gguf-py (memory-mapped, no full-model load).
Writes raw quantized bytes into xshard — no dequantize step.
Fold assignment uses the same pattern table as safetensors_to_xshard.py.

Usage:
  python tools/gguf_to_xshard.py \\
      --input  model.gguf \\
      --output model.xshard \\
      [--gguf-py  path/to/gguf-py]    # default: auto-locate from llama.cpp
      [--max-shard-mb 256]
      [--model <name>]
      [--dry-run]

GGUF dtypes stored verbatim (no dequantize):
  F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q6_K, Q4_K, Q5_K, Q8_K ...
  xshard_adapt reads dtype to know how to interpret raw bytes.
"""

import argparse
import hashlib
import json
import math
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Locate gguf-py ─────────────────────────────────────────────────────────────

_GGUF_PY_CANDIDATES = [
    Path(r"C:\Users\canna\.ASX.cpp\llama-b9968-bin-win-cpu-x64\llama.cpp\gguf-py"),
]

def _find_gguf_py(hint: str | None) -> Path:
    if hint:
        p = Path(hint)
        if (p / "gguf").exists():
            return p
        raise FileNotFoundError(f"--gguf-py {hint}: no 'gguf' sub-package found")
    for c in _GGUF_PY_CANDIDATES:
        if c.exists() and (c / "gguf").exists():
            return c
    raise FileNotFoundError(
        "gguf-py not found. Pass --gguf-py <path> pointing to the directory "
        "that contains the 'gguf/' sub-package."
    )


# ── Constants (shared with safetensors_to_xshard) ──────────────────────────────

MAGIC        = b'XSHD'
FOOTER_MAGIC = 0x44485358
VERSION      = 1
ALIGNMENT    = 64

# GGML type name → xshard dtype string.
# Raw quantized bytes are stored verbatim; xshard_adapt knows the block layout.
_GGML_TYPE_TO_DTYPE: dict[int, str] = {
    0:  "F32",
    1:  "F16",
    2:  "Q4_0",
    3:  "Q4_1",
    6:  "Q5_0",
    7:  "Q5_1",
    8:  "Q8_0",
    9:  "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    31: "Q4_0_4_4",
    32: "Q4_0_4_8",
    33: "Q4_0_8_8",
    34: "TQ1_0",
    35: "TQ2_0",
}

# ── Fold assignment (identical to safetensors_to_xshard.py) ───────────────────

import re

_FOLD_PATTERNS = [
    # ── HuggingFace / GPT-2 naming ────────────────────────────────────────────
    (re.compile(r'(^|\.)(wte|wpe|embed|token_embed|pos_embed)(\.|\b)'),
     'Pop', 0.0),
    (re.compile(r'(^|\.)mlp\.(c_fc|fc_1|gate|up_proj)(\.|\b)'),
     'Wo', math.pi / 3),
    (re.compile(r'(^|\.)mlp\.(c_proj|fc_2|down|down_proj)(\.|\b)'),
     'Yax', 2 * math.pi / 3),
    (re.compile(r'(^|\.)attn\.(c_attn|c_proj|q|k|v|q_proj|k_proj|v_proj|o_proj|out_proj)(\.|\b)'),
     'Sek', math.pi),
    # ── GGUF / llama.cpp naming (blk.N.*) ─────────────────────────────────────
    (re.compile(r'(^|\.)token_embd(\.|\b)'),
     'Pop', 0.0),
    (re.compile(r'(^|\.)blk\.\d+\.(ffn_up|ffn_gate)(\.|\b)'),
     'Wo', math.pi / 3),
    (re.compile(r'(^|\.)blk\.\d+\.ffn_down(\.|\b)'),
     'Yax', 2 * math.pi / 3),
    (re.compile(r'(^|\.)blk\.\d+\.(attn_q|attn_k|attn_v|attn_output)(\.|\b)'),
     'Sek', math.pi),
    # ── Normalisation (both naming conventions) ────────────────────────────────
    (re.compile(r'(^|\.)(ln_[0-9f]|ln_f|layernorm|layer_norm|norm)(\.|\b)'),
     'Chen', 4 * math.pi / 3),
    (re.compile(r'_norm(\.|\b)'),
     'Chen', 4 * math.pi / 3),
    (re.compile(r'(^|\.)bias(\.|\b)'),
     'Chen', 4 * math.pi / 3),
    # ── Output head ────────────────────────────────────────────────────────────
    (re.compile(r'(^|\.)(lm_head|output\.weight)(\.|\b)'),
     'Xul', 5 * math.pi / 3),
]

_PHASE_ANGLES = {
    'Pop':  0.0,
    'Wo':   math.pi / 3,
    'Yax':  2 * math.pi / 3,
    'Sek':  math.pi,
    'Chen': 4 * math.pi / 3,
    'Xul':  5 * math.pi / 3,
}


def assign_fold(tensor_name: str) -> tuple[str, float]:
    name = tensor_name.lower()
    for pattern, fold, angle in _FOLD_PATTERNS:
        if pattern.search(name):
            return fold, round(angle, 6)
    return 'Pop', 0.0


# ── Shard ID (same logic as safetensors_to_xshard) ───────────────────────────

def _shard_id(tensor_name: str) -> str:
    parts = tensor_name.split('.')
    skip = {'transformer', 'model', 'weight', 'bias'}
    kept = [p for p in parts if p not in skip]
    return '.'.join(kept) if kept else tensor_name


# ── Padding ───────────────────────────────────────────────────────────────────

def pad_up(n: int, align: int = ALIGNMENT) -> int:
    return (n + align - 1) // align * align


# ── Fold summary ──────────────────────────────────────────────────────────────

def print_fold_summary(shards: list[dict]) -> None:
    from collections import Counter
    counts = Counter(s['fold'] for s in shards)
    mb     = {f: sum(s['nbytes'] for s in shards if s['fold'] == f) / 1024 / 1024
              for f in counts}
    print('  fold distribution:')
    for fold in ['Pop', 'Wo', 'Yax', 'Sek', 'Chen', 'Xul']:
        if fold in counts:
            angle = _PHASE_ANGLES[fold]
            print(f'    {fold:<5} angle={angle:.4f}rad  '
                  f'shards={counts[fold]:3d}  '
                  f'data={mb[fold]:7.1f} MB')


# ── Manifest ──────────────────────────────────────────────────────────────────

def build_manifest(shards: list[dict], model_name: str, arch: str,
                   state_start: int = 0, data_start: int = 0) -> dict:
    has_sub = any(s['shard_count'] > 1 for s in shards)
    flags = 0x01 | 0x04
    if has_sub:
        flags |= 0x02

    records = []
    for s in shards:
        rec = {
            'id':          s['id'],
            'seq':         s['seq'],
            'tensor_name': s['tensor_name'],
            'fold':        s['fold'],
            'phase_angle': s['phase_angle'],
            'shape':       s['shape'],
            'dtype':       s['dtype'],
            'offset':      s.get('offset', 0),
            'nbytes':      s.get('nbytes', 0),
            'sha256':      s.get('sha256', '0' * 64),
            'shard_of':    s['shard_of'],
            'shard_index': s['shard_index'],
            'shard_count': s['shard_count'],
        }
        records.append(rec)

    return {
        '@kind':       'xshard/1',
        'version':     1,
        'created':     datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'model':       model_name,
        'arch':        arch,
        'source':      'gguf',
        'flags':       flags,
        'alignment':   ALIGNMENT,
        'n_shards':    len(shards),
        'state_start': state_start,
        'data_start':  data_start,
        'shards':      records,
    }


def serialize_manifest(m: dict) -> bytes:
    return json.dumps(m, separators=(',', ':')).encode('utf-8')


def resolve_offsets(manifest: dict, n_shards: int) -> tuple[int, int]:
    for _ in range(3):
        mb = serialize_manifest(manifest)
        state_start = pad_up(16 + len(mb))
        data_start  = pad_up(state_start + n_shards)
        if manifest['state_start'] == state_start and manifest['data_start'] == data_start:
            break
        manifest['state_start'] = state_start
        manifest['data_start']  = data_start
    return state_start, data_start


# ── Writer ────────────────────────────────────────────────────────────────────

def write_xshard(output_path: Path, shards: list[dict], manifest: dict,
                 data_start: int, state_start: int, dry_run: bool = False) -> None:
    manifest_bytes = serialize_manifest(manifest)
    n_shards = len(shards)

    if dry_run:
        print(f'  manifest_len  : {len(manifest_bytes):,} bytes')
        print(f'  state_start   : {state_start:,}')
        print(f'  data_start    : {data_start:,}')
        total = data_start + sum(pad_up(s['nbytes']) for s in shards) + 8
        print(f'  estimated size: {total:,} bytes ({total/1024/1024:.1f} MB)')
        return

    with open(output_path, 'wb') as dst:
        dst.write(MAGIC)
        dst.write(struct.pack('<H', VERSION))
        dst.write(struct.pack('<H', manifest['flags']))
        dst.write(struct.pack('<Q', len(manifest_bytes)))
        dst.write(manifest_bytes)

        cur = 16 + len(manifest_bytes)
        dst.write(b'\x00' * (state_start - cur))
        dst.write(bytes([0x00] * n_shards))
        cur = state_start + n_shards
        dst.write(b'\x00' * (data_start - cur))

        running_offset = 0
        for i, s in enumerate(shards):
            raw = bytes(s['_raw'])          # numpy uint8 array → bytes
            sha256 = hashlib.sha256(raw).hexdigest()

            s['sha256'] = sha256
            s['offset'] = running_offset
            manifest['shards'][i]['sha256'] = sha256
            manifest['shards'][i]['offset'] = running_offset

            dst.write(raw)
            padded = pad_up(len(raw))
            dst.write(b'\x00' * (padded - len(raw)))
            running_offset += padded

            if (i + 1) % 20 == 0 or i + 1 == n_shards:
                pct = (i + 1) / n_shards * 100
                print(f'  [{i+1:4d}/{n_shards}] {pct:5.1f}%  {s["id"]}  fold={s["fold"]}',
                      flush=True)

        # Rewrite manifest with real sha256s (lengths are invariant)
        final_mb = serialize_manifest(manifest)
        assert len(final_mb) == len(manifest_bytes), \
            f'Manifest size changed ({len(manifest_bytes)} → {len(final_mb)})'
        dst.seek(16)
        dst.write(final_mb)
        dst.seek(0, 2)

        dst.write(struct.pack('<I', n_shards))
        dst.write(struct.pack('<I', FOOTER_MAGIC))

    print(f'  wrote {output_path}')


def write_meta_json(output_path: Path, manifest: dict) -> None:
    meta_path = output_path.with_suffix(output_path.suffix + '.meta.json')
    meta = {
        '@kind':  'xshard.meta/1',
        'model':  manifest['model'],
        'source': 'gguf',
        'passes': [],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'  wrote {meta_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Convert .gguf → XSHARD/1 (fold-tagged, streaming-native)')
    ap.add_argument('--input',        required=True,  help='Input .gguf file')
    ap.add_argument('--output',       required=True,  help='Output .xshard file')
    ap.add_argument('--gguf-py',      default=None,   help='Path to gguf-py package dir')
    ap.add_argument('--max-shard-mb', type=float, default=256.0,
                    help='Sub-shard tensors larger than this MB (F32/F16 only). Default: 256')
    ap.add_argument('--model',        default='',     help='Model name in manifest')
    ap.add_argument('--arch',         default='gguf', help='Architecture hint. Default: gguf')
    ap.add_argument('--dry-run',      action='store_true')
    args = ap.parse_args()

    # Locate and import gguf-py
    gguf_py_dir = _find_gguf_py(args.gguf_py)
    sys.path.insert(0, str(gguf_py_dir))
    from gguf import GGUFReader  # type: ignore

    import numpy as np

    input_path  = Path(args.input)
    output_path = Path(args.output)
    max_bytes   = int(args.max_shard_mb * 1024 * 1024)
    model_name  = args.model or input_path.stem

    if not input_path.exists():
        print(f'error: {input_path} not found', file=sys.stderr)
        sys.exit(1)

    print(f'[xshard] reading {input_path}')
    reader = GGUFReader(str(input_path))
    tensors = reader.tensors
    print(f'  {len(tensors)} tensors')

    # Build shard plan
    shards: list[dict] = []
    seq = 0
    skipped = 0

    for t in tensors:
        dtype_id  = int(t.tensor_type)
        dtype_str = _GGML_TYPE_TO_DTYPE.get(dtype_id)
        if dtype_str is None:
            print(f'  [skip] {t.name}: unknown GGML type {dtype_id}', file=sys.stderr)
            skipped += 1
            continue

        fold, angle = assign_fold(t.name)
        shape  = [int(x) for x in t.shape]   # GGUF shape is already (rows, cols, ...)
        nbytes = int(t.n_bytes)
        raw    = t.data                        # numpy uint8 memmap view

        # Sub-shard only for float types where we can split rows cleanly
        float_types = {'F32', 'F16', 'BF16'}
        can_split   = dtype_str in float_types and len(shape) >= 2 and nbytes > max_bytes

        if not can_split:
            shards.append({
                'id':          _shard_id(t.name),
                'seq':         seq,
                'tensor_name': t.name,
                'fold':        fold,
                'phase_angle': angle,
                'shape':       shape,
                'dtype':       dtype_str,
                'nbytes':      nbytes,
                'shard_of':    t.name,
                'shard_index': 0,
                'shard_count': 1,
                '_raw':        raw,
            })
            seq += 1
        else:
            # Split along axis 0
            n_rows      = shape[0]
            elem_bytes  = nbytes // n_rows
            rows_per    = max(1, max_bytes // elem_bytes)
            n_splits    = math.ceil(n_rows / rows_per)
            raw_bytes   = bytes(raw)

            for split_i in range(n_splits):
                r0 = split_i * rows_per
                r1 = min(r0 + rows_per, n_rows)
                sub_shape = list(shape)
                sub_shape[0] = r1 - r0
                chunk = raw_bytes[r0 * elem_bytes : r1 * elem_bytes]

                shards.append({
                    'id':          f'{_shard_id(t.name)}.s{split_i}',
                    'seq':         seq,
                    'tensor_name': t.name,
                    'fold':        fold,
                    'phase_angle': angle,
                    'shape':       sub_shape,
                    'dtype':       dtype_str,
                    'nbytes':      len(chunk),
                    'shard_of':    t.name,
                    'shard_index': split_i,
                    'shard_count': n_splits,
                    'shard_axis':  0,
                    'shard_slice': [r0, r1],
                    '_raw':        chunk,
                })
                seq += 1

    print(f'  {len(shards)} shards'
          + (f', {skipped} skipped' if skipped else '')
          + f', {sum(1 for s in shards if s["shard_count"]>1)} sub-sharded')
    print_fold_summary(shards)

    # Assign offsets
    running = 0
    for s in shards:
        s['offset'] = running
        running += pad_up(s['nbytes'])

    manifest = build_manifest(shards, model_name, args.arch)
    for i, s in enumerate(shards):
        manifest['shards'][i]['offset'] = s['offset']
        manifest['shards'][i]['nbytes'] = s['nbytes']

    state_start, data_start = resolve_offsets(manifest, len(shards))
    print(f'[xshard] layout: '
          f'manifest={len(serialize_manifest(manifest)):,}B  '
          f'state_start={state_start:,}  '
          f'data_start={data_start:,}')

    if args.dry_run:
        print('[xshard] dry-run — no files written')
        write_xshard(output_path, shards, manifest, data_start, state_start, dry_run=True)
        return

    print(f'[xshard] writing {output_path} ...')
    write_xshard(output_path, shards, manifest, data_start, state_start)
    write_meta_json(output_path, manifest)

    total = output_path.stat().st_size
    print(f'[xshard] done — {total:,} bytes ({total/1024/1024:.1f} MB)')


if __name__ == '__main__':
    main()
