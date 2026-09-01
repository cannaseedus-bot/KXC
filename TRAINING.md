# Training Data and Failure Paths

This repository has two separate training concerns:

1. **Dataset streaming**: JSONL is converted to the native headed token-bin format and consumed one sequence at a time by the trainer.
2. **Model execution**: weights, gradients, optimizer state, and temporary buffers must fit the selected backend's residency limits.

These are related, but they are not the same failure mode.

## Canonical data contract

The native trainer does not train directly from a large JSONL file. It expects a headed `.bin` stream:

```text
uint32 sequence_count
uint32 sequence_length
int32[sequence_count * sequence_length] GPT-2 token ids
```

The conversion path is:

```text
AST/K'UHUL source
  -> tools/prep_ast_corpus.py
  -> JSONL {prompt, completion}
  -> tools/split_jsonl_chunks.py (optional)
  -> tools/jsonl_to_tokens.py
  -> headed token .bin
  -> gpt2_trainer.exe or tools/finetune_hf_tokenbin.py
```

The `.bin` file is the training-session input. Do not make the native trainer parse the entire JSONL corpus or load all source records into memory.

## AST coder preparation

Build the current AST corpus:

```powershell
python tools/prep_ast_corpus.py `
  --src programs\micronauts `
  --src tools `
  --out models\from_zero\coder_ast.jsonl `
  --fold-wrap
```

For bounded source chunks (useful for very large corpora):

```powershell
python tools/split_jsonl_chunks.py `
  --input models\from_zero\coder_ast.jsonl `
  --out-dir models\from_zero\coder_ast_chunks `
  --records 256
```

Tokenize either the single JSONL file or the chunk directory. The directory form processes `chunk_*.jsonl` files in order:

```powershell
python tools/jsonl_to_tokens.py `
  --input models\from_zero\coder_ast_chunks `
  --out models\from_zero\coder_ast_seq256_tokens.bin `
  --seq-len 256 `
  --profile ast-json
```

For a quick bounded smoke stream, add `--max-seq 8`.

Relative input and output paths resolve against the current checkout. The legacy `C:\Users\canna\.gpu_trainer\trainer` location is used only when a relative input does not exist in the checkout.

The existing external coder stream is also valid and should be reused when appropriate:

```text
E:\models\GPT2\coder_micronaut\coder_ast_json_seq256_tokens.bin
```

## Safe CPU smoke path

The token-bin Python path is the diagnostic baseline. It avoids Hugging Face tokenizer downloads and does not require distillation:

```powershell
python tools/finetune_hf_tokenbin.py `
  --base E:\models\GPT2\coder_micronaut\ultrachat_coder_skeleton_slerp_0p40.safetensors `
  --data models\from_zero\coder_ast_seq256_tokens.bin `
  --out models\from_zero\ultrachat_coder_ast_cpu_smoke `
  --max-seqs-per-file 8 `
  --block 256 `
  --steps 1 `
  --batch 1 `
  --lr 3e-5 `
  --threads 8 `
  --save-every 1
```

For a real run, remove `--steps 1` and raise `--max-seqs-per-file` deliberately. The current Python helper merges the selected bounded bins into a CPU tensor, so keep the cap appropriate for available RAM.

## Native trainer path and backend map

There are three distinct paths in this repository:

| Path | Backend | Role |
|---|---|---|
| `C:\Users\canna\.ASX.cpp\trainer\gpt2_trainer.exe` | D3D11 compute shaders / HLSL | Working legacy native GPT-2 trainer |
| `trainer\build\Release\gpt2_trainer.exe` | D3D11 compute shaders / HLSL | Checkout trainer; its full GPU Adam path can remove the HD 4600 device |
| `drivers\opencl_1_2` and OpenCL harnesses | OpenCL 1.2 only | Validated bounded kernels and small tensor/GPU lanes |
| `trainer\dashboard` and WebGL2 driver | WebGL2 GLSL ES 3.00 | Visualization/browser execution |
| `drivers\klsl\xvm_d3d12` | XVM/D3D12 HLSL/DXIL and historical WGSL assets | D3D12-only runtime, unavailable on HD 4600 |

The OpenCL smoke tests passing does not mean either `gpt2_trainer.exe` is currently an OpenCL trainer. The two executables above initialize `D3D11Engine` and load HLSL shaders. OpenCL 1.2 should remain the only OpenCL target; do not infer OpenCL 2.x or WebGPU from the dashboard/KLSL path.

The authoritative KLSL-to-backend map is `programs/klsl.targets.json`. The project-local compiler is `drivers/klsl/bin/klslc.exe`, launched with `drivers/klsl/klslc.cmd`. It records the native `klslc.exe` outputs (HLSL/XVM) and routes semantic opcodes to OpenCL C 1.2 or WebGL2 GLSL ES 3.00 dispatchers.

The working `.ASX.cpp` executable and the checkout executable are different binaries. Verify the resolved backend from their startup log (`[main] D3D11 adapter`) and run them from the directory containing their relative shader tree.

## Native trainer invocation

Run from the Release directory because shader paths are relative to that working directory:

```powershell
Push-Location trainer\build\Release
try {
  .\gpt2_trainer.exe `
    --model E:\models\GPT2\coder_micronaut\ultrachat_coder_skeleton_slerp_0p40.safetensors `
    --data C:\Users\canna\_khanary_inspect\models\from_zero\coder_ast_seq256_tokens.bin `
    --out C:\Users\canna\_khanary_inspect\models\from_zero\ultrachat_coder_ast_native.safetensors `
    --steps 1 --batch 1 --seq-len 256 --lr 3e-5 --save-every 1
}
finally { Pop-Location }
```

The native trainer streams the token `.bin`; this fixes dataset-side over-allocation. It does **not** make a large GPU optimizer footprint safe.

## Explicit error path: `0x887A0020`

Observed sequence:

```text
[adam] CreateQuery failed step=1 cqhr=0x887a0005 device_removed=0x887a0020
[sync] 4-byte probe hr=0x887a0005 NULL
CPU fallback: FAIL (exit -1073741819)
```

Interpretation:

- `0x887A0020` is `DXGI_ERROR_DEVICE_REMOVED`: the D3D device was lost.
- `0x887A0005` is the follow-on failure from issuing a query/readback against the removed device; it is not the original cause.
- `-1073741819` / `0xC0000005` is the process access-violation exit after the backend crash.
- The failure occurs at Adam/query creation after the model is loaded, not while reading JSONL.
- The 24-layer/1024-hidden coder checkpoint allocates weights, gradients, Adam `m`, Adam `v`, and temporary buffers. Four optimizer copies of a roughly 1.4 GB parameter set cannot be treated as resident on this HD 4600 configuration.

Hardware policy for this machine:

```text
128 MB dedicated VRAM
2 GB shared graphics memory reported by the driver
16 GB system RAM
CPU/DirectXMath = authoritative fallback
OpenCL 1.2 = bounded GPU lane for small active planes
D3D11 CS5 = optional compatible shader lane
D3D12/DirectML = unavailable on the HD 4600 feature level
```

Therefore:

- token bins solve data streaming;
- chunk limits solve bounded corpus preparation;
- CPU token-bin training is the safe baseline for the large coder checkpoint;
- native GPU Adam requires a smaller model, tiled/sharded optimizer state, or an explicit CPU-optimizer path;
- do not classify a device-removal crash as a tokenizer or JSONL failure.

## Verification checklist

1. Validate JSONL records while splitting.
2. Confirm the token-bin header and file size before training.
3. Run an 8-sequence, one-step CPU smoke test.
4. Validate the saved SafeTensors output before promoting it.
5. Only then try the native trainer with a model/residency budget that fits the selected backend.

The Semantic Cube/KUBE trainer is separate from this AST coder path. Its model bindings live under `dist\GPU-TRAINER\models`; do not mix its CSO shader assets with the GPT2 coder token stream.

## π-KUHUL Semantic Cube adapter

### Bounded weak-gradient targeting

The π-KUHUL trainer can selectively reinforce parameter tensors whose
post-clipping gradient RMS is below a threshold. This is tensor targeting,
not loss targeting: a batch loss below `1.0` does not imply that every
parameter lane is below `1.0`. The default `smoothstep` curve approaches a
maximum multiplier only for very weak gradients and leaves gradients at or
above the threshold unchanged.

Example for a conservative targeted run:

```powershell
python tools\pi_kuhul_train.py `
  --data models\from_zero\coder_ast_chunks\kson_parser_tree.train.jsonl `
  --tokenizer glyph --steps 2000 --batch 1 --block 128 `
  --gradient-target-threshold 1.0 `
  --gradient-target-gain 0.25 `
  --gradient-target-curve smoothstep `
  --ckpt-dir models\from_zero\trained\pi-kuhul-kson-targeted `
  --shard-dir models\from_zero\trained\pi-kuhul-kson-targeted\shards
```

The log reports `targeted/total tensors` and the mean multiplier. Keep the
gain small (normally `0.10`–`0.25`) and inspect validation/replay output
before increasing it. This correction is compatible with the CPU/OpenCL
Semantic Cube adapter and does not claim that WebGL2 is executing the
training step.

The π-KUHUL checkpoint is loaded by the Python π-KUHUL runtime. The
Semantic Cube adapter is now available in both training and inference:

```powershell
# Training: PyTorch remains the gradient authority; dispatch a CL1.2 field
# projection every 50 steps and record the result for ARC replay.
python tools\pi_kuhul_train.py `
  --data models\from_zero\coder_ast_chunks\kson_parser_tree.train.jsonl `
  --tokenizer glyph --steps 2000 --batch 1 --block 128 --lr 1e-4 `
  --semantic-backend opencl --semantic-every 50 `
  --semantic-loss-weight 1e-6 `
  --semantic-replay scratch\pi_kuhul_semantic_replay.jsonl

# Inference: load the same checkpoint and route observations through the
# Semantic Cube adapter. Use --semantic-backend cpu when OpenCL is unavailable.
python tools\pi_kuhul_serve.py `
  --ckpt models\from_zero\trained\pi-kuhul-kson-parser-split\final.pt `
  --tokenizer auto --semantic-backend opencl --semantic-every 1 --port 9012
```

The adapter contracts the runtime artifacts rather than pretending that a
Python tensor call is a shader dispatch:

| Adapter stage | Active artifact |
|---|---|
| Semantic projection | `dist/GPU-TRAINER/semantic_cube.hlsl` contract |
| OpenCL 1.2 field pass | `drivers/opencl_1_2/matrix5x4.cl` via `scratch/opencl_matrix5x4_bridge.exe` |
| WebGL2 equivalent contract | `drivers/webgl2/gravity_corrective.mjs` |
| Replay | `scratch/pi_kuhul_semantic_replay.jsonl` |

If the OpenCL bridge fails, the adapter records the fallback and continues on
the CPU. The adapter signal is bounded: training receives a small optional
field-energy regularizer, while inference applies only a small logit gain.
The model checkpoint itself remains π-KUHUL-native and is not converted to
GGUF by this path.

### Shard sidecar dispatch

The router assignment shown in the training log is also passed to the
Semantic Cube adapter. With the OpenCL backend enabled, each routed shard is
projected as its own bounded sidecar slice. A successful run reports:

```text
semantic: opencl dispatch=opencl_1_2.matrix5x4_matmul[x6]
sidecar targets: [0,1,2,3,4,5] (6 bounded slices)
```

This is genuine OpenCL 1.2 sidecar work on the selected GPU. PyTorch still
owns the forward graph, loss, backward pass, gradient targeting, and
optimizer. WebGL2 remains a host/WebView2 bounded-slice backend until its
browser bridge is explicitly enabled.

Every completed π-KUHUL run now writes `result.json` beside the final
checkpoint by default. Override it with `--result-json`. The record includes
the resolved configuration, completed steps, token count, shard distribution,
sidecar dispatch count, final metrics, and final checkpoint/shard paths.

## GPT-OSS teacher distillation

`tools/oss_distillation.py` trains the GPT-2-shaped `from_zero` student with
GPT-OSS completions. It now appends canonical MX-2 promotion traces to
`dist/MX-2/io/distillation.jsonl` in addition to its timestamped diagnostic
logs. Use `--trace-jsonl` to select another trace file.

The trace is a candidate record for MX-2/Evolution/CM-1 review; writing it does
not update a helper model or promote weights. BMP observations and JROM control
ROMs are separate artifacts and are never treated as text-training examples.

## INT4/INT8 coder-skeleton artifacts

`models\from_zero\coder_skeleton.safetensors` remains the training master. Its
SCXQ2/KQZ exports are:

```text
coder_skeleton.q8.kqz  407,206,312 bytes  (INT8)
coder_skeleton.q4.kqz  216,339,072 bytes  (INT4, group 64)
```

These are for the SCXQ2/JROM resident loader and are not GGUF. A llama.cpp
test requires a separate GGUF export and a present `llama-quantize.exe`; the
current checkout does not contain that quantizer binary.
# Fold-to-tensor ownership

The canonical mapping is [programs/kuhul.fold-tensor-map.json](programs/kuhul.fold-tensor-map.json).
It assigns every phase to explicit tensor groups:

- `Pop`: context, checkpoint, shard IDs, and sidecar input buffers.
- `Wo`: attention and token/fold routing tensors.
- `Yax`: expert admission and bounded sidecar selection.
- `Sek`: forward tensors, logits, and six bounded OpenCL projections.
- `Chen`: semantic confidence, residual bounds, gradient targeting, and validation.
- `Xul`: checkpoints, replay JSONL, and `result.json`.

The map also binds the existing bone/cluster/gravity shaders to their phase. The
PyTorch optimizer remains the gradient authority; shader/OpenCL paths are bounded
observation or correction sidecars.

References in the map are typed: `tensor:*` and `group:*` identify tensor data,
`state:*` identifies runtime state, `artifact:*` identifies a committed file, and
`stream:*` identifies append-only replay output. This lets a runtime derive fold
dependencies without treating checkpoints or JSONL logs as tensors.

WebGL2 is an inference projection provider only. It may score routes and emit a
bounded correction from `routed_hidden`, but it cannot update model weights,
optimizer state, or authoritative gradients.

For resumed runs on space-constrained disks, use `--save-every 0 --shard-every 0`
to emit only the final checkpoint and final shard set. The trainer also skips the
initial checkpoint boundary when resuming, preventing an unnecessary duplicate
snapshot at the starting step.
