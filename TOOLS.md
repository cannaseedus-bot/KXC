# TOOLS — K'UHUL Tool Dispatch & Inference

This doc covers the **tool dispatch stack** (forge executor, verbs, interceptor
loop, verifier harness) and the **inference path** (GPU fix), plus the G003
tool-call fine-tune pipeline. All of it lives in `kuhul-server.cjs` (the
`:8764` gateway) and `tools/`.

---

## 1. The tool dispatch stack

```
model output
  → parse (structured tool_calls OR raw <tool_call>verb|subject|object</tool_call>)
  → dispatch (forge tool / verb)
  → execute for real (spawn coder binary/script, sandboxed + timed out)
  → inject {role:'tool'} result back
  → model continues → final answer
```

### `/v1/forge` — the executor (12 real tools)

Each tool maps to an actual binary/script that the gateway **spawns as a
subprocess** (not hallucinated). File/dir args are sandboxed to the project
root; unknown tools → 400; 30s subprocess timeout.

| Tool | Backend | Args |
|---|---|---|
| `review` | `micronaut_code_reviewer.exe` | file |
| `review-dir` | `micronaut_code_reviewer.exe` | dir |
| `diff` | `micronaut_code_reviewer.exe` | old, new |
| `refactor` | `micronaut_code_reviewer.exe` | file, goal |
| `optimize` | `micronaut_code_reviewer.exe` | file, metric |
| `todos` | `micronaut_code_reviewer.exe` | file |
| `document` | `micronaut_code_reviewer.exe` | file |
| `test` | `micronaut_code_reviewer.exe` | file |
| `explain` | `micronaut_code_reviewer.exe` | file |
| `github-review` | `micronaut_code_reviewer.exe` | file |
| `v6-hashes` | `compute-v6-hashes.js` (node) | dir |
| `todos-universal` | `extract-todos-universal.js` (node) | dir |

```json
POST /v1/forge  { "tool": "review", "args": { "file": "dist/micronaut-coder/src/main.cpp" } }
GET  /v1/forge/tools   // discover the registry (for tool-calling models)
```

### `/v1/verbs` — SVO action vocabulary (KXML `verb` made real)

The KXML registry's `verb`/`action`/`bot`/`skill` entries had `glyph_token:
null` (unwired). Verbs give a model a light SVO interface — pick a verb + a
subject, the gateway resolves it to the real executable:

```json
GET  /v1/verbs
POST /v1/forge/verb  { "name": "review",   "subject": "dist/micronaut-coder/src/main.cpp" }
POST /v1/forge/verb  { "name": "refactor", "subject": "...", "object": "reduce duplication" }
```

Verbs (10): `review`, `todos`, `explain`, `document`, `test`, `github-review`,
`optimize` (object=metric), `refactor` (object=goal), `v6-hashes`,
`todos-universal`.

### `/v1/forge/chat` — the tool interceptor LOOP

Closes the "model calls the tool but the result isn't written back" gap. Runs
until the model stops emitting calls or `max_iters`:

```json
POST /v1/forge/chat {
  "model_url": "http://127.0.0.1:9002/v1/chat/completions",
  "messages": [{ "role": "user", "content": "review dist/micronaut-coder/src/main.cpp" }],
  "max_iters": 5
}
```

**`mode: "completion"` (raw GPT-2)** — for the fine-tuned mm-toolcall GPT-2,
which has no chat handler. It renders messages to the `USER:/ASSISTANT:/TOOL:`
format and drives the model via `/v1/completions`, injecting results as
`TOOL:` lines and continuing until no more `<tool_call>` or `max_iters`:

```json
POST /v1/forge/chat {
  "mode": "completion",
  "model_url": "http://127.0.0.1:18766/v1/completions",
  "messages": [{ "role": "user", "content": "review dist/micronaut-coder/src/main.cpp" }],
  "sampling": { "temperature": 0, "max_tokens": 90 },
  "max_iters": 5
}
```

Returns `{ ok, final, iterations, trace }` — `trace` records each dispatched
call + its real exit code. Handles **both** structured `tool_calls` (gemma3 +
chatml) and raw `<tool_call>verb|subject|object</tool_call>` text (GPT-2 SLERP).

> Caveat (measured): the 12-sample G003 fine-tune on the 124M mm-toolcall model
> reliably emits the `<tool_call>` **structure**, but tool-*name* selection is
> noisy (it can emit fixture/overfit names like `lookup_city` or `v6-coder`
> instead of `review`). Auto-dispatch is mechanically proven; production-grade
> name reliability needs more per-tool samples / a larger base model.

### 3B tool-call controller (dolphin-phi-2)

A 3B model serves as the reliable tool-call controller in the interceptor:
`dolphin-2_6-phi-2.Q5_K_S.gguf` (phi-2, chatml, strong function-calling).
Served via llama-cpp-python (`--chat_format chatml`), it emits **clean
`tool_calls` with exact file paths** when `tool_choice` is named — no
path-mangling (unlike gemma3-1B).

```powershell
start-dolphin.bat            # serve on :9003 (clean llama-cpp-python, no xcfe)
```

The interceptor **forces `tool_choice` on the first turn, then drops
tools/tool_choice** so the model answers after the result is injected —
giving a clean 2-iteration loop (force call → execute → inject → answer).
`activeModelChatUrl()` prefers `active-model.json`'s `controller` field and
defaults to the 3B dolphin on `:9003`. Verified: `iterations 2`, `review`
executed (exit 0), final answer returned.

> Note: like gemma3-1B, dolphin-phi-2 answers directly under `tool_choice:auto`
> (it does not self-dispatch). True **autonomous** dispatch still requires the
> fine-tune path (G003).

### GGUF → safetensors (general converter)

`tools/gguf_to_safetensors.py` now uses **gguf-py** and converts **any arch**
(not just GPT-2) — it dequantizes via `gguf.quants`, maps GGUF tensor names to
HF names for **gpt2 and phi2**, and transposes linear weights. Validated by a
Q8 round-trip on the G003 model: biases exact (0.0), weights within Q8 noise
(~1%). Use it to get dolphin-phi-2 into safetensors form if needed (e.g. a GPU
trainer); the dequant output for the 3B Q5 model is ~3 GB F16.

```bash
python tools/gguf_to_safetensors.py <in.gguf> <out.safetensors> [--out-f32]
```

### peft LoRA fine-tune (forge-tool auto-dispatch)

`tools/lora_finetune_forge.py` — the same concept as DDS LoRA shards (base
frozen, only adapters learn) via the standard HF peft stack. Trains a rank-8
LoRA on the forge-tool dataset so the GPT-2 **auto-emits** a valid
`<tool_call>` with the correct tool name + exact path (no forced
`tool_choice`).

```bash
python tools/lora_finetune_forge.py --forge-only --records 300 --steps 250
```

- Base: `E:/models/GPT2/mini-GPT/gpt2_lora_base` (re-prefixed `transformer.h.`)
- Adapter: `E:/models/GPT2/mini-GPT/gpt2_forge_lora` (4.7 MB, 0.94% params)
- Verified output: `<tool_call>{"name":"review","arguments":{"file":"dist/micronaut-coder/src/main.cpp"}}</tool_call>`

> Caveat: with a small forge-only set the model overfits to the most common
> tool (`review` for everything). Distinct per-tool selection needs more
> balanced examples per tool (todos/explain/v6-hashes).

### Dolphin → small distillation (`oss_distillation.py`)

Distill the 3B dolphin's tool-calling into the small GPT-2 as a LoRA. The
script was extended to pass the **tools schema + forced `tool_choice`** to the
teacher (so it emits `<tool_call>` the student learns), with prompts as
`TOOL|prompt` lines:

```bash
start-dolphin.bat   # teacher on :9003
python tools/oss_distillation.py \
  --student E:/models/GPT2/mini-GPT/gpt2_lora_base/model.safetensors \
  --engine http://127.0.0.1:9003 \
  --ollama-url http://127.0.0.1:1 \
  --prompts tools/forge_prompts.txt \
  --teacher-tools tools/forge_tools.json \
  --out E:/models/GPT2/mini-GPT/gpt2_dolphin_distill_lora.safetensors \
  --rank 8 --steps 20 --lr 1e-4 --teacher-tokens 60
```

- Teacher (dolphin) emits `<tool_call>{"name":"review",...}` — verified in log.
- Output: `gpt2_dolphin_distill_lora.safetensors` (4 MB, 96 tensors = LoRA
  A/B skeleton on the frozen base).
- Verified: the distilled model emits the `<tool_call>` **structure**, but the
  124M student still can't reliably reproduce specific tool *names* (same
  ceiling as direct fine-tune). The LoRA skeleton is in place; name fidelity
  needs a bigger student base.

### `tools/test_forge_toolcalls.mjs` — the verifier harness

Tests whether a model makes a **real, executable** tool call (vs just printing
JSON). Streaming, bounded, clean exit.

```bash
node tools/test_forge_toolcalls.mjs <model_chat_url> [gateway_url] ["prompt"]
TOOLCALL_CHOICE=review TOOLCALL_MAX_TOKENS=300 node tools/test_forge_toolcalls.mjs \
  http://127.0.0.1:9002/v1/chat/completions http://127.0.0.1:8764 "review the file"
```

Verdicts: `real-toolcall` (parseable + executed), `printed-json-not-toolcall`,
`no-toolcall`, `model-not-ready`, `timeout`, `max-tokens-hit`,
`model-produced-nothing` (degenerate vocab/template).

---

## 2. Inference path — GPU fix (G004/D015/D016)

**G004 root cause:** `ggml-xcfe.dll` loads even at `-ngl 0` and zeroes Gemma
logits → empty output. The model weights are healthy (verified via
`llama-cpp-python` directly).

- **D015:** use the clean llama-cpp-python backend (`llama-py-server`, no xcfe).
  `active-model.json` → `127.0.0.1:9002` (gemma3 `models/gemma-3-1b-it-q8.gguf`,
  `chat_format=gemma`). The gateway `/v1/code` proxies there.
- **D016:** Gemma 3 answers directly by default; **named `tool_choice` forces
  `tool_calls`**.

```powershell
llama-py-server gemma3        # clean gemma3 on :9002
```

> Template trap: GPT-2 BPE-vocab toolcall GGFs (e.g.
> `ultrachat_coder_skeleton_slerp_0p40.gguf`) do **not** contain chatml tokens
> (`<|im_start|>/<|im_end|>`). Serving them with `--chat-template chatml`
> produces garbage Unicode. They need raw text mode + the `<tool_call>`
> interceptor.

---

## 3. G003 — tool-call fine-tune pipeline

Goal: true **auto** tool dispatch (no forced `tool_choice`). **DONE** — the
fine-tuned `mm-toolcall` model now emits `<tool_call>` on its own, with the
correct forge tool + exact file path (fixes the gemma3-1B path-mangling issue).

### Data chain (executed)

```
tool_call_samples.jsonl + tool_result_continuation_samples.jsonl + tools/forge_toolcall_samples.jsonl
  → tools/build_toolcall_tokenbin.py  (render <tool_call> markers → tiktoken gpt2 → headed bin)
  → E:/models/GPT2/mini-GPT/toolcall_g003_tokens.bin   (12 seqs x 128)
  → tools/finetune_hf_tokenbin.py    (PyTorch AdamW, CPU)
  → E:/models/GPT2/mini-GPT/gpt2_small_lite_tool_g003/model.safetensors
  → to_gguf.py                       (q8_0)
  → E:/models/GPT2/mini-GPT/gpt2_small_lite_tool_g003_q8.gguf  (386 MB)
```

`tools/forge_toolcall_samples.jsonl` adds forge-tool samples (review / todos /
explain / v6-hashes + continuations) so the model learns the actual forge tools,
not just the fixtures' calculate/lookup tools.

### Render format (matches the /v1/forge/chat interceptor's raw parser)

```
SYSTEM: ...
USER: Review the file dist/micronaut-coder/src/main.cpp
ASSISTANT: <tool_call>{"name":"review","arguments":"{\"file\":\"...\"}"}</tool_call>
TOOL: <result>
ASSISTANT: <final answer>
```

### Commands

```bash
python tools/build_toolcall_tokenbin.py \
  khanary-llama-build/llama.cpp/tools/server/training/tool-call-fixtures/tool_call_samples.jsonl \
  khanary-llama-build/llama.cpp/tools/server/training/tool-call-fixtures/tool_result_continuation_samples.jsonl \
  tools/forge_toolcall_samples.jsonl \
  --out E:/models/GPT2/mini-GPT/toolcall_g003_tokens.bin --seq-len 128

python tools/finetune_hf_tokenbin.py \
  --base E:/models/GPT2/mini-GPT/gpt2_small_lite_tool.safetensors \
  --data E:/models/GPT2/mini-GPT/toolcall_g003_tokens.bin \
  --out E:/models/GPT2/mini-GPT/gpt2_small_lite_tool_g003 \
  --epochs 30 --batch 1 --lr 3e-5 --threads 8 --save-every 60 --block 128

python E:/models/GPT2/mini-GPT/to_gguf.py \
  E:/models/GPT2/mini-GPT/gpt2_small_lite_tool_g003/model.safetensors \
  --out E:/models/GPT2/mini-GPT/gpt2_small_lite_tool_g003_q8.gguf --quant q8_0
```

### Verified

- Loss converged ~7.6 → ~0.05 over 360 steps (30 epochs).
- Python greedy generation emits `<tool_call>` for `review`/`todos` prompts with
exact paths.
- Served via `/v1/completions` (raw, `chat_format` has no `raw` handler so use
`/v1/completions` for this raw-text model), it emits
`<tool_call>{"name":"...","arguments":{"file":"dist/micronaut-coder/src/main.cpp"}}</tool_call>`
on its own — no forced `tool_choice`.

### Deployment note

This is a raw-text GPT-2 model: integrate via `/v1/completions` (or render
messages to the `USER:/ASSISTANT:` text and use a `chatml` handler). Wire it as
the model_url in `/v1/forge/chat` to close auto-dispatch end-to-end.

---

## 4. Quick reference

| Endpoint | Purpose |
|---|---|
| `GET  /v1/forge/tools` | list executable tools |
| `POST /v1/forge` | execute a tool |
| `GET  /v1/verbs` | list SVO verb vocabulary |
| `POST /v1/forge/verb` | dispatch a verb |
| `POST /v1/forge/chat` | tool interceptor loop (model → execute → continue) |
| `tools/test_forge_toolcalls.mjs` | verify a model makes real tool calls |

---

## 5. Quantum Trinity bots

Five JSON-RPC C++ executables in `dist/Quantum/build/`, registered in
`micronauts/registry.json` under `sidecar://quantum/dispatch`.
All smoke-tested 2026-08-25. Built with VS2022 `cl` from `dist/Quantum/src/`.

| Exe | Operations | Role |
|---|---|---|
| `quantum_hybrid.exe` | `process` / `analyze_code` / `extract_relations` / `extract_patterns` / `get_history` | CHEESE code-edge emitter — Roslyn AST parser + Regex engine + ELIZA chatbot + ADAM12 engine |
| `quantum_trinity.exe` | `research` / `analyze_ngrams` / `translate_notation` / `store_memory` / `retrieve_memory` / `get_metrics` | Web research driver — DuckDuckGo Instant API + Wikipedia API + DDG HTML fallback |
| `quantum_microagents.exe` | `process` / `get_agents` / `get_history` / `get_context` / `get_config_paths` | Candidate-only sidecar — 6 agent templates (parser/therapist/cognitive/regex/quantum/code); `authority_boundary:"candidate_only"` |
| `quantum_personality.exe` | `interact` / `get_profile` / `get_personas` | Adaptive personality engine — CognitiveState (arousal, dominance, valence, trust, rapport, quantum_coherence), 5 persona affinities |
| `quantum_grammar.exe` | `parse` / `get_grammar` / `get_parse_tree` | Grammar parser |

### Authority boundary

`quantum_microagents` emits **candidate JSON only** — it never creates, updates, merges,
or promotes micronauts. All responses carry `"authority_boundary": "candidate_only"`.
Only the micronaut factory/evolution sidecars have promotion authority.

### Protocol

All bots accept JSON on stdin, emit JSON on stdout:

```bash
echo '{"operation":"research","query":"SCXQ2 quantization"}' | quantum_trinity.exe
echo '{"operation":"interact","input":"Hello"}' | quantum_personality.exe
echo '{"operation":"process","code":"int x=1;"}' | quantum_hybrid.exe
```
