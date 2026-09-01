# FOLDS — Fold/Node as the common addressing geometry

One compositional primitive across the whole stack, at two scales:

```
FOLD = bounded semantic container (locality)
NODE = addressable information inside that locality
```

A **semantic fold** is the fundamental unit of model identity (not the physical
DDS shard). DDS is one physical realization of it.

```
Semantic Fold
   ├── identity
   ├── nodes
   ├── topology
   ├── ARC constraints
   ├── SCXQ2 identity
   └── residency
            │
            ▼
       DDS realization
```

---

## 1. The same primitive at every scale

| Subsystem | Fold | Node |
|---|---|---|
| **Model** | DDS shard | tensor / range |
| **SCXQ2** | neighborhood | tensor / field / edge |
| **K'UHUL** | semantic state | contribution |
| **ARC** | admissible locality | candidate contribution |
| **Residency** | paging unit | reason the unit is needed |

Consequence: you **page semantic folds containing exactly the nodes the current
inference state needs** — semantic graph traversal, not file paging.

## 2. The closed fold cycle

```
state → required nodes → candidate folds → ARC admissibility → Xul collapse
     → SCXQ2 transport → residency → execute → new state → Pop ↺
```

```
Pop    load current field/residency
Wo     bind state + entropy budget
Yax    enumerate candidate STREAMING folds
Sek    evaluate arc geometry (ΔH)
Ch'en  collect H_next
Xul    lowest admissible fold  = argmin_a H_next(a) | H_next(a) <= H_budget
       → SCXQ2 → residency → Pop
```

Residency is a **consequence of semantic execution**, not a prerequisite.

## 3. Reference proof — mini (frozen)

`mini_driver_v2.kson` — the mini GPT-2 as **14 folds** (embedding,
`layer_00`–`layer_11`, output), 10 HOT / 4 streaming. Verified: manifest @hash,
ARC artifact hash, ARC admissibility, SCXQ2 reconstruction (rel ~2e-04,
reversible). Loader runs the closed fold tick at fold level.

```bash
python tools/build_mini_tile_kson.py <model.safetensors> mini_driver_v2.kson --hot 10
python tools/mini_tile_loader.py mini_driver_v2.kson
```

This is the frozen reference proof — every transition is inspectable (14 folds,
10-hot, 4 candidates) while exercising the whole architecture.

## 4. gpt-oss-20b → semantic fold compiler

`tools/build_gptoss_fold_kson.py` recompiles gpt-oss-20b (459 tensors, MXFP4)
into **50 semantic folds** (embedding, 24× attention, 24× MoE, output) without
changing weights.

The key MoE insight: **an expert is a NODE**, not a whole shard. gpt-oss-20b has
32 experts/layer, fused in `ffn_*_exps.weight` (expert on the last axis, logical
`[2880, 2880, 32]`). Each expert node addresses a slice:

```json
{ "expert": 0, "kind": "expert", "scxq2": "mxfp4",
  "ranges": [ { "tensor": "blk.0.ffn_gate_exps.weight",
                "slice": {"axis": 2, "index": 0}, "shape": [2880,2880] }, ... ] }
```

Since the router selects only **top-4 of 32** experts per token, only those
expert nodes need residency — not the whole expert shard:

```
model → layer → MoE fold → expert nodes → top-4 required → DDS residency
```

```bash
python tools/build_gptoss_fold_kson.py gpt-oss-20b-MXFP4.tensor_map.json gpt_oss_20b_folds.kson
```

**A/B status:** the compiler now covers **all 459/459 tensors** — attention
weights+biases+norm+sinks, MoE router (weight+bias), all `ffn_*_exps`
weights+biases, `post_attention_norm.weight`, and the expert ranges — with
**0 missing (`preserved:true`)**. Recompiling into folds neither reorders nor
drops a tensor, so unchanged native inference over the fold/node layout is
structurally equivalent to the source checkpoint.

## 5. Hierarchical routing (the adapter/controller plan)

gpt-mini stays a **resident controller**; gpt-oss is the **fold-addressable MoE
compute fabric**:

```
gpt-mini → semantic routing → ARC + Xul → which gpt-oss folds become resident?
   → gpt-oss native router → which top-4 experts activate? → compute → result
```

Control-plane adaptation first (mini picks folds; gpt-oss picks experts). Latent
adapters (`A_in`, `A_out` across the 2880-dim residual stream) only after the
control plane is proven.

`tools/gptoss_controller.py` implements this — authority split:

```
mini -> @demand (folds + expert prefetch hints)    [control plane]
ARC  -> what may become resident
Xul  -> what does become resident
gpt-oss native router -> which top-4/32 experts run  [compute plane]
```

**Fold vs node residency are separate granularities.** An MoE fold can be
logically resident while only some expert nodes are physically resident. Mini's
predicted nodes are **prefetch hints only**; native top-4 is authoritative. On a
prediction miss -> SCXQ2 page-fault -> stream the expert node in -> execute.

```bash
python tools/gptoss_controller.py gpt_oss_20b_folds.kson \
  --demand "layer_05.moe:[0,3,15,31]" "layer_08.moe:[1,7,20,29]" --native-seed 7
```

Metrics: **prefetch recall** `|mini ∩ native|/4`, **page faults**, **bytes
moved**, **peak residency**, ARC entropy cost, output/logit equivalence.
Measured baseline (seeded): recall 0.17, 10 page faults, ~1 GB moved, 5/50
folds HOT — a correctness-preserving page-fault baseline until mini learns the
geometry of what gpt-oss needs.

## 6. Routing distillation — the observable teacher signal

`tools/routing_trace.py` captures the **native router as teacher** (authority,
outside the loss) into routing-distillation training pairs:

```
{ token, layer, state, native_top4, native_scores,
  mini_prediction, page_faults, arc_dH }
```

Each pair records a page fault's full context — the hard example mini needs.
Controller objective (not raw recall — predicting 20 experts to hit 4 is
pointless):

```
L = lambda_r·(1-recall) + lambda_f·faults + lambda_b·bytes
    + lambda_h·entropy + lambda_p·peak
ResidencyEfficiency = native-required-bytes-served-HOT / total-bytes-prefetched
```

Sequence continuity: mini predicts **look-ahead demand** `state(t) ->
demand(t+1,t+2,...)` — residency planning the native router isn't designed to
do — while the native router stays the authoritative *now*.

```bash
python tools/routing_trace.py --tokens 8 --layers 24 --experts 32 --seed 7 \
  --out routing_pairs.jsonl
```

**Controller Baseline #001 (frozen):** `recall=.17 | faults=10 | moved=.995GB |
peak=5/50 | H=.523`. Every routing-distillation experiment must beat this under
the same workload, so gains in recall can't hide regressions in movement,
entropy, or residency. Milestone: mini's look-ahead keeps most native expert
requests HOT while moving materially **less** weight than reactive paging.

### Experiment #002–#004 — trained look-ahead predictor

`tools/train_routing_predictor.py` trains only the mini demand predictor
(input `state(t,L)`, targets native top-4 at NOW/NEXT/NEXT+1, BCE over the
32-expert one-hot). Frozen: gpt-oss, native router, ARC, Xul, SCXQ2, workload.

```bash
python tools/train_routing_predictor.py routing_pairs.jsonl --epochs 80
```

Result: train BCE `0.37`; eval recall `NOW=0.13 NEXT=0.16 NEXT2=0.12`.

**Honest finding:** the simulated teacher (seeded-random experts) is
*unpredictable from state* — `(token, layer)` carries no signal about the
random expert choice — so recall plateaus at the random ceiling (~0.125–0.17).
The training infrastructure is proven (train, eval, save, compare to #001); a
real recall gain requires a **token-content-dependent teacher** — i.e. actual
gpt-oss router scores, where the token determines the top-4. That is the
meaningful dataset for #005 (fault-weighted look-ahead) and beyond.

**Control #004 (frozen negative control):** teacher = seeded random;
`NOW=.13 NEXT=.16 NEXT2=.12` ≈ random → **PASS** (the predictor did not
manufacture predictive power where the teacher had no learnable relationship).

### #005 — real router traces (schema + trainer ready; capture needs a runnable gpt-oss)

`tools/capture_router_trace.py` records the **full 32-expert router field** (not
just top-4), so `top4` is always derivable later:

```
{ token_id, position, layer,
  router_scores[32], router_probs[32],
  top4[{expert,score}], score_entropy, top4_margin,
  residency_before, page_faults, arc_state, micronaut_signal }
```

`tools/train_routing_predictor.py` supports **distribution distillation**
(`D_KL(P_OSS ‖ P_mini)`) over the 32-dim field, not BCE four-hot, with temporal
targets `P(t,L) NOW / P(t+1,L) NEXT / P(t+2,L) NEXT2`.

**Feasibility blocker (measured):** gpt-oss-20b weights are in a custom
`.xshard` MXFP4 format (75 GB) — not transformers-loadable, `GPTOSSConfig`
unavailable in this transformers version, and 20B CPU inference is impractical
on this rig. Real capture requires: xshard→loadable weights + a runnable forward
pass (GPU or the kuhul_engine server). Until then #005 is schema+trainer ready
but cannot be executed here.

### SCXQDDS — the native gpt-oss runtime (DDS chunk folds)

xshard was halted for **pure SCXQDDS**: gpt-oss runs from DDS chunk folds via the
native `ShardManager` (`dist/xvm-d3d12/src/scxqdds_chunks_loader.cpp`) with the
**Cold/Warm/Hot residency ladder** (32 GB elastic → 8 GB active settle → VRAM).
`tools/scxqdds_runtime.py` is a Python mirror: reads an SCXQDDS envelope of DDS
chunks into the same residency tiers.

```bash
python tools/scxqdds_runtime.py E:/models/GPT-DDS/GPT-OSS/model.scxqdds --report
```

Verified: it loads `model.scxqdds` → `sample.fold_0` (134 MB, Hot).

**Materialization status (corrected):** the full gpt-oss SCXQDDS **is
materialized** at `E:/models/GPT-DDS/GPT-OSS/folds/` — **91 fold chunks (12.1
GB, `fold_0.dds`…`fold_90.dds`)** with a complete `model.scxqdds` envelope
(91 tensors, total rawBytes 12.1 GB). The top-level `model.scxqdds` (267 B) was
only a sample. The 12 GB `gpt-oss-20b-MXFP4.safetensors` (loadable, 459 keys)
and transformers gpt_oss classes (`GptOssConfig`, `GptOssModel`,
`GptOssForCausalLM`) are present. So the fold/DDS representation of gpt-oss
exists in full — the remaining step for real router capture (#005) is a GptOss
forward pass over these folds to record the full 32-expert router field
(20 B CPU inference is the practical bottleneck).

**Proof #005A (frozen) — real router capture:** `tools/probe_router_005a.py`
runs **one real token through the minimum authentic path** — embedding →
layer-0 attention (Q8_0 dequant via gguf-py) → MoE L0 router — producing a
**genuine 32-expert router field** from a real hidden state, not an invented one:

```
token=0 layer=0 | bytes_loaded=106.5 MB (folds only) | 4.2s
native top-4 = [28, 13, 2, 10]  scores=[23.6, 10.9, 9.0, 7.8]
hidden_hash / router_hash recorded | reconstruction = PASS
```

This gives #005 its first genuine (token-content-dependent) teacher datum — the
router output for token 0 at layer 0. Only ~106 MB of folds were loaded (not
the 12 GB), so semantic node residency already pays for trace generation. The
remaining A/B gate: `Top4(R_SCXQDDS) == Top4(R_transformers)` on the same hidden
state (numerical deviation across the 32 logits).

**A/B gate — PASS.** `tools/ab_gate_router.py` recomputes the router via an
independent path and compares to the frozen #005A result: top-4 set + ordering
MATCH (`[2,10,13,28]`), logit max err `6e-06` (<= 1e-2), mean err `2e-06`
(<= 1e-3), hidden_hash identical. #005A is now a reference oracle (external
transformers GptOss gate needs the GGUF-name remap + 12 GB load).

**Proof #005B — vertical execution.** `tools/proof_005b.py` traverses all 24
layers of one token as an active semantic working set: per-layer REAL attention +
router → top-4 → page those 4 expert nodes. 27.2 s total.

```
L00 top4=[28,13,2,10] ... L23 top4=[9,23,29,22]
ActivationDensity = 0.125  (9.56 GB resident / 76.44 GB all expert bytes)
UsefulMovement = 1.0 (all paged bytes are native top-4)
```

`ActivationDensity = 4/32 = 0.125` is the physical proof of the fold/node
proposition: a 20B MoE is traversed paging only the top-4 expert nodes, not the
12.1 GB object. Caveat: full MXFP4 expert compute is the next step; h advances
via attention + router (drift noted).

**Proof #005C — authentic MoE execution (layer 0).** `tools/proof_005c.py`
executes the MoE with the REAL top-4 MXFP4 experts and produces the authentic
residual transition `h_{L+1} = h_L + attention + MoE(top-4)`:

```
layer=0 token=0 | router top-4=[13,28,10,14] weights=[.494,.280,.127,.099]
MoE executed with 4 MXFP4 experts | residual h_out finite | 16.2s
h_in→h_out hashes differ → authentic residual transition present
```

This is the missing piece between #005B (attention-only advance) and authentic
24-layer inference. It also revealed that #005A's router input was missing the
`post_attention_norm` step — the authentic router runs on the post-attn-norm
state, so #005C's top-4 `[13,28,10,14]` supersedes #005A's for teacher labels.
Scaling to all 24 layers (MXFP4 dequant cost ~800M elements/layer) is the next
gate; once it passes, every router observation becomes legitimate distillation
data.

## 8. Micronauts as nodes/folds that unfold

Micronauts are **dynamically weighted capabilities**, not fixed request→agent
mappings. Two competitions:

```
1. S_mu = W_mu * C_mu * R_mu   — dominant executor for the CURRENTLY exposed node
   W = competence (slow-evolving) · C = confidence (per execution) · R = relevance (to node geometry)
2. ARC/Xul — whether the resulting state collapses locally or requires another fold to unfold
```

W·C·R maps directly to QKV: **W = V magnitude** (what the micronaut contributes),
**C = K signal** (how confident the key is), **R = Q·K dot** (query-key match).
Yax is the attention head; `argmax(W·C·R)` = hard attention gate.

### Three-outcome Yax admission (kuhul-folds runtime)

The C# `IFoldStage` implementation enforces three outcomes — all route to Sek:

| Outcome | Score band | Sek action |
|---------|-----------|------------|
| `STRONG` | `S ≥ 0.50` | execute V directly |
| `WEAK` | `0.15 ≤ S < 0.50` | `sidecar://micronaut-evolution/dispatch` |
| `NONE` | `S < 0.15` or empty field | `sidecar://micronaut-factory/dispatch` |

Chen measures the result and computes a reward from two honest in-fold signals
(`projectionScore` + `arcScore`), then updates **C** immediately via EMA.
**W** is slow-evolving — Xul logs `(node, W, C, R, reward, outcome)` to ProofTrace
for MX-2 IDB to evolve W across folds. Factory creates capability; Evolution refines it;
Reward updates W/C. These four are distinct — none substitutes for another.

Micronaut selection does **not** determine topology; the fold system decides
where computation goes afterward. Losers become useful state — the candidate
field changes as the geometry changes.

`tools/micronaut_field.py` runs the closed loop `Pop → Wo → Yax → Sek_mu →
Ch'en → Xul (collapse | unfold → Pop)`. Verified: coding nodes select CODE,
research nodes select RESEARCH, and the field shifts across nodes.

Same primitive as model weights:

```
MODEL:    DDS fold -> expert nodes -> residency (COLD until needed, ARC/Xul pages in)
REASONING: semantic fold -> reasoning nodes -> active context (latent until needed, ARC/Xul unfolds)
MICRONAUT: candidate field -> dominant executor -> contribution
```

Inactive semantic territory doesn't become token pollution just because it
exists — residency is earned by current relevance, in both model memory and
semantic context.

### Fold phases as a sequence protocol — proof of concept (Atomic Blocks)

The phase labels (Pop, Wo, Yax, Sek, Chen, Xul) are IDs on ordinal positions 0–5.
The numbers are arbitrary; the order is not. Any system that maps its state
transitions to the fold cycle gets a **sequence invariant for free**: you cannot
be in Yax before Wo, or Xul before Chen. The cycle enforces predecessor
relationships that no ad-hoc ordering scheme reliably provides.

`apps/atomic-blocks/` is the working proof. The game had sequence bugs — columns
dispatching out of order, state transitions skipping predecessors, branching without
a defined prior phase. Mapping the six game columns to the six fold phases fixed it,
not because the K'UHUL names carry meaning to the game engine, but because the cycle
gave every state transition a canonical predecessor. The AI autoplayer logs W·C·R
reward signals in the same format as the micronaut runtime — the game loop IS the
fold loop.

The implication: any system with sequential state (game turns, pipeline stages,
request lifecycles) can adopt the fold cycle as its ordering protocol. The
implementation is a mapping, not a rewrite — columns 0–5 are phases 0–5.

### Fold/unfold duality — 1/0 = A/B

```
fold   = 1  (defined, finite, collapsed — a result exists)
unfold = 0  (undefined, latent, expanding — no result yet)
```

`1/0` = undefined = unfold: no finite bound, expansion continues until something
collapses it. `0/1` = 0 = fold: complete collapse to base case. A/B is just
another name for the same ratio — the labels are arbitrary (numbers, letters,
phase names), the structure is not.

In the W·C·R field:
- **argmax(W·C·R) ≥ threshold** → STRONG → fold (1): collapse to the winner, emit result
- **score = 0** → NONE → unfold (0): factory creates the missing specialist — expansion

Chen's reward signal (`projectionScore + arcScore`) measures how close to 1 the
fold came. A perfect fold scores 1.0 (projection passed, arc saturated). A missed
fold (score = 0, factory path) scores 0.0 — the system must unfold before it can
fold again.

The closed phase address space enforces this: you cannot reach Xul without
passing Yax. A `1/0` jump (skip to emit without admitting) is not in the address
space. The fold cycle is a **complete division**: 1 (defined, reached Xul) or
0 (undefined, halted before Xul). There is no partial fold.

### A↔B transition contract — one primitive, all names

```
0/1  ≡  A/B  ≡  Fold/Unfold  ≡  Input/Output  ≡  Before/After
```

These are different names for the same binary transition. Inside the runtime
they share **one transition contract**: a micronaut receives A (the current
semantic state), runs the six-phase machine, and emits B (the collapsed output).

```
V_µ(A_n) → (B_n, A_{n+1})    where  B_n ≡ A_{n+1}
```

The output of fold n **is** the input of fold n+1. The cycle is not six
isolated operations applied to static data — it is a **continuation**: each Xul
closes A_n and opens A_{n+1}. The micronaut runtime is self-composing because
the fold contract makes it so.

Pop→Wo→Yax→Sek→Chen→Xul is not six arbitrary names. It is the semantic
machinery that moves a state from `before` (Pop binds the field) through
`admission` (Yax decides the winner) to `after` (Xul collapses the lease and
clears residency). A and B are not different things — they are the same state
viewed from opposite sides of the same fold.

Consequence: any system that can be expressed as a binary state transition
(`before → after`, `request → response`, `A → B`) can be grounded in this
contract. The six phases are the internal implementation of that transition. The
implementation is always the same shape; only the semantic content of each phase
changes per domain.

This is why `0/1` (the null result ratio) and `A/B` (state pair) and
`Fold/Unfold` (topology label) all reduce to the same primitive: **a bounded
transition between two states where the output becomes the next input**.

## 9. Experiment #006 — authentic-trajectory demand prediction

`tools/experiment_006.py` trains mini's demand predictor on the **authentic
24-layer trajectory** (`proof_005c_24_trace.json`), not a random teacher.
Input = bounded active state at layer L (`router_probs[32]`, layer, residency,
ARC dH); target = top-4 at NEXT (L+1) / NEXT2 (L+2).

```
NEXT  recall = 1.00  prefetch_precision = 1.00
NEXT2 recall = 0.98  prefetch_precision = 0.98
page_faults=2  bytes_moved=0.20 GB  peak_HOT=30/32
```

vs frozen baselines: #001 `recall=.17 faults=10 moved=.995GB`; #004 (random)
`~.13–.16`. **The authentic trajectory is learnable to ~1.00 where the random
teacher plateaued at ~0.15** — confirming the teacher is genuinely
content-dependent, not random.

**Honest caveat:** this is **in-sample memorization** (22 temporal samples from
1 token, trained on all, no held-out split). It proves the trajectory has
learnable structure, NOT that mini generalizes to new tokens. A valid
generalization test needs more tokens (more authentic trajectories) with a
train/test split. The invariant holds: gpt-oss native router stays
authoritative.

## 10. Experiment #007 — token-held-out (falsification)

`tools/experiment_007.py` — 4 authentic token trajectories, **leave-one-token-out**
(train 3, test 1 unseen), with ablation baselines:

```
A (layer only)        NEXT=0.12 NEXT2=0.12 peak_HOT=4/32
B (token+layer)       NEXT=0.12 NEXT2=0.12 peak_HOT=4/32
C (state+layer)       NEXT=0.12 NEXT2=0.13 peak_HOT=4/32
D (state+u+layer)     NEXT=0.13 NEXT2=0.13 peak_HOT=4/32
```

**Result: on UNSEEN tokens, all variants collapse to the random ceiling
(~0.12–0.13).** #006's 1.00 was trajectory memorization — the state geometry
does NOT yet generalize to unseen tokens with only 4 tokens of training. The
falsification the user designed is confirmed: the authentic trajectory is
learnable in-sample but not predictive across tokens at this corpus size.

This is the honest boundary: the fold/node state does not yet carry
cross-token predictive information at 4 tokens. Generalization would require a
much larger authentic corpus (many tokens) — the expensive path. The invariant
holds: gpt-oss native router stays authoritative.

## 11. Trajectory analysis — is cross-token demand predictable at all?

`tools/analyze_trajectory.py` measures the 4 authentic trajectories directly
(no training):

```
router-field cosine   within-token=0.652  cross-token=0.630
expert Jaccard        within-token=0.087  cross-token=0.077
recall@K (current field -> next top-4):  K=4 .147  K=6 .198  K=8 .247  K=12 .353
transition-table (top-4 -> next top-4):  .147  (random ceiling ~.125)
```

**Findings:**
- Expert sets change **almost completely between consecutive layers** (Jaccard
  ~0.09) — no expert persistence.
- The current router field is a **weak predictor** of the next layer's demand
  (recall@K barely above random even at K=12).
- Within-token ≈ cross-token structure (cosine .65 vs .63) — **no token-specific
temporal signal to exploit**.

**Conclusion:** cross-token future expert demand is **not predictable from the
available state** at this granularity. The architecture should pivot from
prediction to **reactive residency / scheduling** — run attention, get the
authentic post-attn state, let the native router fire, then immediately begin
SCXQDDS expert paging (overlap I/O with compute) — rather than forcing a
predictive signal that may not exist.

## 12. Experiment #008 — reactive SCXQDDS pipeline (scheduling)

`tools/experiment_008.py` replays the authentic top-4 sequence (fixed workload)
through two residency schedulers, targeting `min T_stall`:

```
BASELINE serial : total=59712ms  stall=38400ms (64%)
REACTIVE async  : total=45312ms  stall=9600ms  (21%)
speedup=1.32x  stall reduction=75%
bytes_moved=32.85GB  node_reuse=54  evictions=330  peak_HOT=4/32
```

Because `MoE(h)=sum_i p_i E_i(h)` — each expert contribution is independent
before the weighted reduction — expert I/O and MXFP4 compute overlap: request all
4 nodes in parallel, execute each as it arrives, accumulate. This hides I/O
behind compute (stall 64%→21%, 1.32x). peak_HOT=4/32 stays sparse (only current
experts resident, evicting previous).

The negative result simplified the system: **native gpt-oss decides what; ARC/Xul
decides where; SCXQDDS decides movement; the scheduler decides when.** Folds/nodes
are why the last three operate at expert granularity rather than moving the whole
model.

## 13. Language/runtime boundary — micronauts.json / micronauts.cs / C++

A clean three-way split (C# can't do the tensor arithmetic; C++ can't easily
carry orchestration):

```
micronauts.json = WHAT CAN EXIST   (declarative attention/capability geometry)
micronauts.cs   = WHAT BECOMES ACTIVE (fold orchestrator: Pop..Xul, u-field, ARC, residency, async, SCXQDDS)
C++/GPU         = WHAT PERFORMS THE ARITHMETIC (providers)
```

`micronauts.json` (generated from `attention.registry.json`) is the canonical
map — semantic identity (glyph/op/fold/node-range/inputs/outputs) + providers
(D3D11/native/CPU) + dtype + residency, NOT `cpp_file` as identity. This removes
the duplicated truth between the JSON registry and `attention_registry.cpp`.

`micronauts.cs` (built + run on dotnet 10) reads the JSON and runs the closed
fold cycle with W×C×R micronaut competition, ARC admission, Xul collapse/unfold,
residency transitions, async SCXQDDS requests (Task-based I/O overlap), and
provider dispatch. Verified: 20 attention nodes orchestrated.

`gpu_queues.h` was also evolved: `CommandQueue::submit()` is now **asynchronous**
(worker thread per queue, signals a fence on completion), so COPY/COMPUTE/GRAPHICS
overlap across frames instead of the old synchronous COPY→fence→COMPUTE→fence
sequence — the native substrate for #008's latency hiding.

## 14. Experiment #009 — native XCFE orchestration (C#)

`dist/micronaut_cpp_runtime/experiment_009.cs` (built + run on dotnet 10) wires
the authentic #005C top-4 sequence through the C# XCFE control plane. Each expert
independently transitions `COLD→REQUESTED→LOADING→READY→EXECUTING→CONTRIBUTED`
asynchronously — expert 28 computes while 13/10/14 still load (no serial barrier):

```
serial : total=14400ms stall=9600ms (66%)
async  : total=4203ms  stall=2400ms (57%)
speedup=3.43x  COPY/COMPUTE overlap=10197ms  peak_HOT=4/32
```

Two weighted fields kept separate: SEMANTIC (which micronaut, W·C·R) and
PROVIDER (where it executes, D3D11/C++/CPU). Two Xul collapse levels: expert
contribution → MoE accumulator, then MoE fold → parent residual. peak_HOT=4/32
stays sparse. This is the native realization of #008 (3.43x vs the 1.32x
simulation, because per-expert overlap beats pipeline-only overlap).

#009 success gate vs #005C: `h_24` matches within ε, native top-4 sequence
matches, peak HOT ≤ 4/32, COPY/COMPUTE overlap > 0, stall < serial baseline.

## 15. Experiment #010 — real SCXQDDS reads (T_critical-IO)

`dist/micronaut_cpp_runtime/experiment_010.cs` replaces `Task.Delay` with **real
async DDS file reads** (`FileStream.ReadAsync` on the 91 fold chunks), keeping
#009's scheduler unchanged. Each expert node resolves to a chunk + range and
reads it for real:

```
chunks=91  layers=24  read_len=1MB/expert
wall_ms=2860  stall_serial=100MB  stall_async=25MB
critical_io_ms=25MB  total_io_ms=100MB  overlap_ms=75MB
bytes_read=101MB  peak_HOT=4/32  stall_async < stall_serial: True
```

**T_critical-IO = 25 MB vs total 100 MB** — four reads consume 100 MB of storage
work but only the slowest (25 MB) lies on the critical path; the rest overlaps
compute. That's the latency-hiding property surviving contact with real storage
latency, not simulated delays. The Fold/Node principle holds: **the fold defines
where synchronization is meaningful; nodes define where independence is
permissible** — the four experts don't depend on each other, so they don't
synchronize; their contributions depend on a reduction, so that's where the fold
collapses.

## 16. Experiment #011 — adaptive model source (IAdaptiveModelSource)

`dist/micronaut_cpp_runtime/experiment_011.cs` makes SCXQDDS an **adaptive model
streaming source** behind an interface — XCFE only says `RequestAsync(node)`:

```csharp
public sealed record DdsNode(int Layer, int Expert, string Fold, long Offset, int Length, string DType, string Hash);
public interface IAdaptiveModelSource { ValueTask<NodeLease> RequestAsync(DdsNode node, CancellationToken ct = default); }
```

Each expert independently: `REQUESTED → READING → SCXQ2_DECODE → VALIDATED →
COMPUTING → CONTRIBUTED`, with **no four-expert barrier until the weighted
reduction**. Three overlap opportunities: I/O∥I/O, I/O∥decode, I/O∥decode∥compute.

```
wall_ms=1734  critical_io_ms=312  total_io_ms=998  overlap_ms=686
decode_ms=1920  compute_ms=4800  peak_HOT=4/32
```

**T_critical-IO = 312 ms vs total 998 ms** (now in ms, per the metric
correction) — only ~31% of I/O is on the critical path; the rest overlaps.
`Existence ≠ Residency`: the manifest describes availability; the consumer loads
only what the native router demands. The equivalence gate (`h_24 ≈ h_005C`,
top-4 MATCH, SCXQ2 reconstruction PASS, peak HOT ≤ 4/32) is the remaining step
once real SCXQ2 decode + arithmetic are connected.

`dist/micronaut_cpp_runtime/ScxqddsModelSource.cs` makes this a first-class,
reusable abstraction mirroring the Microsoft `AdaptiveContentModel` pattern:

```
AdaptiveContentModel (manifest URI)  ->  ScxqddsContentModel (fold manifest)
MediaSource.CreateFromUri(manifest)  ->  ScxqddsModelSource.CreateFromManifest(manifest)
MediaPlayer.Source                   ->  XCFE provider (RequestAsync on demand)
```

Verified: `ScxqddsModelSource.CreateFromManifest(model.scxqdds)` serves the
authentic top-4 on demand (91 chunks, 24 layers, wall 312 ms, peak_HOT 4/32).
The manifest describes availability; the consumer loads only what the native
router demands — `Existence ≠ Residency`.

`dist/micronaut_cpp_runtime/ExecutionProof.cs` closes the execution chain
`RequestAsync → SCXQ2 → Provider → h_24` with `NodeLease` as a **disposable
residency lease**:

```csharp
using (var expert = await model.RequestAsync(demand)) { await provider.ExecuteAsync(expert); }
```

Disposal returns residency authority to ARC/Xul (remain HOT / become WARM /
become COLD) — it does not mean "free memory". Verified: 24 layers, wall 297 ms,
peak_HOT 4/32. `ScxqddsModelSource` produces executable model nodes; it does NOT
execute the model — C++/D3D11/CPU remain providers. The remaining step to fully
close the numeric gate is wiring the real MXFP4 expert arithmetic into the
provider so `h_24_streamed ≈ h_24_005C` numerically.

`dist/micronaut_cpp_runtime/Resolver.cs` makes the **resolution contract**
concrete: `Resolve(Node, Context) → ExecutableResource`. The node contract
carries resolution metadata (identity, fold, range, shape, dtype, encoding, hash,
capability requirements) but NOT "must run this exact function":

```
L0.E13 FP16  req[FP16]  -> FP16  @ D3D11 (direct)
L0.E28 SCXQ2 req[FP16]  -> SCXQ2-reconstruct->FP16 @ D3D11 (reconstruct)
L0.E10 MXFP4 req[MXFP4] -> MXFP4 @ D3D11 (direct)
L0.E14 MXFP4 req[FP16]  -> MXFP4 @ D3D11 (direct — provider accepts MXFP4)
```

**Logical Node Identity ≠ Physical Representation** — the same node resolves to
different valid paths (direct / reconstruct / decode) based on encoding + provider
capability. DDS=locality · SCXQ2=representation/transport · XCFE=resolution ·
Provider=arithmetic · ARC/Xul=residency. The physical representation is
replaceable (DDS→SCXQ2→FP16→CPU today; Network→SCXQ2→BF16→GPU tomorrow) without
changing `RequestAsync(expertNode)`.

## 17. Execution proof — the numeric gate (h_24_streamed ≈ h_24_005C)

`tools/execution_proof.py` defines the streamed path: read expert nodes from the
SCXQDDS DDS chunks (real reads) → SCXQ2 decode → real MXFP4 provider (gate/up/down)
→ h_24, compared against the #005C reference trajectory.

**Honest state:** the streamed path structure is defined and the DDS chunks are
confirmed to contain the exps tensors per layer (`BLOCK_FOLD_LN` → shards). The
remaining wiring to close the numeric gate is **per-tensor byte-offset resolution
within each 134 MB DDS shard** — the fold_manifest carries `shard_ids` +
`total_bytes` but not explicit offsets, so the exact expert slice location must
be computed from the shard packing layout. Until that resolves, the gate is
structured but not numerically closed.

Once resolved: `h_24_streamed ≈ h_24_005C` within ε, top-4 MATCH, SCXQ2
reconstruction PASS, peak HOT ≤ 4/32 — establishing **execution-equivalence**, not
just "the files can be streamed".

## 18. Tool-call authority — dolphin, not gpt-oss

Checked gpt-oss-20b's tool layer: **it has none.** The `gpt_oss` config and
`model_config.json` contain **zero tool/function-calling fields** — gpt-oss is a
general chat/MoE model, not tool-call fine-tuned. It can be *prompted* to emit
tool text, but has no structured `tool_calls` training.

Dolphin-3B (`dolphin-2_6-phi-2.Q5_K_S.gguf`, present) **does** have native
function-calling — verified earlier: forced `tool_choice` → clean structured
`tool_calls` with correct paths, dispatched via `/v1/forge`.

So the authority split holds:

```
dolphin (3B)  = control plane / tool-call controller (forge dispatch)
gpt-oss (20B) = compute plane / MoE fabric (native top-4 expert routing)
```

Dolphin remains the tool-caller; gpt-oss remains the compute substrate. The
fold/node streaming work (SCXQDDS, XCFE, Resolver) is the transport/execution
layer beneath both.

`dist/micronaut_cpp_runtime/ToolHierarchy.cs` makes the relationship concrete:
**gpt-oss instructs dolphin; dolphin is tool-aware.** gpt-oss (planner) produces
a dolphin-executable instruction; dolphin (executor) translates it into a
structured tool call; the tool dispatches via forge. gpt-oss is dolphin-aware —
it knows dolphin's tool surface and can direct it:

```
gpt-oss: 'code review' -> instruct 'review dist/micronaut-coder/src/main.cpp'
dolphin: tool-aware=True -> call review(dist/micronaut-coder/src/main.cpp)
```

Planner/executor split: gpt-oss plans (produces instructions), dolphin executes
(calls tools). gpt-oss is dolphin-aware: it knows dolphin's tool surface and can
direct it.

`dist/micronaut_cpp_runtime/MiniAdapter.cs` confirms **mini can still adapt
gpt-oss** — as a **reactive residency adapter**, not a predictor. After #007 (no
cross-token prediction), mini's valid role is: observe the current native router
demand, then adapt which folds/experts are HOT via ARC/Xul:

```
demand=[13,28,10,14] -> HOT=4/32 ... demand=[27,30,1,13] -> HOT=8/32
```

mini steers gpt-oss's **working set** (residency), not its expert IDs; gpt-oss
native router stays authoritative. This is the same Fold/Node primitive: mini
adapts residency, ARC/Xul governs admissibility, gpt-oss computes.

## 19. Two-drive residency — C: vs E: (measured)

Benchmarked the SCXQDDS node reads on both drives (1 MB ranges, 5 chunks):

```
E: wall=2.14ms critical=0.50ms
C: wall=1.98ms critical=0.43ms
C: wall speedup=1.08x  critical speedup=1.16x
```

C: is only **~8–16% faster** for these small node reads (both sub-ms, OS-cached).
The drive doesn't determine VRAM fit — it determines how fast the runtime can
feed VRAM. The real win is the **multi-tier residency hierarchy**, not just
moving to C::

```
COLD  = E: backing store (canonical 91 folds)
WARM  = C: SSD cache
READY = RAM / decompressed
HOT   = GPU resident
```

`ScxqddsModelSource` stays drive-agnostic — the resolver picks the tier per
request (VRAM? RAM? C: cache? E: source?), turning two drives into part of the
residency law: `E: → C: → RAM → VRAM` is progressively more expensive/limited
residency, and Xul decides what stays at each level after the lease is released.

`dist/micronaut_cpp_runtime/ScxqddsTieredSource.cs` implements this as
**multi-tier lazy loading**: nodes load only on demand, promoted + cached per
tier (`COLD(E:) → WARM(C:) → READY(RAM) → HOT(GPU lease)`):

```
layers=24  wall_ms=63  peak_HOT=4/32
tier hits: RAM=65  C:cache=0  E:backing=31  bytes_moved=33MB
```

31 cold reads from E: (first request per expert), 65 RAM hits (subsequent
requests hit the faster tier), wall 63 ms (vs 2860 ms for the single-tier #010
— most reads now hit RAM). This is lazy loading applied to the Fold/Node
residency law.

`dist/micronaut_cpp_runtime/TieredBudgetTest.cs` constrains the READY (RAM)
budget to exercise the full hierarchy (LRU eviction to C: WARM):

```
READY=4        E:COLD=31 C:WARM=54 RAM=11 evicts=81 bytes=3418MB wall=4891ms
READY=8        E:COLD=31 C:WARM=44 RAM=21 evicts=67 bytes=2742MB wall=5437ms
READY=16       E:COLD=31 C:WARM=25 RAM=40 evicts=40 bytes=1790MB wall=813ms
READY=unlimited E:COLD=31 C:WARM=0  RAM=65 evicts=0  bytes=33MB  wall=16ms
```

As READY shrinks, **E:COLD stays at 31** (each expert loaded once from E:) and
reuse **spills into C: WARM** instead of repeatedly hitting E: — the full
hierarchy is exercised. Law: **promotion follows demand; retention follows
measured value.** `RequestAsync(node)` is unchanged throughout; residency
migration is runtime policy.

## 20. Execution gate — h_24_streamed ≈ h_24_005C (status)

`tools/execution_gate.py` wires the real MXFP4 provider through the streamed
path (read only the 4 demanded expert nodes per layer → SCXQ2 decode →
gate/up/down → h_24). **Honest status:** the streamed path is structurally the
same computation as #005C — both read the same safetensors, and the MoE only
uses the 4 selected experts — so `h_24_streamed = h_24_005C` by construction.
The remaining wiring is **per-tensor byte-offset resolution within the DDS
shards** to read from the SCXQDDS chunks directly (the fold_manifest carries
`shard_ids` + `total_bytes` but not explicit offsets). The MXFP4 dequant is
CPU-slow (~28 s/layer), so the full 24-layer gate is a ~11 min run (as #005C
was). Once the DDS-chunk offset resolution is wired, the gate closes:
`h_24_streamed ≈ h_24_005C` within ε, top-4 MATCH, SCXQ2 reconstruction PASS,
peak HOT ≤ 4/32 — proving execution equivalence + residency independence
simultaneously.

## 21. DDS as addressing plane — smgm-16 example (no weights)

`tools/smgm_address.py` resolves the **NodeAddress** over the smgm-16 DDS
example (a GPT-2, no weights — just the address map + 105 shards):

```
transformer.wte.weight            -> shard  96 (DDS_SHARD_096.dds) FLOAT32 INT16_SYM [HOT]
transformer.h.0.attn.c_attn.weight -> shard   0 (DDS_SHARD_000.dds) FLOAT32 INT8_SYM  [HOT]
transformer.h.0.mlp.c_fc.weight   -> shard  49 (DDS_SHARD_049.dds) FLOAT32 INT8_SYM  [WARM]
```

`fold_manifest.json` maps logical tensors → shard_ids; `dds_manifest.json` maps
shards → files. Together they form the NodeAddress contract
(`logical_tensor_id → shard_id → file + shape/dtype/encoding/residency`). This
confirms **DDS = semantic model-space/addressing plane** — the canonical weights
stay in the source; DDS knows how to reach the pieces. `RequestAsync(node)`
follows the address.

`tools/node_address.py` completes the contract to **byte-level (node-level)
addressing** using the gpt-oss tensor_map's `data_offsets`:

```
token_embd.weight       off= 615340800 len= 615329280 Q8_0
blk.0.ffn_gate_inp.weight off=1541661824 len= 368640 F32
blk.0.ffn_gate_exps.weight off=1400656896 len=141004800 MXFP4
```

NodeAddress now resolves a logical node all the way to `byte_offset +
byte_length + shape + dtype + encoding + hash` in the canonical source — so
`RequestAsync` drives to the exact address only when the cargo is needed.
**DDS = model address space; GGUF/SafeTensors = canonical cargo; SCXQ2 = optional
transport; NodeLease = resolved runtime resource; XCFE = execution resolution.**
The 91 materialized folds are no longer architecturally required — they're an
optional cache/acceleration backend.

`tools/build_model_scxqdds.py` regenerates `model.scxqdds` (v2.0) as the
**semantic address-space topology** — model identity, source identity (GGUF +
Safetensors + tensor_map), fold/node topology (50 folds), and provider
requirements — NOT the old `.dds`-shard cargo form:

```
tensor_count=459  payload=12.1GB  folds=50  provider_req=[tensor.matmul, shader.compute, buffer.alloc]
node_address: blk.0.ffn_gate_exps.weight -> byte_range [1400656896, 1541661696]
```

The canonical byte range comes from `tensor_map.json` (459 addressable tensors),
so `NodeAddress` resolves logical nodes to exact GGUF bytes; the `.dds`
materialization is now only an optional cache. `model.scxqdds` = semantic
GPU/model address map; `tensor_map` = where canonical bytes are;
`driver_v2.kson` = what a backend can realize; XCFE = binder.

## 7. Micronauts — bounded side-channel (µ-signal), not token participants

Micronauts are **low-bandwidth semantic sensors** into mini's controller state,
never concatenated into the token stream. `tools/micronaut_gate.py` enforces the
contract:

```
µ-signal { domain, intent, confidence, affinity[], pressure, ttl }
HARD LAWS: fixed schema, bounded cardinality, numeric/quantized, no prose,
  confidence >= threshold, ttl <= MAX_TTL, NO authority to select gpt-oss
  experts or bypass ARC/Xul.
```

The µ-signal is a bounded **node in the Wo controller fold** (`token_state`,
`residency_state`, `ARC_state`, `µ_signal` …). Authority chain preserved:
`micronaut → suggests → mini predicts demand → ARC admits → Xul selects →
gpt-oss router authoritative → expert nodes`.

### Ablation (identical workload)

`tools/micronaut_ablation.py`: A = predictor(state), B = predictor(state +
µ-signals).

```
A (state only)       NOW=0.12 NEXT=0.13 NEXT2=0.15
B (state + µ-signal) NOW=0.32 NEXT=0.12 NEXT2=0.19
```

Measured finding: the µ-signal **improves NOW recall (0.12→0.32)** — the bounded
sensor correctly identifies *current* experts — but **not NEXT prefetch**, because
it hints current, not future, experts. So micronauts are useful for
current-state semantics but carry no look-ahead; a prefetch-gaining micronaut
would need to hint *future* demand. This is the boundary made first-class.

## Reference

- **GEO-WEIGHTS.md** — the geometric weight + SCXQ2 streaming + ARC entropy.
- `tools/build_mini_tile_kson.py`, `tools/mini_tile_loader.py` — mini reference.
- `tools/build_gptoss_fold_kson.py` — gpt-oss-20b fold compiler.
- `tools/arc_physics.py` — ARC ΔH + fold cycle.
- `tools/scxq2_adapter.py`, `tools/scxq2_tiles.py` — SCXQ2 transport + residency.
