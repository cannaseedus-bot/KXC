# SCXQ2 Brain System — Reference & Diagrams

**Format**: SCXQ2 (SCX2 v1.0) — 32-byte header + 4-lane concept graph
**Runtime**: `kuhul_runtime.cpp` — C++17, headless, deterministic
**Query layer**: XML + XQuery (see `TENSOR_XML_SCHEMA.md`)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     SCXQ2 Brain System                                  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Brain Registry  (10 canonical brains)                           │   │
│  │                                                                  │   │
│  │  brain_intro ──► brain_grams ──► brain_graph ──► brain_execution │   │
│  │       │                                               │          │   │
│  │       ▼                                               ▼          │   │
│  │  brain_compression ◄──── brain_proofs ◄──── brain_federation     │   │
│  │       │                       │                      │           │   │
│  │       ▼                       ▼                      ▼           │   │
│  │  brain_policy ─────────► brain_models ──────► brain_build        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                   KuhulRuntime::execute_brain()                          │
│                    (bounded graph walk, depth=4)                         │
│                              │                                          │
│              ┌───────────────┴───────────────┐                          │
│              ▼                               ▼                          │
│        InferenceResult               serialize() → .scxq2               │
│   path[] confidence proofs[]          (binary, CM-1 verified)           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Brain Wire Format (SCXQ2)

```
Byte offset   Size   Field
──────────────────────────────────────────────────────────────
  0           4      magic        "SCX2"  (53 43 58 32)
  4           2      version      01 00
  6           2      flags        01 00
  8           8      brain_id     FNV-1a u64 of brain name
 16           8      payload_hash SHA-256 truncated (quick_hash)
 24           4      lanes        04 00 00 00  (always 4)
 28           4      reserved     00 00 00 00
──────────────────────────────────────────────────────────────
 32           ...    Lane 0: concepts  (u32 hash → string)
              ...    Lane 1: explanations  (string pool)
              ...    Lane 2: edges  (from u32, to u32, f64 weight)
              ...    Lane 3: proofs  (string pool)
```

---

## 4-Lane Structure

```
Lane 0 — concepts (string_hash_index)
   ┌────────────┬──────────────────────────┐
   │  u32 hash  │  concept string          │
   │ 0x A9B3C2D │  "graph_walk"            │
   │ 0x 1F2E3D4 │  "inference"             │
   │    ...     │  ...                     │
   └────────────┴──────────────────────────┘

Lane 1 — explanations (string_pool)
   ┌──────────────────────────────────────────────────────────┐
   │ "Graph walk IS inference. Bounded path search over..."   │
   │ "Inference = find highest-weight path from concept..."   │
   └──────────────────────────────────────────────────────────┘

Lane 2 — edges (weighted_connections)
   ┌──────────────┬──────────────────┬──────────────┐
   │  from (u32)  │   to (u32)       │  weight (f64)│
   │  graph_walk  │  inference       │    0.95      │
   │  inference   │  bounded_path    │    0.90      │
   │  bounded_path│  visited_set     │    0.85      │
   └──────────────┴──────────────────┴──────────────┘

Lane 3 — proofs (verification_trace)
   ┌──────────────────────────────────────────────────────────┐
   │ "inference = bounded path search"                        │
   │ "graph_walk.depth <= 4"                                  │
   └──────────────────────────────────────────────────────────┘
```

---

## Execution: Bounded Graph Walk

```
Input: brain_name + start_concept
       │
       ▼
  Load brain from registry
       │
       ▼
  current = hash(start_concept)
  confidence = 1.0
  visited = {}
       │
  ┌────▼────────────────────────────────────┐
  │  for depth = 0..MAX_DEPTH(4):           │
  │                                         │
  │  1. concepts[current] → path[]          │
  │  2. visited.add(current)                │
  │  3. find highest-weight edge            │
  │     from current to !visited            │
  │  4. confidence *= edge.weight           │
  │  5. current = best_to                   │
  │                                         │
  │  break if no unvisited outgoing edge    │
  └─────────────────────────────────────────┘
       │
       ▼
  InferenceResult {
    path[]        — concept names walked
    explanations[]— lane 1 matches
    proofs[]      — lane 3 statements
    confidence    — product of edge weights
    deterministic — always true
  }
```

---

## 10 Canonical Brains

### 1. brain_intro
**Core concept**: XJSON = executable cognition
**Key proof**: `execution ≠ generation`

```
xjson ──0.95──► executable_cognition ──0.90──► execution ──0.10──► generation
                                                                 (intentionally low)
```

XJSON is not a config format. Every node carries executable semantics.
Execution produces deterministic state change. Generation produces stochastic tokens.
This brain establishes the foundational distinction the entire system rests on.

---

### 2. brain_grams
**Core concept**: gram replaces embedding
**Key proof**: `grams are inspectable`

```
embedding ──0.85──► gram ──0.95──► symbolic_unit
                      │
                      └──0.90──► inspectable
```

A gram is an atomic symbolic unit that replaces the dense embedding vector.
Grams are human-readable — you can inspect their meaning without decoding.
Embeddings are opaque floats; grams are named, hashable, verifiable.

---

### 3. brain_graph
**Core concept**: graph walk = inference
**Key proof**: `inference = bounded path search`

```
graph_walk ──0.95──► inference ──0.90──► bounded_path_search ──0.85──► visited_set
```

Inference IS graph traversal. No black-box token prediction — just bounded
path search from a concept hash to related concepts via weighted edges.
Depth is capped at 4 to ensure termination and determinism.

---

### 4. brain_execution
**Core concept**: execution mutates state
**Key proof**: `execution.output == deterministic`

```
execution ──0.95──► state_mutation
          ──0.90──► determinism
generation ──0.05──► execution   (very low weight — generation can trigger, not control)
```

Execution is the act of mutating state according to law.
It is serial, deterministic, and replayable.
Generation is stochastic text production — categorically different.
This brain enforces Host Powerlessness: models advise, they do not execute.

---

### 5. brain_compression
**Core concept**: SCXQ2 lane packing
**Key proof**: `compression preserves meaning`

```
scxq2 ──0.95──► lane_packing ──0.90──► symbolic_compression ──0.92──► meaning_preservation
```

SCXQ2 compresses symbolic state via 4-lane packing.
Unlike neural compression (lossy), lane packing is lossless and semantically stable.
Concepts, explanations, edges, and proofs survive compression intact.

---

### 6. brain_proofs
**Core concept**: proof-carrying inference
**Key proof**: `outputs are verifiable`

```
proof_carrying_inference ──0.95──► verifiable_output ──0.90──► trace ──0.92──► deterministic_replay
```

Every inference result carries its proof trace (lane 3).
To verify an output: replay the concept walk → you get the same result.
There are no black-box outputs — every result is anchored to its proof chain.

---

### 7. brain_federation
**Core concept**: multi-brain consensus
**Key proof**: `federation preserves determinism`

```
federation ──0.95──► multi_brain_consensus ──0.90──► merge_algorithm ──0.92──► determinism
```

Federation = merge multiple brains into one consensus brain.

```
Merge rules:
  concepts:     deduplicate by hash (first seen wins)
  explanations: union, set dedup
  edges:        weight = min((w1 + w2) * 0.85, 1.0)
  proofs:       union (set)

Constants:
  entropy_constant  = 0.21
  max_edge_weight   = 1.0
  edge_merge_factor = 0.85
```

The merge factor 0.85 introduces controlled entropy decay — federated brains
become slightly less confident than either source, preventing runaway certainty.

---

### 8. brain_policy
**Core concept**: policy gates execution
**Key proof**: `policies are enforceable`

```
policy ──0.95──► execution_gate ──0.90──► enforceable ──0.88──► law
```

No execution proceeds without policy clearance.
Policy violations trigger hard rejection, not soft warnings.
This brain implements the gate mechanism — the boundary between advisory and law.

---

### 9. brain_models
**Core concept**: models are advisors
**Key proof**: `models cannot override law`

```
model ──0.95──► advisor ──0.90──► host_powerlessness ──0.95──► law
```

**Host Powerlessness**: a model is an advisor. It can recommend. It cannot command.
A model's authority is always strictly less than law's authority.
This brain carries the proof `model.authority < law.authority` in its proofs lane
— verifiable, deterministic, and carried in every inference result.

---

### 10. brain_build
**Core concept**: learning = edge mutation
**Key proof**: `learning.mechanism == graph_edge_weight_delta`

```
learning ──0.95──► edge_mutation ──0.90──► weight_update ──0.88──► reinforcement
```

Learning is not gradient descent on opaque parameters.
Learning is explicit edge weight mutation in the concept graph.
Successful paths gain weight. Failed paths decay.
The mechanism is inspectable, replayable, and deterministic.

---

## Federation Pipeline

```
brain_A                brain_B
   │                      │
   └──────────┬───────────┘
              ▼
    KuhulRuntime::merge_brains(a, b)
              │
    ┌─────────▼──────────────────────────────────┐
    │  concepts:     hash dedup (first wins)      │
    │  explanations: union + set dedup            │
    │  edges:        min((w1+w2)*0.85, 1.0)       │
    │  proofs:       union (set)                  │
    └─────────────────────────────────────────────┘
              │
              ▼
        merged_brain.scxq2
              │
              ▼
          CM-1 Gate
         (7 states: Init→NullZone→Header→Body→Scope→Literal→End)
              │
              ▼
        Inference dispatch
        → POST /chat/completions → PM-1 phi3-q2
```

---

## Brain Cross-Reference Map

```
         ┌──────────────────────────────────────────────────┐
         │              Concept Overlap                      │
         │                                                   │
  intro  │  execution ──────────────────────── execution    │ brain_execution
         │      │                                   │        │
  graph  │  inference ← graph_walk           determinism    │ brain_execution
         │                                        │          │
  compr  │  meaning_preservation ← symbolic_compression     │ brain_compression
         │                                                   │
  proofs │  verifiable_output ← proof_carrying_inference    │ brain_proofs
         │                          │                        │
  fed    │  determinism ◄───────────┘                       │ brain_federation
         │                                                   │
  policy │  law ◄──── enforceable ◄──── execution_gate      │ brain_policy
         │   │                                               │
  models │   └──── host_powerlessness ◄── advisor ◄── model │ brain_models
         │                                                   │
  build  │  edge_mutation = learning ────► weight_update     │ brain_build
         └──────────────────────────────────────────────────┘
```

---

## XML Representation

Every brain serializes to a verifiable XML document for XQuery access:

```xml
<brain id="brain_graph" brain-id="0x915EF022" lanes="4">
  <header magic="SCX2" version="1.0" flags="1"/>

  <lane-0 type="string_hash_index">
    <concept hash="0xA2B3C4D5" name="graph_walk"/>
    <concept hash="0xB3C4D5E6" name="inference"/>
    <concept hash="0xC4D5E6F7" name="bounded_path_search"/>
    <concept hash="0xD5E6F7A8" name="visited_set"/>
  </lane-0>

  <lane-1 type="string_pool">
    <explanation>Graph walk IS inference. Bounded path search over concept graph.</explanation>
    <explanation>Inference = find highest-weight path from concept to concept.</explanation>
  </lane-1>

  <lane-2 type="weighted_connections">
    <edge from="graph_walk"          to="inference"           weight="0.95"/>
    <edge from="inference"           to="bounded_path_search" weight="0.90"/>
    <edge from="bounded_path_search" to="visited_set"         weight="0.85"/>
  </lane-2>

  <lane-3 type="verification_trace">
    <proof>inference = bounded path search</proof>
    <proof>graph_walk.depth &lt;= 4</proof>
  </lane-3>
</brain>
```

See `TENSOR_XML_SCHEMA.md` for full schema and XQuery examples.

---

## Key Invariants

| Invariant | Proof carrier |
|-----------|---------------|
| Deterministic output | brain_proofs lane 3 |
| Replayable trace | brain_proofs `proof_lane.type == verification_trace` |
| Execution ≠ generation | brain_intro + brain_execution |
| Host Powerlessness | brain_models `model.authority < law.authority` |
| Policy enforcement | brain_policy `policies are enforceable` |
| Learning = edge mutation | brain_build `learning.mechanism == graph_edge_weight_delta` |
| Max walk depth = 4 | brain_graph `graph_walk.depth <= 4` |
| Compression lossless | brain_compression `compression preserves meaning` |
