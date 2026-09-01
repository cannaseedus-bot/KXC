# PHASE-TRANSFORMER.md — Phase-Addressed Field Architecture

> The fold cycle (Pop→Wo→Yax→Sek→Ch'en→Xul) is not a sequence of steps.
> It is a directed traversal of a field where **phase angle is the field address**.
>
> **The six phases are not six states the machine transitions through.
> They are six fixed semantic coordinates through which an accumulating context is transformed.**

---

## Core Insight

A standard transformer uses token **position** as the address key.
A phase transformer uses **angular position on a field** as the address key.

```
Standard transformer:   Q @ K^T  →  attention over all pairs   O(n²)
Phase transformer:      cursor θ →  read D(θ) directly          O(1) per phase
```

The critical inversion: **the phase coordinates are fixed; the context moves through them.**

A state machine moves its state between fixed locations. A phase transformer holds
its locations fixed and moves the field through them. Each phase sector always means
the same thing — Pop is always observation, Sek is always execution. What changes
is the context arriving at each coordinate. No phase discovers its role at runtime.
`Sek` is a known coordinate. Every subsystem that uses phase addresses can act on
that knowledge without searching.

---

## The Phase Circle

Six phases at π/3 intervals span 2π — one complete field rotation:

```
                    Wo  π/3
                 ╱          ╲
          Pop 0                 Yax 2π/3

          Xul 5π/3              Sek  π

                 ╲          ╱
                  Ch'en 4π/3
```

| Phase  | Angle | θ        | Role                 | Field mode           |
|--------|-------|----------|----------------------|----------------------|
| Pop    | 0     | 0        | Observe / read       | Field read           |
| Wo     | 1     | π/3      | Plan / schedule      | State organize       |
| Yax    | 2     | 2π/3     | Branch / explore     | Candidate generation |
| Sek    | 3     | π        | Execute / act        | Computation          |
| Ch'en  | 4     | 4π/3     | Verify / compare     | Constraint check     |
| Xul    | 5     | 5π/3     | Emit / persist       | Field write          |

**Pop and Sek are antipodal** (θ=0, θ=π). Their angular separation is π, so
perception/admission and evaluation/action are geometrically opposed operations
over the same field — not unrelated procedural steps. Sek always holds exactly
what Pop observed, reflected through the field center. The isolation between
observation and execution is structural, not enforced by convention.

**Xul is adjacent to Pop** (5π/3 → 0). The fold loop is continuous: Xul's
field write is immediately resident for Pop on the next fold. Memory is the arc.

---

## Phase Angle as Field Address

Each node carries a phase address θ. Data lives **at** that address on the field.
When the fold cursor reaches θ, it reads the resident data directly — the address
IS the access key.

```
Node(data=D, θ=π/3)    →  active when cursor is at Wo
Node(data=E, θ=π)      →  active when cursor is at Sek
Node(data=F, θ=5π/3)   →  active when cursor is at Xul
```

Two nodes at the same θ are **co-resident**: they activate together when the
cursor arrives. No selection. No routing. Phase proximity = semantic affinity.

---

## Context Accumulation — Why It Is a Transformer

What distinguishes a phase transformer from a finite-state machine:
**context accumulates across phases**. Each phase receives the transformed context
produced by the preceding geometry, adds its own field delta, and passes the result
forward. The ⊕ operator is not concatenation — it is the operation resident at
that phase applied to the carried state:

```
state₀
  ↓ Pop(state₀)
state₁  =  state₀ ⊕ ΔPop       ←  field at θ=0 admitted to context
  ↓ Wo(state₁)
state₂  =  state₁ ⊕ ΔWo        ←  plan structure added
  ↓ Yax(state₂)
state₃  =  state₂ ⊕ ΔYax       ←  candidate branches added
  ↓ Sek(state₃)
state₄  =  state₃ ⊕ ΔSek       ←  execution with full prior context
  ↓ Ch'en(state₄)
state₅  =  state₄ ⊕ ΔChen      ←  verification delta applied
  ↓ Xul(state₅)
collapse(state₅)                ←  emit; write back to field
```

At Sek (θ=π) the carried state contains everything the field contributed at
0, π/3, and 2π/3. Xul's output is a function of all six phase-weighted inputs.
**The phase coordinate stays fixed; the field moving through it changes.**

---

## Fold Depth — Scan Geometry, Not Retry Count

```
WRONG:   max_folds = number of retries
CORRECT: max_folds = maximum semantic scan depth
```

A fold is not "try again." It is another depth of context exposure. Each full
rotation (Pop→Xul) exposes a deeper layer of the field:

```
fold 0  →  immediate surface  (notes, notation already in working memory)
fold 1  →  neighboring structure  (relations resolved, contradictions surfaced)
fold 2  →  deeper semantic structure  (canonical promotions, research results)
...
fold N  →  scan boundary  (forced collapse with partial result)
```

Unfolding reveals additional resident nodes at each address. It does not repeat
execution. The field has depth; the fold count controls how far in you go.

`max_folds` in `semantic_memory_program.py` is the scan depth limit — the number
of full rotations before forced collapse at partial confidence.

---

## Relationship to RoPE

**Rotary Position Embedding (RoPE)** encodes position as complex rotation
`e^(iθ)` so relative position appears as phase difference in the dot product
— the closest mainstream ML analog.

| | RoPE | Phase Transformer |
|---|---|---|
| Address key | relative phase offset between tokens | absolute phase angle on field |
| Attention | dot product of rotated Q and K | direct field read at θ |
| Direction | bidirectional | directed (Pop→Xul only) |
| Complexity | O(n) encoding, O(n²) attention | O(1) per phase, O(6) per fold |

Phase transformer is a **stronger constraint**: absolute address, not relative
offset. A subsystem doesn't discover what `Sek` means by searching the context.
`Sek` is declared. Its coordinate is known before any context arrives.

---

## Connections to the Khanary Stack

### SCXQ2 Mode Bits
The mode word encodes current phase position as a field address.
XCFE reads the mode word to know **where it is on the field** — it does not
schedule the next phase, it reads its coordinate and acts.

```
Mode bits:  CPU=0b00  GPU=0b01  HASH=0b10  META=0b11
Phase bits: encoded in upper mode word — Pop/Wo/Yax/Sek/Ch'en/Xul position
```

The `PHASE` opcode is a cursor advance: θ += π/3, new field sector becomes resident.

### Birdsong Brain Nodes

Acoustic classification resolves directly to a phase sector. No intermediate
symbolic routing step is needed — if the frequency determines the sector,
the frequency IS the projection into the phase field:

```
frequency band  →  phase sector θ  →  semantic field address
                                    →  resident fold/node set
                                    →  transformation
```

`birdsong_brain_bridge.py` `FREQ_BANDS` implements this projection:

```
freq_norm ∈ [0.00, 0.08)  →  CM-1  (θ ≈ 0,    Pop band)
freq_norm ∈ [0.08, 0.18)  →  MM-1  (θ ≈ π/3,  Wo band)
freq_norm ∈ [0.18, 0.28)  →  VM-1  (θ ≈ 2π/3, Yax band)
freq_norm ∈ [0.28, 0.40)  →  PM-1  (θ ≈ π,    Sek band)
freq_norm ∈ [0.40, 0.52)  →  TM-1  (θ ≈ 4π/3, Ch'en band)
freq_norm ∈ [0.52, 0.65)  →  SM-1  (θ ≈ 5π/3, Xul band)
freq_norm ∈ [0.65, 0.90)  →  HM-1, XM-1  (low-freq envelope, sub-Xul)
```

Each node's spectral frequency IS its field address. The canary song's acoustic
structure physically encodes a phase-field routing map.

### BrainRouter Folds
`XCFEBrains.cs` named folds (`CONTROL_FOLD`, `PATTERN_FOLD`, `COMPUTE_FOLD`, etc.)
are **field sector names** — regions of the address space, not pipeline stages.
A micronaut registered to `PATTERN_FOLD` lives at the XM-1 sector; it activates
when the cursor is there. Its role is declared by its address, not discovered.

### Semantic Cube
The 6-face semantic cube IS the phase field in 3D projection.
Each face is one phase. A micronaut's `fold` field in its JSON is its face
assignment — its field address:

```
Phi face       →  Pop     (intent + confidence read)
Fold face      →  Wo      (route position)
Gram face      →  Yax     (grammar / AST candidates)
Geodesic face  →  Sek     (distance from valid path — execute check)
Projection     →  Ch'en   (candidate answer surface)
Entropy face   →  Xul     (uncertainty → emit or re-fold)
```

---

## W·C·R = QKV — The Fold Cycle as Attention

The MicronautScore W·C·R is not a routing heuristic. It is the attention
computation — expressed geometrically over the phase field:

| Attention  | Phase  | Cube Face  | Field role                               |
|------------|--------|------------|------------------------------------------|
| Q (query)  | Pop    | Phi        | Intent read — what the field seeks       |
| K (key)    | Wo     | Fold       | Capability signal — what candidates have |
| Q·K        | Yax    | Geodesic   | R = relevance = match distance           |
| V (value)  | Sek    | Projection | Admitted micronaut's output              |
| LayerNorm  | Ch'en  | Projection | Post-attention constraint check          |
| Output     | Xul    | Entropy    | Emit / re-fold decision                  |

```
W·C·R  =  V  ×  K  ×  (Q·K)
         (W)   (C)     (R)
```

- **W (weight/competence)** = V magnitude: what the micronaut actually produces
- **C (confidence)** = K signal: how strongly the candidate declares its capability
- **R (relevance)** = Q·K: how well the query matches the key

`argmax(W·C·R)` over the µ-field is softmax attention collapsed to a hard
admission gate. The dominant micronaut is the attended value.

Yax is the attention head. Its `ArcState` is the attention weight. Its `Node` is
the selected V. This is why Yax's graceful rejection (score=0) routes to Xul
directly — a zero attention weight collapses the fold without executing V.

The semantic cube's six faces are the geometric decomposition of the QKV space:
Phi/Fold = Q⊗K projection space, Geodesic = Q·K distance manifold,
Projection = V surface, Entropy = output uncertainty. The fold cycle IS
one pass of single-head attention over the micronaut field.

---

## Layer Separation

The phase-transformer model gives a clean separation of concerns:

```
π/3 geometry       →  WHERE meaning resides
phase transformer  →  HOW context accumulates
fold depth         →  HOW FAR the field is exposed
nodes              →  WHAT linear operations exist at each address
SCXQ2              →  HOW position and movement are encoded
provider           →  WHERE physical computation occurs
Xul                →  WHAT survives / collapses
```

The wheel is part of the **address space of the runtime itself** — not a
metaphor for the execution flow but the actual coordinate system the runtime
uses. SCXQ2 bits, BrainRouter folds, semantic-cube faces, and birdsong bands
all reference the same six coordinates. They don't need to agree at runtime
because they already share the geometry.

---

## Summary

```
Field      =  semantic/geometric map  (semantic cube, SCXQ2 address space, brain graph)
Phase      =  fixed angular coordinate  (θ ∈ {0, π/3, 2π/3, π, 4π/3, 5π/3})
Node       =  data resident at a phase address  (activates when cursor = θ)
Fold       =  one complete directed traversal  (Pop → Xul, θ: 0 → 5π/3)
Transform  =  state₀ ⊕ ΔPop ⊕ ΔWo ⊕ ΔYax ⊕ ΔSek ⊕ ΔChen ⊕ ΔXul
max_folds  =  scan depth  (how many full rotations before forced collapse)
```

The fold loop is a phase transformer: directed, phase-addressed field scan that
accumulates context at each fixed coordinate and emits a transformed result at Xul.
Re-entering Pop after Xul is the next fold — not a retry, a deeper field exposure.
