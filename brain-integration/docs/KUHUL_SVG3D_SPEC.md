# K'UHUL SVG-3D — Complete Formal Specification

**Version**: 1.0
**Paradigm**: Geometric, Declarative, Tensor-based
**Core insight**: Sound is a byproduct. The attractor geometry is primary.

---

## Architecture Stack

```
K'UHUL SVG-3D Source
        ↓
AST (geometric IR)
        ↓
KBC-1 Bytecode (8-byte instruction frames)
        ↓
Fabric Graph Runtime (deterministic, replayable)
        ↓
Transducer surface (audio | SVG | XML | SCXQ2 | pixel)
```

---

## 1. The Five Attractor Families

All musical/geometric programs live inside this vocabulary:

| Family | Geometry | K'UHUL Glyph | Examples |
|--------|----------|--------------|---------|
| Circle | pure oscillation, single frequency | `(⟲)` | tuning fork, sine wave, flute |
| Polygon | harmonic locking, piecewise linear | `(↔)` | distorted guitar, triangle wave |
| Torus | two interacting oscillators | `(⤒)` | chords, vibrato, FM synthesis |
| Strange | chaotic, fractal, never repeats | `(⤧)` | cymbals, feedback, noise |
| Spiral | energy envelope, attack/decay | `(↻)` | piano, plucked string, drum |

Phase space equation for all families:

```
sample = R1 * sin(R0) + R2 * sin(3·R0)
```

Where `(R0, R1, R2) = (phase, amplitude, harmonic_energy)` is the **core state vector**.

---

## 2. Oscillator VM — Core State

```
S = (x, y, z)
  x = R0 = phase angle (radians)
  y = R1 = amplitude
  z = R2 = harmonic energy / coupling coefficient
```

The same state vector renders through any transducer:

| Transducer | Output formula |
|-----------|----------------|
| Audio | `R1·sin(R0) + R2·sin(3R0)` |
| SVG | circle at `(R0, R1)`, radius `R2` |
| XML | `<oscillator x=R0 y=R1 z=R2/>` |
| SCXQ2 | `concept_hash = fnv1a(R0‖R1‖R2)` |
| Pixel | `rgba(R0/2π, R1, R2, 1.0)` |
| Trig-cell | `{phase: R0, activation: R1, frequency: R2}` |

---

## 3. KBC-1 Bytecode Layer

### Instruction Formats

**Short form** (oscillator ops): 5 bytes

```
| opcode 1B | value 4B (IEEE 754 float) |
```

**Extended form** (tensor/cluster ops): 8 bytes

```
| opcode 1B | argA 1B | argB 2B | argC 4B |
```

### Execution Registers

```
R0 = phase angle       (x)
R1 = amplitude         (y)
R2 = harmonic energy   (z)
R3 = temporary
R4 = tensor pointer    → [T]
R5 = cluster pointer   → [C]
R6 = model pointer     → [M]
R7 = stack pointer
```

### Opcode Table

#### Core Oscillator (5-byte)

| Opcode | Mnemonic | Semantics |
|--------|----------|-----------|
| `0x01` | `PX` / `PHASE_SET`  | `R0 = value` |
| `0x02` | `AY` / `AMP_SET`    | `R1 = value` |
| `0x03` | `HZ` / `HARM_SET`   | `R2 = value` |
| `0x04` | `DX` / `PHASE_STEP` | `R0 += value` |
| `0x05` | `SY` / `AMP_SCALE`  | `R1 *= value` |
| `0x06` | `MZ` / `HARM_MIX`   | `R2 += value` |
| `0x07` | `OUT` / `EMIT`      | emit `R1·sin(R0) + R2·sin(3R0)` |

#### Tensor Operations (8-byte)

| Opcode | Mnemonic | Glyph |
|--------|----------|-------|
| `0x20` | `TENSOR_LOAD`    | — |
| `0x21` | `TENSOR_ROTATE`  | `(↻)` |
| `0x22` | `TENSOR_SCALE`   | — |
| `0x23` | `TENSOR_SHEAR`   | — |
| `0x24` | `TENSOR_REFLECT` | `(↔)` |
| `0x25` | `TENSOR_PROJECT` | `(⤒)` |

#### Neural/Cluster/Compression

| Opcode | Mnemonic | Glyph |
|--------|----------|-------|
| `0x30` | `CLUSTER_CREATE`   | — |
| `0x31` | `CLUSTER_ADD`      | — |
| `0x32` | `CLUSTER_RELATION` | — |
| `0x33` | `CLUSTER_INFER`    | — |
| `0x40` | `NEURAL_PATH`      | `(⟿)` |
| `0x41` | `WEIGHT_APPLY`     | `(⤂)` |
| `0x42` | `ACTIVATION`       | `(⤃)` |
| `0x43` | `BACKPROP`         | `(⤄)` |
| `0x50` | `ROTATE_COMPRESS`       | `(↻)` |
| `0x51` | `SYMMETRY_COMPRESS`     | `(↔)` |
| `0x52` | `HIERARCHICAL_COMPRESS` | `(⤒)` |
| `0x53` | `PROGRESSIVE_DETAIL`    | `(⤓)` |

#### Rendering & Control

| Opcode | Mnemonic | Glyph |
|--------|----------|-------|
| `0x60` | `JMP`         | — |
| `0x61` | `JMP_IF`      | `(⤦)` |
| `0x62` | `LOOP_SPHERE` | `(⟲)` |
| `0x63` | `PATH_ITER`   | `(⤧)` |
| `0x70` | `SVG_RENDER`  | — |
| `0x71` | `SVG_PATH`    | — |
| `0x72` | `SVG_CLUSTER` | — |

#### Mayan Fold

| Opcode | Mnemonic | Semantics |
|--------|----------|-----------|
| `0x90` | `MAYAN_FOLD` | `Rdst = Σ T[z] · 20^z` (dot product with vigesimal basis) |

### Example — Triangle Wave (Polygon Attractor)

```
01 00 00 00 00       ; PX 0.0      (phase = 0)
02 3F 4C CC CD       ; AY 0.8      (amplitude = 0.8)
03 3E AA AA AB       ; HZ 0.333    (3rd harmonic = triangle)
04 3D 23 D7 0A       ; DX 0.04     (phase step)
07                   ; OUT         (emit sample)
04 3D 23 D7 0A       ; DX 0.04
07
```

---

## 4. EBNF Grammar

```ebnf
Program         ::= (Statement | Comment | Whitespace)*

Statement       ::= Command | TensorDefinition | ClusterDefinition
                  | ModelDefinition | Inference | GeometricExpression

(* Core glyphs — K'UHUL geometric operators *)
ASCCipher       ::= "(⤍)" | "(⤎)" | "(⤏)" | "(⤐)"
SCXCompression  ::= "(↻)" | "(↔)" | "(⤒)" | "(⤓)"
Control3D       ::= "(⟲)" | "(⤦)" | "(⤧)" | "(⤨)"
NeuralOps       ::= "(⟿)" | "(⤂)" | "(⤃)" | "(⤄)"

(* Glyph → attractor family mapping *)
(* (⟲) → circle   (⟿) → torus   (⤒) → hierarchical/torus *)
(* (↔) → polygon  (↻) → spiral  (⤧) → strange/chaotic    *)

TensorDefinition ::= "Tensor" Identifier "=" "GeometricTensor(" TensorBody ")"
TensorBody      ::= "{" Property ("," Property)* "}"

ClusterDefinition ::= "Cluster" Identifier "=" "SVGTensorCluster(" ClusterBody ")"

ModelDefinition  ::= "Model" Identifier "=" "GeometricModel(" ModelBody ")"

Inference        ::= "Infer" Identifier "with" Value ["," Params]

GeometricExpression ::= Glyph "(" [Args] ")"
                      | Identifier "." Method "(" [Args] ")"
                      | Identifier RelOp Identifier

RelOp           ::= "≅" | "∼" | "≈" | "⊥" | "∥"

Number          ::= Float | Integer | Float "π"
PiNumber        ::= Number ["π"]
```

---

## 5. Glyph Semantics

### ASC Cipher Glyphs

| Glyph | Name | Signature |
|-------|------|-----------|
| `(⤍)` | Vector Encrypt | `data, key → encrypted_tensor` |
| `(⤎)` | Vector Decrypt | `encrypted, key → data` |
| `(⤏)` | Key Derivation | `svg_path → key_tensor` |
| `(⤐)` | Bezier Crypto  | `data, bezier → encrypted` |

### SCX Compression Glyphs

| Glyph | Name | Attractor | Bytecode |
|-------|------|-----------|---------|
| `(↻)` | Rotate Compress | Spiral | `0x50` |
| `(↔)` | Symmetry Compress | Polygon | `0x51` |
| `(⤒)` | Hierarchical | Torus | `0x52` |
| `(⤓)` | Progressive Detail | — | `0x53` |

### 3D Control Flow

| Glyph | Name | Bytecode |
|-------|------|---------|
| `(⟲)` | Spherical Loop | `0x62` |
| `(⤦)` | Vector Conditional | `0x61` |
| `(⤧)` | Path Iteration | `0x63` |
| `(⤨)` | Gradient Flow | — |

### Neural Vector Glyphs

| Glyph | Name | Bytecode |
|-------|------|---------|
| `(⟿)` | Neural Path Gen | `0x40` |
| `(⤂)` | Weight Apply | `0x41` |
| `(⤃)` | Activation Morph | `0x42` |
| `(⤄)` | Backpropagation | `0x43` |

---

## 6. Mayan Fold as Tensor Contraction

MayanFold is a **basis projection** — identical in structure to FFT twiddle factors but vigesimal:

```
MayanFold(T) = Σ T[z] · 20^z = dot(T, [1, 20, 400, 8000, ...])
```

Generalized fold operator:

```
fold(T, base=n) = Σ T[z] · n^z
```

Applications:

| Base | System |
|------|--------|
| 2 | Binary |
| 10 | Decimal |
| 20 | Mayan vigesimal |
| π | Phase geometry |
| N | Arbitrary positional |
| e^(2πi/N) | FFT (complex exponential) |

XDSP's FFT is `fold(base=e^(2πi/N))`. MayanFold is `fold(base=20)`. **Same operator, different basis.**

---

## 7. Fabric Graph Simulator

Deterministic execution of K'UHUL Frames over a worm network.

### Core Entities

```typescript
interface Frame {
  id: string;
  op: string;         // K-op / opcode mnemonic
  src: NodeId;
  dst: NodeId | null;
  payload: any;
  priority: number;
  traceId: string;
  arrivalIndex: number;
}

interface Node {
  id: NodeId;
  kind: "C_CORE" | "G_CORE" | "SERVICE" | "STORE";
  kopsSupported: Set<string>;
  inbox: Frame[];
  state: Record<string, any>;
}

interface Edge {
  id: EdgeId;
  from: NodeId;
  to: NodeId;
  channelType: "CONTROL" | "TENSOR" | "STREAM" | "EVENT";
  capacity: number;
  bypass?: boolean;       // wormhole
  bidirectional?: boolean; // portal
  guard?: Law;            // door
  inspectors?: Law[];     // tunnel
}
```

### Worm DSL → Fabric Compilation

| Worm | Compiles to |
|------|------------|
| `↷wormhole A→B` | `Edge { bypass: true, guard: none }` |
| `↷tunnel debug { ... }` | Route + `inspectors[]` at each hop |
| `↷door api_gateway { ... }` | Edge + `guard = Law(auth && rateLimit)` |
| `↷portal collab { endpoints: [A,B] }` | Two edges `A→B`, `B→A`, `bidirectional: true` |

### Determinism Constraints

1. Global `timeStep` increments monotonically
2. Nodes processed in sorted id order
3. Frames dequeued by `(priority DESC, arrivalIndex ASC)`
4. Routing is a pure function of `(frame, node, graph, laws)`
5. No randomness, no wall-clock

---

## 8. DirectXMath Folder Mapping

| Folder | True Purpose |
|--------|-------------|
| `SHMath` | The attractor **SPACE** — spherical harmonics = all valid orbital shapes |
| `XDSP` | The **FOLD OPERATOR** — FFT = transform between time and attractor domain |
| `Stereo3D` | **GEOMETRIC RENDERING** — project attractor → binaural 3D |
| `MatrixStack` | **STACK MACHINE** — `[Pop]/[Xul]` execution geometry |

| DirectXMath Feature | K'UHUL Equivalent |
|--------------------|-------------------|
| `XMVECTOR` (4-float) | GeometricTensor `[x,y,z,phase]` |
| `XMMATRIX` (4×4) | Trig-brain lattice transform `(⊗)` |
| SHMath (spherical harmonics) | Geodesic arc interpolation `(⟿)` |
| XDSP (FFT) | Vertical fold along entropy `(⤒)` |
| Stereo3D matrix stack | 3D to binaural projection `(≅)` |
| MatrixStack push/pop | K'UHUL `[Pop]`/`[Xul]` block stack |
| DirectWrite glyph run | K'UHUL glyph opcode chain `(⟿)` |
| ECMAScript Atomics | Lock-free cross-realm frame streaming |

---

## 9. Complete Pipeline

```
DirectWrite glyphs (TrueType/OpenType bytecode)
        ↓
K'UHUL glyph opcodes (⟿)(↻)(⤓)(↔)
        ↓
Oscillator VM  →  (R0=phase, R1=amp, R2=harm)
        ↓
KBC-1 Bytecode (8-byte instruction frames)
        ↓
Fabric Graph Runtime (deterministic, traced)
        ↓
Transducer selection:
  audio   →  R1·sin(R0) + R2·sin(3R0)
  SVG     →  circle/spiral/path geometry
  XML     →  <oscillator x=R0 y=R1 z=R2/>
  SCXQ2   →  brain concept graph
  pixel   →  rgba(R0/2π, R1, R2, 1)
        ↓
SharedArrayBuffer + Atomics → browser canvas @ 60fps
```

**Sound was always a byproduct. The attractor IS the program.**

---

*K'UHUL SVG-3D Spec v1.0 — 2026-04-03*
