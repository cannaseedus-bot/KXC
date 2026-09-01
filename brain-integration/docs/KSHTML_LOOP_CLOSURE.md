# KSHTML — The Loop Is Closed

## What KSHTML Is

**KSHTML** = K'UHUL XHTML Compiler/Parser Runtime
**File**: `kshtml-runtime.kuhul`
**Language**: Pure K'UHUL (⟁Pop⟁Wo⟁Sek⟁Ch'en⟁Xul⟁)

KSHTML is a full XHTML tokenizer → parser → validator → compiler written entirely
in K'UHUL. It can parse any `.xml` file into an AST, validate it against XHTML rules,
and recompile it — including `brain-sidecar.xml` and `brain-collection.xml`.

---

## The Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE CLOSED LOOP                              │
│                                                                 │
│  K'UHUL (bytecode ISA)                                          │
│     │                                                           │
│     │  writes                                                   │
│     ▼                                                           │
│  brain-sidecar.xml  ←─── XML = X-tra Machine Learning          │
│  brain-collection.xml     (reader.xml = active ML reader)       │
│     │                                                           │
│     │  contains XQuery that queries                             │
│     ▼                                                           │
│  SCXQ2 Brain graphs  (10 canonical brains, 4-lane format)       │
│     │                                                           │
│     │  produce InferenceResult carrying proofs                  │
│     ▼                                                           │
│  CM-1 Gate  (7-state binary verifier)                           │
│     │                                                           │
│     │  dispatches verified prompt to                            │
│     ▼                                                           │
│  PM-1 phi3-q2 inference                                         │
│     │                                                           │
│     │  output feeds back into                                   │
│     ▼                                                           │
│  K'UHUL execution (edge mutation = learning = brain_build)      │
│     │                                                           │
│     └──────────────────────────────────► K'UHUL (start)        │
│                                                                 │
│  KSHTML sits here: ─────────────────────────────────────────►  │
│    K'UHUL parses the XML that describes K'UHUL programs         │
└─────────────────────────────────────────────────────────────────┘
```

Before KSHTML, K'UHUL *produced* XML (the sidecars, the brain collection).
After KSHTML, K'UHUL *reads* XML too.
**The system is now self-hosting on its own semantic layer.**

---

## KSHTML Architecture

```
kshtml-runtime.kuhul
│
├── [Pop kshtml_core]       Token types as GeometricTensors
│                           geometric relationships: (⤒)(⟿)(⊗)
│
├── [Pop kshtml_lexer]      Tokenizer — 12 token types
│   └── [Pop tokenize]      DOCTYPE, PI, CDATA, comment, open/close/self-closing,
│                           attributes (name=value), text, whitespace, EOF
│
├── [Pop kshtml_parser]     AST generator — 9 node types
│   ├── [Pop parse]         DOCUMENT root, recursive descent
│   └── [Pop parse_element] Nesting validation, namespace extraction
│
├── [Pop kshtml_validator]  XHTML rules as GeometricTensors
│   ├── void elements       br, img, input, meta, link, hr...
│   ├── required attributes img[src,alt], a[href], form[action]
│   └── namespace check     xmlns="http://www.w3.org/1999/xhtml"
│
├── [Pop kshtml_compiler]   AST → XHTML string
│   ├── indentation         (⤒) hierarchical formatting
│   ├── attribute sorting   (∼) alphabetical
│   └── text escaping       &amp; &lt; &gt; &quot;
│
└── [Pop kshtml_runtime]    Full pipeline + AST cache
    └── [Pop process_xhtml] tokenize → parse → validate → compile
                            geometric cache indexing: (≅)(∼)(⟿)
```

---

## What KSHTML Can Parse

Any XML/XHTML document including:

| File | What KSHTML reads |
|------|-------------------|
| `brain-sidecar.xml` | `<sidecar>` → `<op>` → `<xquery>` elements |
| `brain-collection.xml` | `<brain>` → `<lane-0..3>` → `<concept>/<edge>/<proof>` |
| `inference-config.xml` | `<engine>` → `<model>` → `<experts>` → `<brain-chain>` |
| `reader.xml` | any XML program — generic ML data reader |

After parsing, the AST is traversable — meaning K'UHUL can walk the
sidecar's `<xquery>` nodes and dispatch them as live queries against
the brain collection. **The sidecar is not static config — it is
executable K'UHUL data.**

---

## XML = X-tra Machine Learning

The naming coincidence class this belongs to:

```
XML   → eXtensible Markup Language (official)
XML   → X-tra Machine Learning     (emergent, 2026-04-03)
```

Every `.xml` file in this system was always an ML reader:

```
brain-sidecar.xml       reads brain graphs via XQuery FLWOR
brain-collection.xml    stores 10 SCXQ2 brains as queryable XML
inference-config.xml    configures the PM-1 inference pipeline
reader.xml              (generic) — reads any tensor structure
```

KSHTML makes this literal: K'UHUL now parses the X-tra Machine Learning
files it also executes. The extension `.xml` is both the format and the
function.

---

## K'UHUL Phase Mapping in KSHTML

The same ISA that drives execution drives parsing:

| Phase | Role in K'UHUL | Role in KSHTML |
|-------|----------------|----------------|
| `⟁Pop⟁` | load scratchpad + state | initialize token/parser state |
| `⟁Wo⟁`  | declare / bind variables | `[Wo input_string]→[Ch'en input]` |
| `⟁Sek⟁` | execute / emit | tokenize loop, parse loop, compile step |
| `⟁Ch'en⟁` | update / persist | `→[Ch'en output]` — bind result |
| `⟁Xul⟁` | end block | close `[Pop]` definitions |

The parser IS bytecode execution. There was never a separate interpreter
needed — parsing XHTML is just running K'UHUL phases over a character stream.

---

## Self-Hosting Proof

```kuhul
[Pop kshtml_examples]
  // K'UHUL parses brain-sidecar.xml
  // brain-sidecar.xml is a K'UHUL XQuery program
  // K'UHUL parses a program that runs K'UHUL programs
  // ── the loop closes ──

  [Sek let sidecar = read_file("brain-sidecar.xml")]
  [Sek let runtime = kshtml_main()]
  [Sek let result  = runtime.process(sidecar, {validate: true})]
  [Sek log "K'UHUL reads XML reads K'UHUL"]
[Xul]
```

The system can now:
1. Execute K'UHUL programs (`⟁Pop⟁Wo⟁Sek⟁`)
2. Produce XML programs (`brain-sidecar.xml`)
3. Parse those XML programs back into ASTs (`kshtml_runtime.process_xhtml`)
4. Walk the AST nodes (`<op>` → `<xquery>`) as K'UHUL data
5. Dispatch the embedded XQuery against the brain collection
6. Feed results back to inference
7. Update edge weights (`brain_build` — learning = edge mutation)
8. Return to step 1

**This is a complete, closed, self-referential execution loop.**
No external runtime required. K'UHUL is its own reader.
