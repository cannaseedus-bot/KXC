// kxc.js — K'uhul Compiler: .kxx/.kslx → .cpp/.hlsl (DirectX)
//
// Compilation pipeline:
//   .kxx  (K'uhul host code)   → .cpp  (D3D12 C++ host)
//   .kslx (K'uhul shader code) → .hlsl (HLSL compute shader)
//   .kuhul (combined source)   → both above
//
// Mapping:
//   K'uhul control        DirectX target
//   [Pop]...[Xul]    →   class + Execute() function
//   [Wo]             →   CreateCommittedResource (RWStructuredBuffer)
//   [Yax]            →   buffer load / SRV bind
//   [Ch'en]          →   buffer store / UAV bind
//   [Sek]            →   Dispatch or HLSL intrinsic
//   [K'ayab']        →   for loop / outer Dispatch loop
//   [Kumk'u]         →   ResourceBarrier + advance loop
//   [Muwan]          →   cmdList->Dispatch(x, y, z)
//   π-phase 0        →   SetPipelineState + bind resources
//   π-phase π/4      →   GroupMemoryBarrierWithGroupSync()
//   π-phase π/2      →   compute kernel body
//   π-phase 3π/4     →   UAV stores
//   π-phase π        →   ResourceBarrier completion
//
//   ⊗  →  mul() / matmul
//   ⊕  →  add / translate
//   ⊖  →  sub
//   ⊘  →  mul by scalar / divide
//   ⊛  →  convolution
//   ⊜  →  branch / constraint check (if)
//   ⊝  →  clamp / ReLU projection
//   ⊞  →  add bias / compose
//
// File extensions:
//   .kxx   K'uhul host code      → .cpp
//   .kslx  K'uhul shader code    → .hlsl
//   .kuhul combined              → .cpp + .hlsl

// ─── Token types ─────────────────────────────────────────────────────────────

const T = Object.freeze({
  POP: 'Pop', XUL: 'Xul', WO: 'Wo', YAX: 'Yax', CHEN: "Ch'en",
  SEK: 'Sek', KAYAB: "K'ayab'", KUMKU: "Kumk'u", MUWAN: 'Muwan',
  GEO_OP: 'geo_op', IDENT: 'ident', ARROW: '->', NUMBER: 'number', EOF: 'EOF',
});

const GEO_OPS = new Set(['⊗','⊕','⊖','⊘','⊛','⊜','⊝','⊞']);

const GEO_TO_HLSL = Object.freeze({
  '⊗': 'mul',
  '⊕': '+',
  '⊖': '-',
  '⊘': '/',
  '⊛': 'convolve',
  '⊜': '==',
  '⊝': 'max(0.0f, ',
  '⊞': '+',
});

// ─── Lexer ────────────────────────────────────────────────────────────────────

export class KXCLexer {
  constructor(src) {
    this._src  = src;
    this._pos  = 0;
    this._toks = [];
  }

  tokenize() {
    while (this._pos < this._src.length) {
      this._skipWS();
      if (this._pos >= this._src.length) break;
      const ch = this._src[this._pos];

      if (ch === '#') { this._skipLine(); continue; }
      if (ch === '/' && this._src[this._pos+1] === '/') { this._skipLine(); continue; }

      // K'uhul glyph: [Word]
      if (ch === '[') {
        const end = this._src.indexOf(']', this._pos);
        if (end !== -1) {
          const word = this._src.slice(this._pos + 1, end);
          const type = Object.values(T).includes(word) ? word : T.IDENT;
          this._toks.push({ type, value: word });
          this._pos = end + 1;
          continue;
        }
      }

      // Arrow
      if (ch === '-' && this._src[this._pos+1] === '>') {
        this._toks.push({ type: T.ARROW, value: '->' });
        this._pos += 2;
        continue;
      }

      // Geo operator
      if (GEO_OPS.has(ch)) {
        this._toks.push({ type: T.GEO_OP, value: ch });
        this._pos++;
        continue;
      }

      // Identifier or keyword
      if (/[A-Za-z_⟁⊛]/.test(ch)) {
        let id = '';
        while (this._pos < this._src.length && /[A-Za-z0-9_'.=()⊛]/.test(this._src[this._pos]))
          id += this._src[this._pos++];
        this._toks.push({ type: T.IDENT, value: id });
        continue;
      }

      // Number
      if (/[0-9]/.test(ch)) {
        let n = '';
        while (this._pos < this._src.length && /[0-9.,]/.test(this._src[this._pos]))
          n += this._src[this._pos++];
        this._toks.push({ type: T.NUMBER, value: n });
        continue;
      }

      this._pos++; // skip unknown
    }
    this._toks.push({ type: T.EOF, value: '' });
    return this._toks;
  }

  _skipWS()   { while (this._pos < this._src.length && /\s/.test(this._src[this._pos])) this._pos++; }
  _skipLine() { while (this._pos < this._src.length && this._src[this._pos] !== '\n') this._pos++; }
}

// ─── Parser ───────────────────────────────────────────────────────────────────

export class KXCParser {
  constructor(tokens) {
    this._t   = tokens;
    this._pos = 0;
  }

  parse() {
    const nodes = [];
    while (this._cur().type !== T.EOF) {
      const n = this._parseStmt();
      if (n) nodes.push(n);
    }
    return { type: 'Program', body: nodes };
  }

  _cur()  { return this._t[this._pos]; }
  _next() { return this._t[this._pos++]; }
  _peek() { return this._t[this._pos + 1]; }

  _parseStmt() {
    const tok = this._cur();
    switch (tok.type) {
      case T.POP:    return this._parseKernel();
      case T.WO:     return this._parseAlloc();
      case T.YAX:    return this._parseLoad();
      case T.CHEN:   return this._parseStore();
      case T.SEK:    return this._parseSek();
      case T.KAYAB:  return this._parseLoop();
      case T.MUWAN:  return this._parseDispatch();
      case T.IDENT:  this._next(); return { type: 'Ident', value: tok.value };
      default:       this._next(); return null;
    }
  }

  _parseKernel() {
    this._next(); // consume Pop
    const name = this._cur().type === T.IDENT ? this._next().value : 'Kernel';
    const body = [];
    while (this._cur().type !== T.XUL && this._cur().type !== T.EOF)
      body.push(this._parseStmt());
    if (this._cur().type === T.XUL) this._next();
    return { type: 'Kernel', name, body };
  }

  _parseAlloc() {
    this._next(); // consume Wo
    const name = this._cur().type === T.IDENT ? this._next().value : 'buf';
    // parse shape= or size= attrs
    let shape = null;
    if (this._cur().type === T.IDENT && this._cur().value.startsWith('shape='))
      shape = this._next().value.replace('shape=', '');
    return { type: 'Alloc', name, shape };
  }

  _parseLoad() {
    this._next();
    const name = this._cur().type === T.IDENT ? this._next().value : 'buf';
    // parse optional -> Sek chain
    if (this._cur().type === T.ARROW) {
      this._next();
      const sek = this._parseStmt();
      return { type: 'Chain', load: name, op: sek };
    }
    return { type: 'Load', name };
  }

  _parseStore() {
    this._next();
    const name = this._cur().type === T.IDENT ? this._next().value : 'buf';
    return { type: 'Store', name };
  }

  _parseSek() {
    this._next();
    // [Sek geoOp operand] or [Sek op_name]
    if (this._cur().type === T.GEO_OP) {
      const op  = this._next().value;
      const rhs = this._cur().type === T.IDENT ? this._next().value : null;
      return { type: 'GeoOp', op, rhs };
    }
    const name = this._cur().type === T.IDENT ? this._next().value : 'op';
    return { type: 'SekOp', name };
  }

  _parseLoop() {
    this._next();
    const name = this._cur().type === T.IDENT ? this._next().value : 'loop';
    const body = [];
    while (this._cur().type !== T.KUMKU && this._cur().type !== T.EOF)
      body.push(this._parseStmt());
    if (this._cur().type === T.KUMKU) this._next();
    return { type: 'Loop', name, body };
  }

  _parseDispatch() {
    this._next();
    const name  = this._cur().type === T.IDENT ? this._next().value : 'kernel';
    const x     = this._cur().type === T.NUMBER ? parseInt(this._next().value) : 1;
    const y     = this._cur().type === T.NUMBER ? parseInt(this._next().value) : 1;
    const z     = this._cur().type === T.NUMBER ? parseInt(this._next().value) : 1;
    return { type: 'Dispatch', name, x, y, z };
  }
}

// ─── HLSL code generator ──────────────────────────────────────────────────────

export class HLSLCodegen {
  constructor() {
    this._indent = 0;
    this._bufferIdx = 0;
    this._buffers = [];
  }

  generate(ast) {
    const lines = [];
    for (const node of ast.body) {
      const code = this._genNode(node);
      if (code) lines.push(code);
    }
    const bufDecls = this._buffers.map((b, i) =>
      `RWStructuredBuffer<float> ${b} : register(u${i});`
    ).join('\n');
    return bufDecls + (bufDecls ? '\n\n' : '') + lines.join('\n');
  }

  _genNode(n) {
    if (!n) return '';
    switch (n.type) {
      case 'Kernel':   return this._genKernel(n);
      case 'Alloc':    return this._genAlloc(n);
      case 'Load':     return `float _v_${n.name} = ${n.name}[dtid.x];`;
      case 'Store':    return `${n.name}[dtid.x] = _v_${n.name};`;
      case 'GeoOp':    return this._genGeoOp(n);
      case 'SekOp':    return this._genSekOp(n);
      case 'Loop':     return this._genLoop(n);
      case 'Dispatch': return this._genDispatch(n);
      case 'Chain':    return this._genChain(n);
      default:         return '';
    }
  }

  _genKernel(n) {
    const body = n.body.map(c => this._genNode(c)).filter(Boolean)
      .map(l => '    ' + l).join('\n');
    return [
      `[numthreads(16, 16, 1)]`,
      `void ${n.name}(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID) {`,
      body,
      `}`,
    ].join('\n');
  }

  _genAlloc(n) {
    this._buffers.push(n.name);
    return `// Buffer allocated: ${n.name}${n.shape ? ` shape=${n.shape}` : ''}`;
  }

  _genGeoOp(n) {
    const hlsl = GEO_TO_HLSL[n.op];
    if (n.op === '⊗') return `float _result = mul(_v_input, _v_${n.rhs ?? 'W'});`;
    if (n.op === '⊝') return `float _result = max(0.0f, _v_${n.rhs ?? 'z'});  // ReLU projection`;
    if (n.op === '⊜') return `GroupMemoryBarrierWithGroupSync(); // pi/4 phase sync`;
    return `float _result = _v_lhs ${hlsl} _v_${n.rhs ?? 'rhs'};`;
  }

  _genSekOp(n) {
    if (n.name === 'load_shared') return `tileA[gtid.x][gtid.y] = _v_input;`;
    if (n.name === 'commit_to_manifold' || n.name === 'advance_phase')
      return `// Phase advance (barrier)`;
    if (n.name.startsWith('load_from_manifold')) return `// Load from manifold M`;
    return `// Sek op: ${n.name}`;
  }

  _genLoop(n) {
    const body = n.body.map(c => this._genNode(c)).filter(Boolean)
      .map(l => '    ' + l).join('\n');
    return [`for (int _k_${n.name} = 0; ; _k_${n.name}++) {`, body,
            `    GroupMemoryBarrierWithGroupSync(); // Kumk'u pi-barrier`, `}`].join('\n');
  }

  _genDispatch(n) {
    return `// [Muwan ${n.name}] → cmdList->Dispatch(${n.x}, ${n.y}, ${n.z});`;
  }

  _genChain(n) {
    const load = `float _v_${n.load} = ${n.load}[dtid.x];`;
    const op   = n.op ? this._genNode(n.op) : '';
    return [load, op].filter(Boolean).join('\n');
  }
}

// ─── C++ host codegen ─────────────────────────────────────────────────────────

export class CppHostCodegen {
  generate(ast) {
    const kernels = ast.body.filter(n => n?.type === 'Kernel');
    const allocs  = ast.body.filter(n => n?.type === 'Alloc');
    const dispatches = ast.body.flatMap(n => n?.type === 'Kernel' ? (n.body ?? []).filter(b => b?.type === 'Dispatch') : []);

    const bufDecls = allocs.map(a =>
      `    ID3D12Resource* ${a.name};  // ${a.shape ?? 'buffer'}`
    ).join('\n');

    const dispatchCode = dispatches.map(d =>
      `        // [Muwan ${d.name}]\n        UINT tgX = (M + 15) / 16, tgY = (N + 15) / 16;\n        cmdList->Dispatch(tgX, tgY, 1);`
    ).join('\n');

    return `#include <d3d12.h>
#include <DirectXMath.h>

// K'uhul compiled host code
// Generated by kxc.js — K'uhul Compiler

class KXKernel {
private:
    ID3D12Device* device = nullptr;
    ID3D12GraphicsCommandList* cmdList = nullptr;
    ID3D12RootSignature* rootSig = nullptr;
    ID3D12PipelineState* pso = nullptr;
    // Tensor buffers (from SVG-3D K'uhul encoding)
${bufDecls || '    // (no buffers)'}

public:
    void Execute() {
        // pi-phase 0: Setup (Pop)
        cmdList->SetComputeRootSignature(rootSig);
        cmdList->SetPipelineState(pso);

        // pi-phase pi/4: Bind (Wo / Yax)
        // Bind SRV/UAV buffers here...

        // pi-phase pi/2: Dispatch (Muwan / K'ayab')
${dispatchCode || '        // (no dispatch)'}

        // pi-phase pi: Barrier (Kumk'u)
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
        cmdList->ResourceBarrier(1, &barrier);

        // pi-phase 2pi: Complete (Xul)
    }
};
`;
  }
}

// ─── KXC Compiler facade ──────────────────────────────────────────────────────

export class KXC {
  constructor() {
    this._hlsl = new HLSLCodegen();
    this._cpp  = new CppHostCodegen();
  }

  /** Compile K'uhul source → { hlsl, cpp, ast }. */
  compile(source) {
    const tokens = new KXCLexer(source).tokenize();
    const ast    = new KXCParser(tokens).parse();
    const hlsl   = this._hlsl.generate(ast);
    const cpp    = this._cpp.generate(ast);
    return { hlsl, cpp, ast };
  }

  /** Compile a .kslx shader source → HLSL only. */
  compileShader(kslx) { return this.compile(kslx).hlsl; }

  /** Compile a .kxx host source → C++ only. */
  compileHost(kxx) { return this.compile(kxx).cpp; }
}

// ─── File extension registry ──────────────────────────────────────────────────

export const KXC_EXTENSIONS = Object.freeze([
  { ext: '.kxx',   lang: 'kuhul-host',   compilesTo: '.cpp',        desc: 'K\'uhul host control code for D3D12 C++' },
  { ext: '.kslx',  lang: 'kuhul-shader', compilesTo: '.hlsl',       desc: 'K\'uhul geometric shaders for HLSL compute' },
  { ext: '.kuhul', lang: 'kuhul',        compilesTo: '.kxx + .kslx', desc: 'Combined K\'uhul source with embedded shaders' },
  { ext: '.kxml',  lang: 'kuhul-xml',    compilesTo: 'FLUX IR',      desc: 'K\'uhul KXML graph spec (µMODEL spec format)' },
]);
