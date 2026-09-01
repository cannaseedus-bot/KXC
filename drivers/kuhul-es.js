#!/usr/bin/env node
'use strict';

/*
 * KUHUL-ES CLI
 * Version Integrity + Trace Artifact
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { spawnSync } = require('child_process');
const { program } = require('commander');

// -----------------------------------------------------------------------------
// Version (single source of truth)
// -----------------------------------------------------------------------------
const pkg = require('../package.json');

// -----------------------------------------------------------------------------
// Banner
// -----------------------------------------------------------------------------
console.log(`
╔══════════════════════════════════════╗
║        KUHUL-ES v${pkg.version}               ║
║  ECMAScript syntax, KUHUL semantics  ║
╚══════════════════════════════════════╝
`);

// -----------------------------------------------------------------------------
// CLI Definition
// -----------------------------------------------------------------------------
program
  .name('kuhul-es')
  .description('KUHUL-ES - The Trojan Horse for KUHUL semantics')
  .version(pkg.version);

// -----------------------------------------------------------------------------
// init
// -----------------------------------------------------------------------------
program
  .command('init [project-name]')
  .description('Initialize a new KUHUL-ES project')
  .action((projectName = 'my-kuhul-app') => {
    console.log(`Creating project: ${projectName}`);
    console.log('Use `kuhul-es new <name>` for full scaffold');
    process.exit(0);
  });

// -----------------------------------------------------------------------------
// new (FULL PROJECT SCAFFOLD)
// -----------------------------------------------------------------------------
program
  .command('new <name>')
  .description('Create a new KUHUL-ES project from template')
  .action((name) => {
    console.log(`Creating project: ${name}`);

    fs.mkdirSync(name, { recursive: true });

    const template = `
π config = {
  name: "${name}",
  version: "1.0.0",
  author: "${process.env.USERNAME || 'Developer'}"
};

τ frame = 0;

function* main() {
  yield* Sek('log', \`🚀 Starting \${config.name} v\${config.version}\`);

  @for (let i = 0; i < 5; i++) {
    yield* Sek('log', \`Frame: \${frame}\`);
    frame += 1;
    yield* Sek('wait', 100);
  }

  yield* Sek('log', '✅ Project ready!');
}

main();
`.trim();

    fs.writeFileSync(path.join(name, 'main.kuhules'), template);

    const packageJson = {
      name,
      version: '1.0.0',
      private: true,
      scripts: {
        start: 'kuhul-es run main.kuhules',
        dev: 'kuhul-es run main.kuhules --record'
      },
      dependencies: {
        'kuhul-es': `^${pkg.version}`
      }
    };

    fs.writeFileSync(
      path.join(name, 'package.json'),
      JSON.stringify(packageJson, null, 2)
    );

    console.log(`✓ Project created: ${name}`);
    console.log(`  cd ${name}`);
    console.log(`  npm install`);
    console.log(`  npm start`);
    process.exit(0);
  });

// -----------------------------------------------------------------------------
// compile — KUHUL-ES source -> canonical KAST (.kson, protocol kast/1)
program
  .command('compile <input>')
  .description('Compile KUHUL-ES source to canonical KAST (.kson, protocol kast/1)')
  .option('-o, --output <file>', 'Output .kson file')
  .option('--driver', 'Emit with a @driver contract (provider binding)')
  .option('--driver-only', 'Emit driver-only KAST (secure admission surface: capabilities + phase hooks, application body stripped)')
  .option('--provider <id>', 'Provider id for the @driver contract')
  .action((input, options) => {
    const parserPath = path.join(__dirname, '..', 'compiler', 'src', 'parser.js');
    const { KUHULParser, toKast } = require(parserPath);
    const abs = path.resolve(process.cwd(), input);
    const source = fs.readFileSync(abs, 'utf8');
    const prog = new KUHULParser(source, path.basename(input)).parse();
    const kast = toKast(prog, path.basename(input), {
      driver: !!options.driver,
      driverOnly: !!options.driverOnly,
      provider: options.provider || undefined,
    });
    const out = options.output || path.join(path.dirname(abs), path.basename(input, path.extname(input)) + '.kson');
    fs.writeFileSync(out, JSON.stringify(kast, null, 2));
    console.log('compiled: ' + input + ' -> ' + out);
    const nodeCount = Array.isArray(kast.nodes) ? kast.nodes.length : 0;
    const edgeCount = Array.isArray(kast.edges) ? kast.edges.length : 0;
    console.log('  protocol=' + kast.protocol + '  nodes=' + nodeCount +
      '  edges=' + edgeCount + '  @driver=' + ('@driver' in kast));
    process.exit(0);
  });

// run
// -----------------------------------------------------------------------------
program
  .command('run <file>')
  .description('Run a KUHUL-ES program through the deterministic runtime + physics engine')
  .option('--record', 'Record deterministic execution trace to trace.json')
  .option('--replay <trace>', 'Replay from a recorded trace file')
  .option('--physics-out <file>', 'Write physics history JSON after run')
  .option('--thoughts-out <file>', 'Write thinking-engine trace JSON after run')
  .action(async (file, options) => {
    console.log(`▶ Running: ${file}`);

    const absFile = path.resolve(process.cwd(), file);
    const source = fs.readFileSync(absFile, 'utf8');

    const { KUHULRuntimeNode } = require(path.join(__dirname, '..', 'runtime', 'src', 'node.js'));
    const rt = new KUHULRuntimeNode();

    if (options.record) {
      rt.on('hash', ({ frame, hash }) => {
        // nothing extra; hash chain is already deterministic
      });
    }

    await rt.execute(source);

    const trace = {
      version: pkg.version,
      file,
      argv: process.argv.slice(2),
      cwd: process.cwd(),
      timestamp: new Date().toISOString(),
      frames: rt.frame,
      hashChain: rt.hashChain,
      physicsHistoryLength: rt.physics.history.length,
      thoughtCount: rt.thinker.thoughts.length,
    };
    trace.hash = crypto.createHash('sha256').update(JSON.stringify(trace)).digest('hex');

    if (options.record) {
      const outPath = path.resolve(process.cwd(), 'trace.json');
      fs.writeFileSync(outPath, JSON.stringify(trace, null, 2));
      console.log('trace.json written');
      console.log(`trace hash: ${trace.hash}`);
    }

    if (options.replay) {
      const replayPath = path.resolve(process.cwd(), options.replay);
      const oldTrace = JSON.parse(fs.readFileSync(replayPath, 'utf8'));
      const expectedHash = crypto.createHash('sha256')
        .update(JSON.stringify({ ...oldTrace, hash: undefined })).digest('hex');
      if (oldTrace.hash !== expectedHash) {
        console.error('Trace hash mismatch');
        process.exit(1);
      }
      console.log('Trace verified');
      console.log(`trace hash: ${oldTrace.hash}`);
    }

    if (options.physicsOut) {
      fs.writeFileSync(path.resolve(process.cwd(), options.physicsOut), JSON.stringify(rt.physics.history, null, 2));
      console.log(`physics history written: ${options.physicsOut}`);
    }

    if (options.thoughtsOut) {
      fs.writeFileSync(path.resolve(process.cwd(), options.thoughtsOut), JSON.stringify(rt.thinker.thoughts, null, 2));
      console.log(`thoughts trace written: ${options.thoughtsOut}`);
    }

    console.log(`✓ Execution complete: ${rt.frame} frames, ${rt.hashChain.length} hashes, ${rt.thinker.thoughts.length} thoughts`);
    process.exit(0);
  });

// -----------------------------------------------------------------------------
// train — GLSL trainer (semantic skeleton + physics + GLSL kernels)
program
  .command('train <config>')
  .description("Train a semantic-skeleton net with the K'UHUL physics engine (GLSL kernels probed)")
  .option('--glsl-endpoint <url>', 'json_runtime glsl_gpu sidecar endpoint', 'http://127.0.0.1:8787')
  .option('--backend <name>', 'backend profile: glsl, powernaut-glsl, xvm-d3d12, hybrid-cluster-glsl, cpu', 'glsl')
  .option('--backend-endpoint <url>', 'backend-specific endpoint / directory')
  .option('--backend-manifest <path>', 'Powernaut GLSL server.glsl.json manifest path')
  .option('--out <file>', 'Write the trained skeleton to a .json file')
  .option('--semantic', 'Enable semantic trainer advisor (reasons over folds/nodes/metrics)')
  .option('--semantic-interval <n>', 'Run semantic analysis every n epochs', '1')
  .action(async (configFile, options) => {
    const { GLSLTrainer, glslHttpTransport, powernautGlslTransport, xvmD3d12Transport, hybridClusterGlslTransport } = require(path.join(__dirname, '..', 'runtime', 'src', 'trainer.js'));
    const { SemanticTrainer } = require(path.join(__dirname, '..', 'runtime', 'src', 'trainer_semantic.js'));
    let cfg;
    try {
      cfg = JSON.parse(fs.readFileSync(configFile, 'utf8'));
    } catch (e) {
      console.error('config error:', e.message);
      process.exit(1);
    }

    // Resolve backend transport.
    let transport = null;
    if (options.backend === 'powernaut-glsl') {
      const endpoint = options.backendEndpoint || 'http://127.0.0.1:9060';
      const manifest = options.backendManifest || path.join(__dirname, 'server.glsl.json');
      transport = powernautGlslTransport(endpoint, { manifest, timeoutMs: 5000 });
    } else if (options.backend === 'xvm-d3d12') {
      transport = xvmD3d12Transport({ xvmDir: options.backendEndpoint });
    } else if (options.backend === 'hybrid-cluster-glsl') {
      const manifest = options.backendManifest || path.join(__dirname, 'server.glsl.json');
      transport = hybridClusterGlslTransport({
        xvmDir: options.backendEndpoint,
        glslEndpoint: options.glslEndpoint || 'http://127.0.0.1:9060',
        manifest,
        timeoutMs: 5000,
      });
    } else if (options.backend === 'glsl') {
      transport = glslHttpTransport(options.glslEndpoint);
    }

    const trainerOpts = {
      inputDim: cfg.inputDim ?? 2,
      hiddenDim: cfg.hiddenDim ?? 40,
      outputDim: cfg.outputDim ?? 1,
      lr: cfg.lr ?? 0.2,
      steps: cfg.steps ?? 500,
      momentum: cfg.momentum ?? 0.9,
      transport,
      verbose: true,
    };
    if (cfg.vocabSize) {
      trainerOpts.vocabSize = cfg.vocabSize;
      trainerOpts.embedDim = cfg.embedDim ?? 64;
    }
    if (options.semantic) {
      trainerOpts.semanticTrainer = new SemanticTrainer({ interval: parseInt(options.semanticInterval, 10) || 1 });
    }

    let dataset = [];
    let datasetStats = null;
    const datasetKind = String(cfg.dataset || '').toLowerCase();
    const textFile = cfg.textFile || cfg.datasetFile || null;
    const chatJsonlFile = cfg.chatFile || cfg.datasetPath || textFile || null;
    const useChatJsonl =
      ['chat', 'chat_jsonl', 'jsonl', 'jsonl_chat', 'ultrachat_jsonl'].includes(datasetKind)
      || (datasetKind === 'text' && typeof textFile === 'string' && textFile.toLowerCase().endsWith('.jsonl'));

    // Support plain text dataset tokenization
    if (!useChatJsonl && datasetKind === 'text' && textFile) {
      const { loadTokenizer } = require(path.join(__dirname, '..', 'runtime', 'src', 'tokenizer.js'));
      const { buildTokenDataset } = require(path.join(__dirname, '..', 'runtime', 'src', 'text_dataset.js'));
      const tok = await loadTokenizer({ path: cfg.tokenizer || 'tokenizer.json' });
      const built = await buildTokenDataset({
        file: textFile,
        tokenizer: tok,
        seqLen: cfg.seqLen || 1,
        maxSamples: cfg.maxSamples || 10000
      });
      dataset = built.dataset;
      datasetStats = built.stats || null;
      // switch to token-mode skeleton
      trainerOpts.vocabSize = built.vocabSize;
      trainerOpts.embedDim = cfg.embedDim ?? 64;
      delete trainerOpts.inputDim;
      delete trainerOpts.outputDim;
    } else if (useChatJsonl && chatJsonlFile) {
      const { loadTokenizer } = require(path.join(__dirname, '..', 'runtime', 'src', 'tokenizer.js'));
      const { buildChatJsonlTokenDataset } = require(path.join(__dirname, '..', 'runtime', 'src', 'text_dataset.js'));
      const tok = await loadTokenizer({ path: cfg.tokenizer || 'tokenizer.json' });
      const built = await buildChatJsonlTokenDataset({
        file: chatJsonlFile,
        tokenizer: tok,
        seqLen: cfg.seqLen || 1,
        maxSamples: cfg.maxSamples || 10000,
        maxRecords: cfg.maxRecords || Infinity,
        includeRoles: cfg.includeRoles !== false,
        minChars: cfg.minChars || 1
      });
      dataset = built.dataset;
      datasetStats = built.stats || null;
      trainerOpts.vocabSize = built.vocabSize;
      trainerOpts.embedDim = cfg.embedDim ?? 64;
      delete trainerOpts.inputDim;
      delete trainerOpts.outputDim;
    } else {
      dataset = (cfg.dataset === 'sin')
        ? Array.from({ length: cfg.samples ?? 80 }, (_, i) => {
            const x = (i / (cfg.samples ?? 80)) * 2 - 1;
            return { x: [x, x * x], y: [Math.sin(2 * Math.PI * x) + (cfg.offset ?? 0.5)] };
          })
        : (cfg.data || []).map(d => ({ x: d.x, y: d.y }));
    }

    const trainer = new GLSLTrainer(trainerOpts);

    console.log('');
    console.log('GLSL trainer: ' + configFile + '  (' + dataset.length + ' samples, ' + trainer.hiddenDim + ' hidden, backend=' + (options.backend || 'cpu') + ')');
    if (datasetStats) {
      console.log('dataset stats: ' + datasetStats.usedRecords + '/' + datasetStats.records + ' records used, ' +
        datasetStats.totalTokens + ' tokens, ' + datasetStats.skippedRecords + ' skipped');
    }
    const r = await trainer.train(dataset);
    if (options.out) {
      const skeleton = trainer.model();
      skeleton.config = {
        source: configFile,
        hiddenDim: cfg.hiddenDim ?? 40,
        ...(trainer.vocabSize ? { vocabSize: trainer.vocabSize, embedDim: trainer.embedDim } : { inputDim: cfg.inputDim ?? 2, outputDim: cfg.outputDim ?? 1 }),
      };
      fs.writeFileSync(options.out, JSON.stringify(skeleton, null, 2));
      console.log('model skeleton written: ' + options.out + ' (' + skeleton.nodes.length + ' nodes, ' + skeleton.edges.length + ' edges)');
    }
    console.log('final loss: ' + r.finalLoss.toFixed(6) + ' | GLSL: ' + (r.gpu ? 'ACTIVE' : 'CPU fallback') +
      (options.semantic ? ' | semantic advice epochs: ' + r.semanticAdvice.length : ''));
  });

// -----------------------------------------------------------------------------
// train-native — bridge gpt2_trainer.exe with KUHUL physics gating
// -----------------------------------------------------------------------------
program
  .command('train-native')
  .description('Launch native gpt2_trainer.exe with KUHUL physics-derived optimizer controls')
  .option('--trainer-exe <path>', 'Path to gpt2_trainer.exe')
  .option('--model <path>', 'Model preset or safetensors path')
  .option('--data <path>', 'Packed token bin path')
  .option('--out <path>', 'Output safetensors path')
  .option('--steps <n>', 'Training steps', '1000')
  .option('--batch <n>', 'Batch size', '4')
  .option('--block <n>', 'Block/sequence length', '128')
  .option('--lr <v>', 'Base learning rate before physics scaling', '3e-5')
  .option('--save-every <n>', 'Checkpoint interval', '200')
  .option('--warmup <n>', 'Optional GPT2_WARMUP override')
  .option('--cwd <dir>', 'Working directory to run trainer from (shader path sensitive)')
  .option('--dry-run', 'Print resolved launch command/env and exit')
  .action((options) => {
    const { KuhulPhysics } = require(path.join(__dirname, '..', 'runtime', 'src', 'physics.js'));
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

    const parseIntOr = (v, dflt) => {
      const n = parseInt(v, 10);
      return Number.isFinite(n) && n > 0 ? n : dflt;
    };
    const parseFloatOr = (v, dflt) => {
      const n = parseFloat(v);
      return Number.isFinite(n) ? n : dflt;
    };
    const firstExisting = (candidates) => candidates.find((p) => !!p && fs.existsSync(p));
    const resolveRunCwd = (trainerExe, explicitCwd) => {
      if (explicitCwd) return path.resolve(process.cwd(), explicitCwd);
      const exeDir = path.dirname(trainerExe);
      const hereShader = path.join(exeDir, 'gpt2_adam.hlsl');
      const childShader = path.join(exeDir, 'shaders', 'gpt2_adam.hlsl');
      const parentShader = path.join(exeDir, '..', 'shaders', 'gpt2_adam.hlsl');
      if (fs.existsSync(hereShader)) return exeDir;
      if (fs.existsSync(parentShader)) return exeDir; // exe expects ../shaders from cwd
      if (fs.existsSync(childShader)) return path.join(exeDir, 'shaders');
      return exeDir;
    };

    const defaultTrainerCandidates = [
      options.trainerExe,
      process.env.KUHUL_TRAINER_EXE,
      path.resolve(__dirname, '..', '..', 'trainer', 'build', 'Release', 'gpt2_trainer.exe'),
    ].filter(Boolean);

    const trainerExe = firstExisting(defaultTrainerCandidates) || defaultTrainerCandidates[0];
    if (!trainerExe) {
      console.error('trainer error: no trainer executable candidate resolved');
      process.exit(1);
    }

    if (!options.dryRun && !fs.existsSync(trainerExe)) {
      console.error('trainer error: executable not found: ' + trainerExe);
      process.exit(1);
    }

    const runCwd = resolveRunCwd(trainerExe, options.cwd);
    if (!options.dryRun && !fs.existsSync(runCwd)) {
      console.error('trainer error: run cwd not found: ' + runCwd);
      process.exit(1);
    }

    // KUHUL physics cycle -> optimizer controls
    const phys = new KuhulPhysics();
    phys.perceive(0.02);
    phys.represent(0.03);
    phys.plan(0.03);
    phys.execute(0.02);
    phys.project();
    phys.consolidate(0.04);

    const gate = phys.computeGravityGate();
    const clip = clamp(gate, 0.5, 2.0);
    const lrScale = clamp(0.7 + 0.3 * gate, 0.7, 1.3);
    const baseLr = parseFloatOr(options.lr, 3e-5);
    const lr = baseLr * lrScale;

    const args = [];
    if (options.model) args.push('--model', options.model);
    if (options.data) args.push('--data', options.data);
    if (options.out) args.push('--out', options.out);
    args.push('--steps', String(parseIntOr(options.steps, 1000)));
    args.push('--batch', String(parseIntOr(options.batch, 4)));
    args.push('--block', String(parseIntOr(options.block, 128)));
    args.push('--lr', String(lr));
    args.push('--save-every', String(parseIntOr(options.saveEvery, 200)));

    const env = {
      ...process.env,
      GPT2_ADAPTIVE_CLIP: '1',
      GPT2_CLIP: clip.toFixed(6),
      KUHUL_PHYSICS_GATE: gate.toFixed(6),
      KUHUL_PHYSICS_GRAVITY: phys.gravity.toFixed(6),
      KUHUL_PHYSICS_ATTENTION: phys.attention.toFixed(6),
      KUHUL_PHYSICS_ENTROPY: phys.entropy.toFixed(6),
      KUHUL_PHYSICS_PRESSURE: phys.pressure.toFixed(6),
    };
    if (options.warmup) env.GPT2_WARMUP = String(parseIntOr(options.warmup, 0));

    console.log('KUHUL Native Trainer Bridge');
    console.log('  trainer:', trainerExe);
    console.log('  cwd:', runCwd);
    console.log(`  physics: gate=${gate.toFixed(6)} gravity=${phys.gravity.toFixed(6)} attention=${phys.attention.toFixed(6)} entropy=${phys.entropy.toFixed(6)}`);
    console.log(`  env: GPT2_ADAPTIVE_CLIP=1 GPT2_CLIP=${clip.toFixed(6)}${options.warmup ? ` GPT2_WARMUP=${env.GPT2_WARMUP}` : ''}`);
    console.log('  cmd:', `${trainerExe} ${args.join(' ')}`);

    if (options.dryRun) {
      process.exit(0);
    }

    const res = spawnSync(trainerExe, args, {
      cwd: runCwd,
      env,
      stdio: 'inherit',
      shell: false,
    });

    if (res.error) {
      console.error('trainer launch error:', res.error.message);
      process.exit(1);
    }
    process.exit(typeof res.status === 'number' ? res.status : 1);
  });

// ---------------------------------------------------------------------------
// gpu — probe GPU / GLSL backends
// ---------------------------------------------------------------------------
program
  .command('gpu')
  .description('Probe available GLSL compute backends and kernel sources')
  .option('--glsl-endpoint <url>', 'json_runtime glsl_gpu sidecar endpoint', 'http://127.0.0.1:8787')
  .option('--backend <name>', 'backend profile: glsl, powernaut-glsl, xvm-d3d12, hybrid-cluster-glsl, cpu', 'glsl')
  .option('--backend-endpoint <url>', 'backend-specific endpoint / directory')
  .option('--backend-manifest <path>', 'Powernaut GLSL server.glsl.json manifest path')
  .option('--kernel-out <dir>', 'Write GLSL kernel source files to a directory')
  .action(async (options) => {
    const { GLSLTrainer, glslHttpTransport, powernautGlslTransport, xvmD3d12Transport, hybridClusterGlslTransport, DEFAULT_XVM_DIR } = require(path.join(__dirname, '..', 'runtime', 'src', 'trainer.js'));
    const { MATMUL, GELU, LAYERNORM } = require(path.join(__dirname, '..', 'runtime', 'src', 'glsl_kernels.js'));

    console.log('KUHUL GPU Probe\n');

    // Resolve backend transport.
    let transport = null;
    let backendName = options.backend || 'glsl';
    if (backendName === 'powernaut-glsl') {
      const endpoint = options.backendEndpoint || 'http://127.0.0.1:9060';
      const manifest = options.backendManifest || path.join(__dirname, 'server.glsl.json');
      transport = powernautGlslTransport(endpoint, { manifest, timeoutMs: 5000 });
    } else if (backendName === 'xvm-d3d12') {
      transport = xvmD3d12Transport({ xvmDir: options.backendEndpoint });
    } else if (backendName === 'hybrid-cluster-glsl') {
      const manifest = options.backendManifest || path.join(__dirname, 'server.glsl.json');
      transport = hybridClusterGlslTransport({
        xvmDir: options.backendEndpoint,
        glslEndpoint: options.glslEndpoint || 'http://127.0.0.1:9060',
        manifest,
        timeoutMs: 5000,
      });
    } else if (backendName === 'glsl') {
      transport = glslHttpTransport(options.glslEndpoint);
    }

    const kernels = [
      { name: 'matmul', source: MATMUL },
      { name: 'gelu', source: GELU },
      { name: 'layernorm', source: LAYERNORM },
    ];

    for (const k of kernels) {
      console.log(`Kernel ${k.name}: ${k.source.length} chars`);
    }

    if (options.kernelOut) {
      fs.mkdirSync(options.kernelOut, { recursive: true });
      for (const k of kernels) {
        fs.writeFileSync(path.join(options.kernelOut, `${k.name}.glsl`), k.source);
      }
      console.log(`\nKernel sources written to: ${options.kernelOut}`);
    }

    // Try a real remote compile via the transport
    const trainer = new GLSLTrainer({
      inputDim: 2, hiddenDim: 8, outputDim: 1,
      transport,
      verbose: false,
    });
    const probe = await trainer.probeGLSL();

    console.log('\nRemote probe:');
    console.log('  backend:', backendName);
    if (backendName === 'powernaut-glsl') {
      console.log('  endpoint:', options.backendEndpoint || 'http://127.0.0.1:9060');
      console.log('  bundled server binary:', path.join(__dirname, 'GLSL_Server.exe'));
    } else if (backendName === 'hybrid-cluster-glsl') {
      console.log('  xvm dir:', options.backendEndpoint || process.env.XVM_D3D12_DIR || DEFAULT_XVM_DIR);
      console.log('  glsl endpoint:', options.glslEndpoint || 'http://127.0.0.1:9060');
      console.log('  bundled server binary:', path.join(__dirname, 'GLSL_Server.exe'));
    } else if (backendName === 'xvm-d3d12') {
      console.log('  xvm dir:', options.backendEndpoint || process.env.XVM_D3D12_DIR || DEFAULT_XVM_DIR);
    } else {
      console.log('  endpoint:', options.glslEndpoint);
    }
    console.log('  gpu:', probe.gpu ? 'ACTIVE' : 'UNAVAILABLE');
    if (probe.kernels) {
      for (const k of probe.kernels) {
        const ms = typeof k.elapsed_ms === 'number' ? ` (${k.elapsed_ms} ms)` : '';
        const backend = k.backend ? ` via ${k.backend}` : '';
        console.log(`  ${k.glyph}: ${k.compiled ? 'compiled' : 'failed'}${backend}${ms}`);
      }
    }
    if (probe.reason) console.log('  reason:', probe.reason);

    process.exit(0);
  });

// -----------------------------------------------------------------------------
// kxml — KHANARY KXML model runner
// -----------------------------------------------------------------------------
program
  .command('kxml')
  .description('Run KHANARY KXML model inference + chat templates')
  .option('--stb <path>', 'STB weights file')
  .option('--manifest <path>', 'Model manifest JSON (khanary-model-stb/v1)')
  .option('--run', 'Run a single forward pass and print output shape')
  .option('--generate <n>', 'Greedy generate n tokens from BOS', '0')
  .option('--chat', 'Render a sample KXML dialogue via Jinja chat template')
  .option('--prompt <text>', 'User prompt for --chat')
  .option('--kast-out <path>', 'Write semantic KAST trace of the forward graph')
  .option('--template-out <path>', 'Write KXML chat_template.json + .jinja')
  .option('--verify-hf', 'Verify HuggingFace model artifacts and optionally launch helper services (micronauts, llama-py-server.bat, AtomicDAG.bat)')
  .action(async (options) => {
    const { KxmlModel, emitTemplate } = require(path.join(__dirname, '..', 'runtime', 'src', 'kxml_driver.js'));
    const { toJinja, renderForGguf } = require(path.join(__dirname, '..', 'runtime', 'src', 'kxml_chat.js'));

    // Emit template artifacts
    if (options.templateOut) {
      fs.mkdirSync(options.templateOut, { recursive: true });
      fs.writeFileSync(path.join(options.templateOut, 'kxml_chat_template.json'),
        JSON.stringify(emitTemplate(options.templateOut), null, 2));
      fs.writeFileSync(path.join(options.templateOut, 'kxml_chat_template.jinja'), toJinja() + '\n');
      console.log(`KXML chat template written to ${options.templateOut}`);
      if (!options.stb) process.exit(0);
    }

    if (!options.stb || !options.manifest) {
      console.error('Usage: kuhul-es kxml --stb <weights.stb> --manifest <manifest.json> [--run | --generate N | --chat]');
      process.exit(1);
    }

    // Optional verification step that can spawn local helper services or CLI tools
    if (options.verifyHf) {
      console.log('[kxml] --verify-hf enabled: attempting environment verification and helper startup');

      const runSync = (cmd, args = [], opts = {}) => {
        try {
          console.log(`[kxml] running: ${cmd} ${Array.isArray(args) ? args.join(' ') : args}`);
          const res = spawnSync(cmd, Array.isArray(args) ? args : [args], Object.assign({ stdio: 'inherit', shell: true, timeout: opts.timeout || 120000 }, opts));
          if (res.error) {
            console.error(`[kxml] spawn error: ${res.error.message}`);
            return false;
          }
          if (res.status !== 0) {
            console.error(`[kxml] exited with status: ${res.status}`);
            return false;
          }
          return true;
        } catch (e) {
          console.error('[kxml] runSync exception:', e.message);
          return false;
        }
      };

      const findCandidate = (candidates) => {
        for (const c of candidates) {
          try {
            const p1 = path.resolve(process.cwd(), c);
            if (fs.existsSync(p1)) return p1;
            const p2 = path.resolve(__dirname, '..', c);
            if (fs.existsSync(p2)) return p2;
          } catch (e) { /* ignore */ }
        }
        // fallback to bare command name (may be on PATH)
        return candidates[0];
      };

      // 1) micronauts (CLI / runtime helper)
      const micronautsCmd = findCandidate(['micronauts', 'micronauts.bat', './micronauts.bat']);
      const micronautsOk = runSync(micronautsCmd, ['--version'], { timeout: 30000 });
      if (!micronautsOk) console.warn('[kxml] micronauts not available or returned non-zero');

      // 2) llama-py server (typical Windows helper)
      const llamaCmd = findCandidate(['llama-py-server.bat', './llama-py-server.bat', 'llama-py-server']);
      const llamaOk = runSync(llamaCmd, [], { timeout: 120000 });
      if (!llamaOk) console.warn('[kxml] llama-py-server not started or missing');

      // 3) AtomicDAG helper
      const dagCmd = findCandidate(['AtomicDAG.bat', './AtomicDAG.bat', 'AtomicDAG']);
      const dagOk = runSync(dagCmd, [], { timeout: 60000 });
      if (!dagOk) console.warn('[kxml] AtomicDAG helper not started or missing');

      console.log('[kxml] verification phase complete (warnings may indicate missing helpers)');
    }

    const model = await KxmlModel.fromFiles(options.stb, options.manifest);
    console.log(`[kxml] ${model.cfg.arch || 'model'}  ${model.cfg.n_layer || '?'}L  n_embd=${model.cfg.n_embd}  vocab=${model.cfg.vocab}`);
    console.log(`[kxml] graph: ${model.graph.length} glyphs  weights: ${Object.keys(model.W).length} tensors`);

    if (options.chat) {
      const chatTemplate = toJinja();
      const messages = [
        { role: 'system', content: 'You are a concise KUHUL assistant.' },
        { role: 'user', content: options.prompt || 'hello' },
      ];
      const prompt = renderForGguf(messages, chatTemplate, { addGenerationPrompt: true });
      console.log('\n--- KXML chat prompt ---\n');
      console.log(prompt);
      console.log('\n--- semantic KAST trace ---');
      model.forward([model.cfg.bos_token || 50256]);
      console.log(JSON.stringify(model.toKast().nodes.map(n => `${n.id}: ${n.glyph} -> fold ${n.fold}`), null, 2));
    }

    if (options.run) {
      const bos = model.cfg.bos_token || 50256;
      const out = model.forward([bos]);
      console.log(`[kxml] forward output shape: [${out.shape.join(', ')}]`);
    }

    const genCount = parseInt(options.generate, 10);
    if (genCount > 0) {
      const bos = model.cfg.bos_token || 50256;
      const tokens = model.generate([bos], genCount, true);
      console.log('[kxml] generated tokens:', tokens.join(' '));
    }

    if (options.kastOut) {
      fs.writeFileSync(options.kastOut, JSON.stringify(model.toKast(), null, 2));
      console.log(`[kxml] KAST trace written: ${options.kastOut}`);
    }

    process.exit(0);
  });

// -----------------------------------------------------------------------------
// doctor
// -----------------------------------------------------------------------------
program
  .command('doctor')
  .description('Run environment diagnostics for KUHUL-ES')
  .action(() => {
    console.log('KUHUL-ES Doctor\n');

    console.log('Version:', pkg.version);
    console.log('Node:', process.version);
    console.log('Platform:', process.platform);
    console.log('CLI path:', __filename);
    console.log('CWD:', process.cwd());

    try {
      require.resolve('commander');
      console.log('Commander: OK');
    } catch {
      console.log('Commander: MISSING');
      process.exit(1);
    }

    console.log('\nDiagnostics complete');
    process.exit(0);
  });

// -----------------------------------------------------------------------------
// Parse
// -----------------------------------------------------------------------------
program.parse(process.argv);

if (!process.argv.slice(2).length) {
  program.outputHelp();
}
