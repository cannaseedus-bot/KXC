// KXC IR format — kernel descriptor builder/validator (v1.0.0-PowerShell-LLM)
//
// KXC (KUHUL eXecution Compiler) produces JSON IR describing GPU kernels with
// SMCA compliance metadata and capability flags used by the sxme_host dispatch layer.
//
// Stack ID: "asx-xcfe-stack/v1"
// Stack CID uses FNV1A-64 over the artifact list.

import { SMCA_LAYERS, SMCA_AUTHORITY_GRADIENT, KXC_FORBID_LIST } from '../smca/index.js';

export const KXC_STACK_ID = 'asx-xcfe-stack/v1';
export const KXC_IR_VERSION = '0.1.0';

export const KXC_CAPABILITY_FLAGS = Object.freeze({
  needsDecompress: 'needsDecompress', // kernel requires INT4 dequant before compute
  needsSoftmax:    'needsSoftmax',    // kernel includes a softmax reduction step
  needsMatMul:     'needsMatMul',     // kernel performs matrix multiplication
  kvInt4:          'kvInt4',          // KV cache stored in INT4 (not FP16)
  isMoEKernel:     'isMoEKernel',     // kernel is a Mixture-of-Experts dispatch kernel
});

export const KXC_ARTIFACT_EXTENSIONS = Object.freeze(['.cpp', '.hlsl', '.wgsl', '.cpu.cpp', '.smca.json']);

function fnv1a64(str) {
  // FNV1A-64 over UTF-8 bytes — returns hex string (lower 32 bits for brevity in JS)
  let h = 0xcbf29ce4; // FNV offset basis low-32
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h  = Math.imul(h, 0x01000193) >>> 0; // FNV prime low-32
  }
  return h.toString(16).padStart(8, '0');
}

export function computeStackCid(artifacts) {
  const sorted = [...artifacts].sort();
  return fnv1a64(sorted.join('|'));
}

export function createKernelIR(opts) {
  const {
    name,
    kernelClass,
    collapseClass,
    capabilities = [],
    resources    = [],
    tensors      = [],
    artifacts    = [],
    lawful       = true,
    registryMatched = false,
    requires     = [],
    forbids      = [],
  } = opts;

  if (!name)         throw new Error('KXC: name is required');
  if (!kernelClass)  throw new Error('KXC: kernelClass is required');
  if (!collapseClass) throw new Error('KXC: collapseClass is required');

  const unknownCaps = capabilities.filter(c => !Object.values(KXC_CAPABILITY_FLAGS).includes(c));
  if (unknownCaps.length) throw new Error(`KXC: unknown capability flags: ${unknownCaps.join(', ')}`);

  return {
    ir_version:  KXC_IR_VERSION,
    stack_id:    KXC_STACK_ID,
    stack_cid:   computeStackCid(artifacts),
    name,
    smca: {
      kernelClass,
      collapseClass,
      lawful,
      registryMatched,
      requires,
      forbids,
    },
    capabilities,
    resources,
    tensors,
    artifacts,
  };
}

export function validateKernelIR(ir) {
  const errors = [];

  if (ir.ir_version !== KXC_IR_VERSION) errors.push(`ir_version must be "${KXC_IR_VERSION}"`);
  if (!ir.name)    errors.push('name is required');
  if (!ir.smca)    errors.push('smca block is required');

  if (ir.smca) {
    if (!SMCA_LAYERS.includes(ir.smca.kernelClass)) {
      errors.push(`smca.kernelClass "${ir.smca.kernelClass}" is not a valid SMCA layer`);
    }
    const authorityLayers = SMCA_AUTHORITY_GRADIENT.order;
    const layerIdx = authorityLayers.indexOf(ir.smca.kernelClass);
    if (layerIdx >= 0 && layerIdx < 3 && ir.capabilities.includes('needsMatMul')) {
      // MatMul kernels should not operate above SCXQ2 (layer 3) — flag as warning only
      errors.push(`warning: needsMatMul kernel in layer "${ir.smca.kernelClass}" (above SCXQ2) — verify authority`);
    }
    const illegalForbids = (ir.smca.forbids || []).filter(f => !KXC_FORBID_LIST.includes(f));
    if (illegalForbids.length) {
      errors.push(`unknown forbid entries: ${illegalForbids.join(', ')}`);
    }
  }

  const ok = errors.filter(e => !e.startsWith('warning:')).length === 0;
  return { ok, errors };
}
