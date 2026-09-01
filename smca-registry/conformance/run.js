import fs from "fs";
import path from "path";
import crypto from "crypto";

const root = path.resolve(new URL("..", import.meta.url).pathname);
const vectorsPath = path.join(root, "conformance", "vectors", "collapse.geometry.conformance.json");
const wasmPath = path.join(root, "wasm", "cm1_verify.wasm");

const vectors = JSON.parse(fs.readFileSync(vectorsPath, "utf8"));
const wasmBuffer = fs.readFileSync(wasmPath);
const module = await WebAssembly.compile(wasmBuffer);
const instance = await WebAssembly.instantiate(module, {});

const { verify, memory, __heap_base } = instance.exports;
if (typeof verify !== "function" || !(memory instanceof WebAssembly.Memory)) {
  throw new Error("WASM verifier exports are missing.");
}

const memoryView = new Uint8Array(memory.buffer);

const hash = crypto
  .createHash("sha256")
  .update(JSON.stringify({ geometry: vectors.geometry, kernel: vectors.kernel, vectors: vectors.vectors }))
  .digest("hex");

if (hash !== vectors.conformance_hash) {
  throw new Error("Conformance hash mismatch.");
}

for (const entry of vectors.vectors) {
  const bytes = Buffer.from(entry.hex, "hex");
  const ptr = Number(__heap_base);
  memoryView.set(bytes, ptr);
  const result = verify(ptr, bytes.length);
  if ((result === 1) !== entry.valid) {
    throw new Error(`Vector ${entry.name} failed.`);
  }
}

console.log("SMCA conformance passed.");
