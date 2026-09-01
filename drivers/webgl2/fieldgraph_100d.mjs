// WebGL2 projection for the KFG100D1 fieldgraph payload.
// Native/PyTorch memory is copied across the WebView2 boundary as bytes.

const HEADER_BYTES = 20;
const RECORD_BYTES = 208; // uint64 hash + 100 float16 values

function halfToFloat(bits) {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

export function decodeKfg100d(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  const magic = new TextDecoder().decode(new Uint8Array(arrayBuffer, 0, 8));
  if (magic !== 'KFG100D1') throw new Error(`Unsupported fieldgraph magic: ${magic}`);
  const version = view.getUint32(8, true);
  const count = view.getUint32(12, true);
  const dimensions = view.getUint32(16, true);
  if (version !== 1 || dimensions !== 100) throw new Error('Invalid KFG100D1 header');
  if (arrayBuffer.byteLength < HEADER_BYTES + count * RECORD_BYTES) throw new Error('Truncated fieldgraph');

  const vectors = new Float32Array(count * dimensions);
  const keys = new BigUint64Array(count);
  const half = new Uint16Array(arrayBuffer, HEADER_BYTES, count * (RECORD_BYTES / 2));
  for (let i = 0; i < count; i++) {
    const recordOffset = i * (RECORD_BYTES / 2);
    keys[i] = BigInt(view.getBigUint64(HEADER_BYTES + i * RECORD_BYTES, true));
    for (let j = 0; j < dimensions; j++) vectors[i * dimensions + j] = halfToFloat(half[recordOffset + 4 + j]);
  }
  return { version, count, dimensions, keys, vectors };
}

export function uploadFieldgraphTexture(gl, fieldgraph) {
  const texture = gl.createTexture();
  if (!texture) throw new Error('Unable to create fieldgraph texture');
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, fieldgraph.dimensions, fieldgraph.count,
    0, gl.RED, gl.FLOAT, fieldgraph.vectors);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return texture;
}

export function postFieldgraphToHost(webview, payload) {
  if (!webview?.postMessage) throw new Error('WebView2 bridge is unavailable');
  webview.postMessage({ type: 'ASX_FIELDGRAPH_100D', payload });
}
