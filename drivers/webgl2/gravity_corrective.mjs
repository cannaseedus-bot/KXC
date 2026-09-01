// WebGL2 adapter for the π-KUHUL gradient-gravity corrective.
// The WASM tree-sitter drivers remain parser backends; this module is the
// numeric WebGL2 pass that consumes their tensor/field outputs.

import vertexSource from './fullscreen.vert?raw';
import fragmentSource from './gravity_corrective.frag?raw';

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`gravity corrective shader compile failed: ${info}`);
  }
  return shader;
}

export function createGravityCorrectiveProgram(gl) {
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`gravity corrective program link failed: ${gl.getProgramInfoLog(program)}`);
  }
  return {
    program,
    weights: gl.getUniformLocation(program, 'u_weights'),
    gradients: gl.getUniformLocation(program, 'u_gradients'),
    scale: gl.getUniformLocation(program, 'u_scale'),
    clip: gl.getUniformLocation(program, 'u_clip'),
    gradientRms: gl.getUniformLocation(program, 'u_gradient_rms'),
  };
}

export function runGravityCorrectivePass(gl, pipeline, {
  weightsTexture,
  gradientsTexture,
  outputFramebuffer,
  width,
  height,
  scale = 0.05,
  clip = 0.25,
  gradientRms = 1.0,
}) {
  if (!pipeline?.program || !weightsTexture || !gradientsTexture || !outputFramebuffer) {
    throw new Error('gravity corrective requires program, weight, gradient, and output resources');
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, outputFramebuffer);
  gl.viewport(0, 0, width, height);
  gl.useProgram(pipeline.program);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, weightsTexture);
  gl.uniform1i(pipeline.weights, 0);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, gradientsTexture);
  gl.uniform1i(pipeline.gradients, 1);
  gl.uniform1f(pipeline.scale, scale);
  gl.uniform1f(pipeline.clip, clip);
  gl.uniform1f(pipeline.gradientRms, gradientRms);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}
