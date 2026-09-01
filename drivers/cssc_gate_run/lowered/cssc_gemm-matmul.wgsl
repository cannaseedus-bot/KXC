// Generated WGSL from KLSL manifest shader: matmul
struct Params {
  M: u32,
  N: u32,
  K: u32,
};
@group(0) @binding(0) var<uniform> params: Params;

@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>,
           @builtin(local_invocation_id) local_id: vec3<u32>,
           @builtin(workgroup_id) workgroup_id: vec3<u32>) {
let col = global_id.x;
let row = global_id.y;
if (row >= params.M) { return; }
if (col >= params.N) { return; }
var acc: f32 = 0.0;
for (var k: u32 = 0u; k < params.K; k = k + 1u) { acc = acc + A[row * params.K + k] * B[k * params.N + col]; }
C[row * params.N + col] = acc;
}
