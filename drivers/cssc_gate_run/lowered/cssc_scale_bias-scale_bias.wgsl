// Generated WGSL from KLSL manifest shader: scale_bias
struct Params {
  count: u32,
  scale: f32,
  bias: f32,
};
@group(0) @binding(0) var<uniform> params: Params;

@group(0) @binding(1) var<storage, read> in_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>,
           @builtin(local_invocation_id) local_id: vec3<u32>,
           @builtin(workgroup_id) workgroup_id: vec3<u32>) {
let i = global_id.x;
if (i >= params.count) { return; }
out_buf[i] = in_buf[i] * params.scale + params.bias;
}
