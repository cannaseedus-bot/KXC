// Generated WGSL from KLSL manifest shader: softmax
struct Params {
  rows: u32,
  cols: u32,
};
@group(0) @binding(0) var<uniform> params: Params;

@group(0) @binding(1) var<storage, read> In: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>,
           @builtin(local_invocation_id) local_id: vec3<u32>,
           @builtin(workgroup_id) workgroup_id: vec3<u32>) {
let row = global_id.x;
if (row >= params.rows) { return; }
var mx: f32 = -100000000.0;
for (var j: u32 = 0u; j < params.cols; j = j + 1u) { if (In[row * params.cols + j] > mx) { mx = In[row * params.cols + j]; } }
var sm: f32 = 0.0;
for (var j: u32 = 0u; j < params.cols; j = j + 1u) { sm = sm + exp(In[row * params.cols + j] - mx); }
for (var j: u32 = 0u; j < params.cols; j = j + 1u) { Out[row * params.cols + j] = exp(In[row * params.cols + j] - mx) / sm; }
}
