// Generated HLSL from KLSL manifest shader: scale_bias
cbuffer Params : register(b0) {
  uint count;
  float scale;
  float bias;
};

StructuredBuffer<float> in_buf : register(t0);
RWStructuredBuffer<float> out_buf : register(u0);

[numthreads(64, 1, 1)]
void CSMain(uint3 global_id : SV_DispatchThreadID,
            uint3 local_id : SV_GroupThreadID,
            uint3 workgroup_id : SV_GroupID) {
uint i = global_id.x;
if (i >= count) { return; }
out_buf[i] = in_buf[i] * scale + bias;
}
