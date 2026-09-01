// Generated HLSL from KLSL manifest shader: softmax
cbuffer Params : register(b0) {
  uint rows;
  uint cols;
};

StructuredBuffer<float> In : register(t0);
RWStructuredBuffer<float> Out : register(u0);

[numthreads(64, 1, 1)]
void CSMain(uint3 global_id : SV_DispatchThreadID,
            uint3 local_id : SV_GroupThreadID,
            uint3 workgroup_id : SV_GroupID) {
uint row = global_id.x;
if (row >= rows) { return; }
float mx = -100000000.0;
for (uint j = 0u; j < cols; j = j + 1u) { if (In[row * cols + j] > mx) { mx = In[row * cols + j]; } }
float sm = 0.0;
for (uint j = 0u; j < cols; j = j + 1u) { sm = sm + exp(In[row * cols + j] - mx); }
for (uint j = 0u; j < cols; j = j + 1u) { Out[row * cols + j] = exp(In[row * cols + j] - mx) / sm; }
}
