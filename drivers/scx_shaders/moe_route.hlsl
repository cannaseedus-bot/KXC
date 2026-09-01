cbuffer Params : register(b0) {
  uint NUM_EXPERTS;
  uint TOPK;
};

StructuredBuffer<float> token : register(t0);
StructuredBuffer<float> router : register(t1);
RWStructuredBuffer<uint> out_idx : register(u0);

[numthreads(32,1,1)]
void main(uint tid : SV_DispatchThreadID) {
  float best1 = -1e9;
  float best2 = -1e9;
  uint id1 = 0, id2 = 0;

  // assumes token length=128, router laid out row-major
  for (uint e = 0; e < NUM_EXPERTS; ++e) {
    float s = 0;
    for (uint i = 0; i < 128; i++) {
      s += token[i] * router[e * 128 + i];
    }
    if (s > best1) {
      best2 = best1; id2 = id1;
      best1 = s; id1 = e;
    } else if (s > best2) {
      best2 = s; id2 = e;
    }
  }

  out_idx[tid * 2 + 0] = id1;
  out_idx[tid * 2 + 1] = id2;
}
