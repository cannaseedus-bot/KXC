StructuredBuffer<uint> A : register(t0); // packed int4, two per byte
StructuredBuffer<uint> B : register(t1); // packed int4
RWStructuredBuffer<int> C : register(u0); // accum in int32

[numthreads(16,16,1)]
void main(uint3 id : SV_DispatchThreadID) {
  uint row = id.x;
  uint col = id.y;
  // placeholder: no real unpack implemented
  C[row * 16 + col] = 0;
}
