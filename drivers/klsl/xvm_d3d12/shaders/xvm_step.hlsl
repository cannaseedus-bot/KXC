// xvm_step.hlsl
// One deterministic instruction step per fiber thread.

struct Fiber {
  uint pc;
  uint sp;
  uint phase;
  uint flags;
};

RWStructuredBuffer<Fiber> Fibers : register(u0);
RWStructuredBuffer<uint> Shared : register(u1);
StructuredBuffer<uint> Code : register(t0);

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
  uint fid = id.x;
  Fiber f = Fibers[fid];

  if (f.flags == 0) {
    return;
  }

  uint instr = Code[f.pc++];
  uint op = instr & 0x3F;

  switch (op) {
    case 0x10: {
      uint index = Code[f.pc++];
      uint value = Code[f.pc++];
      InterlockedAdd(Shared[index], value);
      break;
    }
    case 0x05: {
      f.flags = 0;
      break;
    }
    default: {
      f.flags = 0;
      break;
    }
  }

  Fibers[fid] = f;
}
