#pragma once
#include <cstdint>

/**
 * klsl_opcodes.h
 * KLSL XVM instruction set — extends xvm_core.h for shader-style compute.
 *
 * Encoding: 4 bytes per instruction
 *   [op:1][a:1][b:1][c:1]
 *   a,b,c = register indices, immediate bytes, or address bytes (opcode-defined)
 *
 * Float values are stored as IEEE-754 bit patterns in XVMFiber::r0..r3
 * and in XVMState::shared[].
 */

enum KLSLOp : uint8_t {
    // ── Control ──────────────────────────────────────────────────────────
    OP_NOP          = 0x00,
    OP_RET          = 0xFF,
    OP_JMP          = 0xD0,   // unconditional: pc += (int8_t)a
    OP_JZ           = 0xD1,   // jump if r[a] == 0
    OP_JNZ          = 0xD2,   // jump if r[a] != 0
    OP_CALL         = 0xD3,   // push pc+1, jump to a
    OP_RET_V        = 0xD4,   // pop pc; result in r[a]

    // ── Register / const load ─────────────────────────────────────────────
    OP_LOAD_IMMU    = 0x01,   // r[a] = (uint32_t)((b<<8)|c)  (16-bit unsigned immediate)
    OP_LOAD_CONST   = 0x02,   // r[a] = constPool[b<<8|c] as uint32
    OP_MOV          = 0x03,   // r[a] = r[b]
    OP_ZERO         = 0x04,   // r[a] = 0

    // ── Integer arithmetic ────────────────────────────────────────────────
    OP_IADD         = 0x10,   // r[a] = r[b] + r[c]
    OP_ISUB         = 0x11,
    OP_IMUL         = 0x12,
    OP_IAND         = 0x13,
    OP_IOR          = 0x14,
    OP_IXOR         = 0x15,
    OP_ISHL         = 0x16,
    OP_ISHR         = 0x17,
    OP_IMOD         = 0x18,
    OP_ILT          = 0x19,   // r[a] = (r[b] < r[c]) ? 1 : 0
    OP_IEQ          = 0x1A,

    // ── Float arithmetic (bit-reinterp r[] as IEEE-754) ───────────────────
    OP_FADD         = 0x20,   // r[a] = float(r[b]) + float(r[c])
    OP_FSUB         = 0x21,
    OP_FMUL         = 0x22,
    OP_FDIV         = 0x23,
    OP_FMAD         = 0x24,   // r[a] = r[b]*r[c] + r[a]  (fused)
    OP_FNEG         = 0x25,   // r[a] = -float(r[b])
    OP_FABS         = 0x26,
    OP_FSQRT        = 0x27,
    OP_FMAX         = 0x28,
    OP_FMIN         = 0x29,
    OP_RELU         = 0x2A,   // r[a] = max(0.0f, float(r[b]))
    OP_FLT          = 0x2B,   // r[a] = (float(r[b]) < float(r[c])) ? 1 : 0
    OP_FEQ          = 0x2C,

    // ── Trig (wave propagation / SH) ──────────────────────────────────────
    OP_FSIN         = 0x30,
    OP_FCOS         = 0x31,
    OP_FTAN         = 0x32,
    OP_FEXP         = 0x33,
    OP_FLOG         = 0x34,
    OP_FLOG2        = 0x35,
    OP_FPOW         = 0x36,   // r[a] = pow(float(r[b]), float(r[c]))

    // ── Memory ───────────────────────────────────────────────────────────
    OP_LOAD_SHARED  = 0x40,   // r[a] = shared[r[b]]
    OP_STORE_SHARED = 0x41,   // shared[r[b]] = r[a]
    OP_LOAD_IN0     = 0x42,   // r[a] = input_buffer_0[r[b]]   (t0 register)
    OP_LOAD_IN1     = 0x43,   // r[a] = input_buffer_1[r[b]]   (t1 register)
    OP_LOAD_IN2     = 0x44,   // r[a] = input_buffer_2[r[b]]   (t2 register)
    OP_STORE_OUT0   = 0x45,   // output_buffer_0[r[b]] = r[a]  (u0 register)
    OP_STORE_OUT1   = 0x46,

    // ── Atomic (shared memory, mutex-protected) ────────────────────────────
    OP_ATOMIC_ADD   = 0x50,   // shared[r[b]] += r[a]  (atomic)
    OP_ATOMIC_MAX   = 0x51,
    OP_ATOMIC_MIN   = 0x52,
    OP_ATOMIC_CAS   = 0x53,   // compare-and-swap: shared[r[b]] = (shared[r[b]]==r[c]) ? r[a] : shared[r[b]]

    // ── Thread / fiber primitives ──────────────────────────────────────────
    OP_THREAD_ID    = 0x60,   // r[a] = fiber index (SV_DispatchThreadID.x equiv)
    OP_THREAD_CNT   = 0x61,   // r[a] = total fiber count
    OP_GROUP_ID     = 0x62,   // r[a] = (fid / group_size)
    OP_LOCAL_ID     = 0x63,   // r[a] = (fid % group_size)
    OP_BARRIER      = 0x64,   // phase++ then spin until all fibers reach same phase
    OP_SPAWN        = 0x65,   // fork r[a] child fibers starting at address r[b]
};

// Instruction word (4 bytes)
struct KLSLInstr {
    KLSLOp   op  = OP_NOP;
    uint8_t  a   = 0;
    uint8_t  b   = 0;
    uint8_t  c   = 0;
};
static_assert(sizeof(KLSLInstr) == 4);
