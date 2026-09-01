// gpt2_attn_bwd.hlsl — Attention backward, interleaved QKV, three-pass, race-free, cs_5_0
//
// Three entry points — call in order per (layer, head):
//   Pass 1: CSMain_dVdP  Dispatch(seq_len, 1, 1)  numthreads(1,1,1)  gid.x = j
//   Pass 2: CSMain_dQ    Dispatch(seq_len, 1, 1)  numthreads(1,1,1)  gid.x = i
//   Pass 3: CSMain_dK    Dispatch(seq_len, 1, 1)  numthreads(1,1,1)  gid.x = j
//
// QKV layout: qkv[S, 3E]
//   Q[s,d] = qkv[s*3E + h*D + d]
//   K[s,d] = qkv[s*3E + E + h*D + d]
//   V[s,d] = qkv[s*3E + 2E + h*D + d]
//
// d_qkv: same interleaved layout (accumulate +=)
// d_attn_out: [S, E] — head h at row s: d_attn_out[s*E + h*D .. s*E + (h+1)*D - 1]
//
// Zero dP_tmp[S*S] before Pass 1. Zero dot_row[S] before Pass 2.

cbuffer AttnBwdParams : register(b0) {
    uint  seq_len;       // S
    uint  head_dim;      // D = E/H
    uint  n_embd;        // E
    uint  head;          // h (current head index)
    float scale;         // 1/sqrt(D)
    uint  P_head_offset; // = h * S * S  (byte offset into full per-layer P buffer)
    uint2 pad;
};

StructuredBuffer<float>   qkv        : register(t0);  // [S, 3E] forward QKV
StructuredBuffer<float>   P          : register(t1);  // [S, S] softmax weights for this head
StructuredBuffer<float>   d_attn_out : register(t2);  // [S, E] upstream gradient

RWStructuredBuffer<float> d_qkv      : register(u0);  // [S, 3E] gradient (accumulate +=)
RWStructuredBuffer<float> dP_tmp     : register(u1);  // [S, S] temp (zero before Pass 1)
RWStructuredBuffer<float> dot_row    : register(u2);  // [S] temp (zero before Pass 2)

// ── Pass 1: dV[j,:] and dP_tmp[i,j] ─────────────────────────────────────────
// gid.x = j.  Only thread j writes dV[j,:] — no race.
// dP_tmp[i,j] written by exactly one thread j — no race.
[numthreads(1, 1, 1)]
void CSMain_dVdP(uint3 gid : SV_GroupID) {
    const uint j = gid.x;
    const uint S = seq_len, D = head_dim, E = n_embd, h = head;
    if (j >= S) return;

    // dV[j,d] = sum_{i>=j} P[i,j] * dOut[i, h*D+d]
    for (uint d = 0; d < D; ++d) {
        float acc = 0.f;
        for (uint i = j; i < S; ++i)
            acc += P[P_head_offset + i * S + j] * d_attn_out[i * E + h * D + d];
        d_qkv[j * 3*E + 2*E + h*D + d] += acc;
    }

    // dP_tmp[i,j] = dot(dOut[i, h slice], V[j, h slice])  for i >= j, 0 otherwise
    for (uint i = 0; i < S; ++i) {
        if (i < j) {
            dP_tmp[i * S + j] = 0.f;
        } else {
            float dp = 0.f;
            for (uint d = 0; d < D; ++d)
                dp += d_attn_out[i * E + h*D + d] * qkv[j * 3*E + 2*E + h*D + d];
            dP_tmp[i * S + j] = dp;
        }
    }
}

// ── Pass 2: dQ[i,:] and dot_row[i] ───────────────────────────────────────────
// gid.x = i.  Only thread i writes dQ[i,:] and dot_row[i] — no race.
[numthreads(1, 1, 1)]
void CSMain_dQ(uint3 gid : SV_GroupID) {
    const uint i = gid.x;
    const uint S = seq_len, D = head_dim, E = n_embd, h = head;
    if (i >= S) return;

    // dot_i = sum_{j<=i} P[i,j] * dP_tmp[i,j]
    float dot_i = 0.f;
    for (uint j = 0; j <= i; ++j)
        dot_i += P[P_head_offset + i * S + j] * dP_tmp[i * S + j];
    dot_row[i] = dot_i;

    // dQ[i,d] += sum_{j<=i} dS[i,j] * K[j,d]
    // dS[i,j] = scale * P[i,j] * (dP_tmp[i,j] - dot_i)
    for (uint d = 0; d < D; ++d) {
        float acc = 0.f;
        for (uint j = 0; j <= i; ++j) {
            float dS = scale * P[P_head_offset + i * S + j] * (dP_tmp[i * S + j] - dot_i);
            acc += dS * qkv[j * 3*E + E + h*D + d];  // K[j,d]
        }
        d_qkv[i * 3*E + h*D + d] += acc;  // dQ[i,d]
    }
}

// ── Pass 3: dK[j,:] ──────────────────────────────────────────────────────────
// gid.x = j.  Only thread j writes dK[j,:] — no race.
[numthreads(1, 1, 1)]
void CSMain_dK(uint3 gid : SV_GroupID) {
    const uint j = gid.x;
    const uint S = seq_len, D = head_dim, E = n_embd, h = head;
    if (j >= S) return;

    // dK[j,d] += sum_{i>=j} dS[i,j] * Q[i,d]
    for (uint d = 0; d < D; ++d) {
        float acc = 0.f;
        for (uint i = j; i < S; ++i) {
            float dS = scale * P[P_head_offset + i * S + j] * (dP_tmp[i * S + j] - dot_row[i]);
            acc += dS * qkv[i * 3*E + h*D + d];  // Q[i,d]
        }
        d_qkv[j * 3*E + E + h*D + d] += acc;  // dK[j,d]
    }
}
