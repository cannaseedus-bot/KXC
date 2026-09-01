// xshard_adapt_fold.hlsl — shard-resident fold-gated optimizer update, cs_5_0
//
// First-pass training kernel for XSHARD weight-bearing F32 shards.
// Host flow:
//   XShardFile::read_shard(seq) -> D3D11 buffer
//   Dispatch((numel + 255) / 256, 1, 1)
//   readback -> XShardFile::commit_shard(seq, adapted_data)
//   XShardFile::set_state(seq, XSHARD_STATE_TRAINED)

cbuffer XShardAdaptParams : register(b0) {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float bias_corr1;
    float bias_corr2;
    uint  numel;
    uint  stride_x;
    float grad_clip;
    float phase_angle;
    uint  fold_id;
    float fold_gate;
    uint  update_mode;  // 0 = Adam, 1 = SGD
    uint2 pad;
};

RWStructuredBuffer<float> weights : register(u0);
RWStructuredBuffer<float> grads   : register(u1);
RWStructuredBuffer<float> m       : register(u2);
RWStructuredBuffer<float> v       : register(u3);

float sanitize_grad(float g) {
    if (isnan(g)) return 0.0f;
    return clamp(g, -grad_clip, grad_clip);
}

float phase_scale() {
    const float two_pi = 6.28318530718f;
    const float phase = cos(phase_angle + (float)(fold_id & 7u) * 0.125f * two_pi);
    return clamp(fold_gate * (0.75f + 0.25f * phase), 0.0f, 1.5f);
}

[numthreads(256, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    const uint i = (gid.y * stride_x + gid.x) * 256 + lid.x;
    if (i >= numel) return;

    const float gate = phase_scale();
    if (gate <= 0.0f) {
        grads[i] = 0.0f;
        return;
    }

    float g = sanitize_grad(grads[i]);
    g = (g + weight_decay * weights[i]) * gate;

    if (update_mode == 1u) {
        weights[i] -= lr * g;
    } else {
        const float mi = beta1 * m[i] + (1.0f - beta1) * g;
        const float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
        m[i] = mi;
        v[i] = vi;

        const float m_hat = mi * bias_corr1;
        const float v_hat = vi * bias_corr2;
        weights[i] -= lr * m_hat / (sqrt(v_hat) + eps);
    }

    grads[i] = 0.0f;
}
