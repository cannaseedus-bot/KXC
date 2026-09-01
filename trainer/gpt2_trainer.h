#pragma once
// gpt2_trainer.h — GPT-2 D3D11 trainer
// Phase 1: GPU Adam (CPU forward/backward, GPU Adam)
// Phase 3: Full GPU pipeline (forward + backward + Adam all on GPU)
//
// π-nary <THINK> skinning (Decision A):
//   buildThinkDepth() fills think_depth_buf_ per-sequence using
//   SphericalGeometry::GeodesicDistanceF(wte[tokens[think_start]], wte[tokens[t]], R).
//   think_depth[t] ∈ [0, π] — great-circle arc from <THINK> entry; 0.0 outside span.
//   Dispatches gpt2_kuhul_think_bias.hlsl between QK and softmax (requires attn split).
//   bias scale driven by physics_->GetAntigravityFloat() — grows as training stabilises.

#include "gpt2_config.h"
#include "d3d11_engine.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>

using Microsoft::WRL::ComPtr;

struct AdamParam {
    uint32_t    numel  = 0;
    std::string name;
    std::vector<size_t> shape;             // tensor shape (for save)
    float*       cpu_w = nullptr;          // points into cpu_w_owned
    std::vector<float> cpu_w_owned;        // CPU mirror
    std::vector<float> cpu_g;             // CPU grad (Phase 1 path only)
    ComPtr<ID3D11Buffer> w_buf;            // GPU weights (authoritative)
    ComPtr<ID3D11Buffer> g_buf;            // GPU gradients
    ComPtr<ID3D11Buffer> m_buf;            // Adam first moment
    ComPtr<ID3D11Buffer> v_buf;            // Adam second moment
};

struct TrainerConfig {
    float    lr           = 3e-5f;
    float    beta1        = 0.9f;
    float    beta2        = 0.999f;
    float    eps          = 1e-8f;
    float    weight_decay = 0.01f;
    uint32_t batch_size   = 4;
    uint32_t block_size   = 128;
    uint32_t max_steps      = 1000;
    uint32_t save_every     = 200;
    uint32_t log_every      = 50;
    bool     use_gpu_fwd    = true;  // false = CPU forward/backward (Phase 1)
    bool     completion_only = false; // mask loss to <INSTRUCT>...</INSTRUCT> spans only
    bool     no_shuffle     = false;  // train in dataset order (use with --group-by-lang)
    std::string data_path;
    std::string model_path;       // safetensors weights file; empty = random init from preset
    std::string preset_name;      // "distilgpt2"|"gpt2-small"|"gpt2-medium"|"gpt2-large"; empty = auto from file
    std::string output_path;
};

class KuhulPhysicsSolver;  // pi_kuhul/KuhulPhysics.h — adaptive gradient-gravity controller (fwd-decl)

class GPT2Trainer {
public:
    explicit GPT2Trainer(D3D11Engine* d11);
    ~GPT2Trainer();

    bool  init(const TrainerConfig& cfg);
    float train_step(const std::vector<std::vector<int32_t>>& batch);
    void  train();
    bool  save(const std::string& path = "");

    const GPT2Config& modelConfig() const { return model_cfg_; }

private:
    D3D11Engine*  d11_;
    TrainerConfig cfg_;
    GPT2Config    model_cfg_;
    int           step_    = 0;
    int           warmup_steps_ = 0;   // linear LR warmup (set from GPT2_WARMUP env)
    float         grad_clip_ = 1.0f;   // Adam gradient clip bound (GPT2_CLIP, default 1.0)
    bool          adaptive_clip_ = false; // GPT2_ADAPTIVE_CLIP: drive grad_clip_ from KuhulPhysics
    KuhulPhysicsSolver* physics_ = nullptr; // adaptive-gravity controller (nullptr unless adaptive)
    bool          fullseq_ = false;    // full-sequence loss over all positions (GPT2_FULLSEQ env)
    bool          think_bias_ = false; // GPT2_THINK_BIAS: π-nary <THINK> attention skinning
    ComPtr<ID3D11Buffer> loss_all_;    // [S] per-position loss (fullseq mode)
    float         beta1_t_ = 1.0f;
    float         beta2_t_ = 1.0f;

    // CPU weight map (Phase 1 forward)
    std::vector<uint8_t> weight_blob_;
    std::unordered_map<std::string, std::pair<const float*, uint32_t>> cpu_weights_;

    // Per-parameter Adam state
    std::vector<AdamParam>                    params_;
    std::unordered_map<std::string, uint32_t> param_idx_;

    // ── Compute shaders ──────────────────────────────────────────────────────
    ComPtr<ID3D11ComputeShader> cs_adam_;
    // Phase 3 forward kernels
    ComPtr<ID3D11ComputeShader> cs_embed_fwd_;
    ComPtr<ID3D11ComputeShader> cs_lnorm_fwd_;
    ComPtr<ID3D11ComputeShader> cs_matmul_fwd_;
    ComPtr<ID3D11ComputeShader> cs_matmul_fwd_transb_;
    ComPtr<ID3D11ComputeShader> cs_attn_fwd_;          // monolithic (kept for reference)
    ComPtr<ID3D11ComputeShader> cs_attn_qk_dot_;       // split pass 1: QK logits → P_buf
    ComPtr<ID3D11ComputeShader> cs_attn_softmax_;      // split pass 2: softmax + V accumulation
    ComPtr<ID3D11ComputeShader> cs_gelu_fwd_;       // gpt2_gelu_fwd.hlsl
    ComPtr<ID3D11ComputeShader> cs_resadd_add3_;    // gpt2_residual_add.hlsl CSMain_add3
    ComPtr<ID3D11ComputeShader> cs_resadd_addto_;   // gpt2_residual_add.hlsl CSMain_addto
    // Phase 3 backward kernels
    ComPtr<ID3D11ComputeShader> cs_loss_;
    ComPtr<ID3D11ComputeShader> cs_lnorm_bwd_;        // CSMain: dx only (no dgamma/dbeta race)
    ComPtr<ID3D11ComputeShader> cs_lnorm_bwd_params_; // CSMain_params: dgamma/dbeta, race-free
    ComPtr<ID3D11ComputeShader> cs_gelu_bwd_;
    ComPtr<ID3D11ComputeShader> cs_attn_bwd_dvdp_;
    ComPtr<ID3D11ComputeShader> cs_attn_bwd_dq_;
    ComPtr<ID3D11ComputeShader> cs_attn_bwd_dk_;
    ComPtr<ID3D11ComputeShader> cs_matmul_bwd_dA_;
    ComPtr<ID3D11ComputeShader> cs_matmul_bwd_dB_;
    ComPtr<ID3D11ComputeShader> cs_embed_bwd_;
    ComPtr<ID3D11ComputeShader> cs_bias_bwd_;       // gpt2_bias_bwd.hlsl

    // ── Persistent cbuffers ──────────────────────────────────────────────────
    ComPtr<ID3D11Buffer> adam_cb_;   // 32 bytes, DYNAMIC
    ComPtr<ID3D11Buffer> gen_cb_;    // 64 bytes, DYNAMIC — general params

    // ── Phase 3 activation buffers (allocated once at init) ──────────────────
    uint32_t max_S_ = 0;  // cfg_.block_size

    // Per-layer activation buffers — one buffer per layer (each at offset 0).
    // Intel HD 4600 driver ignores SRV.FirstElement for structured buffers,
    // so flat [NL, ...] layouts with non-zero SRV offsets read wrong data.
    // Using per-layer buffers avoids all non-zero SRV FirstElement usage.
    std::vector<ComPtr<ID3D11Buffer>> h_buf_;        // [NL+1] each [max_S*E]
    std::vector<ComPtr<ID3D11Buffer>> qkv_buf_;      // [NL]   each [max_S*3E]
    std::vector<ComPtr<ID3D11Buffer>> xhat_ln1_buf_; // [NL]   each [max_S*E]
    std::vector<ComPtr<ID3D11Buffer>> xhat_ln2_buf_; // [NL]   each [max_S*E]
    std::vector<ComPtr<ID3D11Buffer>> inv_std_ln1_;  // [NL]   each [max_S]
    std::vector<ComPtr<ID3D11Buffer>> inv_std_ln2_;  // [NL]   each [max_S]
    std::vector<ComPtr<ID3D11Buffer>> P_buf_;        // [NL]   each [H*max_S*max_S]
    std::vector<ComPtr<ID3D11Buffer>> ln1_y_buf_;    // [NL]   each [max_S*E]
    std::vector<ComPtr<ID3D11Buffer>> ln2_y_buf_;    // [NL]   each [max_S*E]
    std::vector<ComPtr<ID3D11Buffer>> attn_out_buf_; // [NL]   each [max_S*E]
    std::vector<ComPtr<ID3D11Buffer>> mlp_pre_buf_;  // [NL]   each [max_S*F]
    std::vector<ComPtr<ID3D11Buffer>> mlp_gelu_buf_; // [NL]   each [max_S*F]
    // Final layernorm
    ComPtr<ID3D11Buffer> xhat_lnf_buf_;  // [max_S, E]
    ComPtr<ID3D11Buffer> inv_std_lnf_;   // [max_S]
    ComPtr<ID3D11Buffer> lnf_y_buf_;     // [max_S, E] — final LN output (LM head input)
    // LM head
    ComPtr<ID3D11Buffer> logits_buf_;    // [V]
    ComPtr<ID3D11Buffer> dlogits_buf_;   // [V] — separate from logits for loss shader
    // Loss
    ComPtr<ID3D11Buffer> loss_buf_;      // [1]
    ComPtr<ID3D11Buffer> loss_staging_;  // [1] CPU-readable
    // Gradient buffers (reused per layer in backward)
    ComPtr<ID3D11Buffer> dh_buf_;        // [max_S, E]
    ComPtr<ID3D11Buffer> d_qkv_buf_;    // [max_S, 3E] — also used as d_attn_out temp
    ComPtr<ID3D11Buffer> d_mlp_buf_;    // [max_S, F]  — MLP gradient working buffer
    ComPtr<ID3D11Buffer> dP_tmp_buf_;   // [max_S, max_S] — attn bwd temp (per head)
    ComPtr<ID3D11Buffer> dot_row_buf_;  // [max_S]         — attn bwd temp (per head)
    // Token buffer
    ComPtr<ID3D11Buffer> tokens_buf_;    // [max_S] int32

    // π-nary <THINK> skinning + 4-bone LBS brain routing (think_bias_ mode)
    ComPtr<ID3D11Buffer>         think_depth_buf_;      // [max_S]   float — geodesic arc depth ∈ [0,π]
    ComPtr<ID3D11Buffer>         seq_bone_ids_buf_;     // [max_S*4] int32 — 4 bone IDs per token (LBS)
    ComPtr<ID3D11Buffer>         seq_bone_weights_buf_; // [max_S*4] float — blend weights (sum=1.0)
    ComPtr<ID3D11Buffer>         think_bias_cb_;        // ThinkBiasCB constant buffer (DYNAMIC, 48 B)
    ComPtr<ID3D11ComputeShader>  cs_kuhul_think_bias_; // gpt2_kuhul_think_bias.hlsl

    // TENSOR FOLD CORE — XVM linear cluster dispatch (GPT2_TENSOR_FOLD env)
    static constexpr uint32_t   N_FOLD_CLUSTERS      = 64;  // max bone cluster IDs
    bool                        tensor_fold_core_    = false;
    ComPtr<ID3D11ComputeShader> cs_bone_argsort_;            // cs_bone_argsort_.hlsl
    ComPtr<ID3D11ComputeShader> cs_fold_kernel_compute_;     // cs_fold_kernel_compute_.hlsl (default COMPUTE_FOLD)
    ComPtr<ID3D11Buffer>        bone_sort_cb_;               // BoneSortCB   (DYNAMIC, 16 bytes)
    ComPtr<ID3D11Buffer>        fold_dispatch_cb_;           // FoldDispatchCB (DYNAMIC, 16 bytes)
    ComPtr<ID3D11Buffer>        sorted_token_idx_buf_;       // [max_S] uint — cluster-ordered token indices
    ComPtr<ID3D11Buffer>        cluster_start_buf_;          // [N_FOLD_CLUSTERS] uint
    ComPtr<ID3D11Buffer>        cluster_count_buf_;          // [N_FOLD_CLUSTERS] uint
    // CPU-side cluster sort (avoids GPU readback stall in per-cluster dispatch loop)
    std::vector<uint32_t>       cpu_sorted_idx_;
    std::vector<uint32_t>       cpu_cluster_start_;
    std::vector<uint32_t>       cpu_cluster_count_;
    // Fold dispatch table: bone cluster ID → compiled fold kernel + pressure metadata
    struct FoldKernel { ComPtr<ID3D11ComputeShader> cs; float pressure; bool antigravity; };
    std::unordered_map<int32_t, FoldKernel> fold_dispatch_table_;
    bool loadFoldKernels();

    // Per-layer gravity well sync (GPT2_GRAVITY_SYNC env)
    bool                        gravity_sync_        = false;
    ComPtr<ID3D11ComputeShader> cs_gravity_field_layer_;  // cs_gravity_field_layer_.hlsl
    ComPtr<ID3D11Buffer>        gravity_layer_cb_;         // GravityLayerCB (DYNAMIC, 16 bytes)
    ComPtr<ID3D11Query>         gravity_sync_query_;       // D3D11_QUERY_EVENT — per-layer fence

    // Brain expert routing — loaded once from brain2/experts.bin
    std::vector<int32_t> brain_experts_;   // [n_brain_nodes_] expert cluster IDs 0-60
    int                  n_brain_nodes_ = 0;
    float                brain_scale_   = 0.0f; // derived per-step from antigravity_scale * 0.5

    // FOLD → expert cluster map: kuhul_fold_experts_[open_tok_id] = cluster_id
    // Loaded from safetensors __metadata__["kuhul_brain_fold"] (written by fold_kuhul_to_brain.py).
    // Nodes inside a fold inherit the fold's cluster rather than their own token hash.
    std::unordered_map<int32_t, int32_t> kuhul_fold_experts_;

    // SRV cache for weight buffers (built at init)
    std::unordered_map<std::string, ComPtr<ID3D11ShaderResourceView>> param_srv_;

    // ── Helpers ──────────────────────────────────────────────────────────────
    bool  loadShaders();
    bool  loadWeights(const std::string& path);
    bool  initWeightsFromScratch();    // random init using model_cfg_ preset (no safetensors)
    bool  allocWorkingBuffers();

    // Load brain expert cluster IDs from path (brain2/experts.bin — int32 per node).
    bool  loadBrainExperts(const std::string& path);

    // Build think_depth[S] and seq_expert[S] for current token sequence:
    //   think_depth[t] = GeodesicDistanceF(wte[tokens[think_start]], wte[tokens[t]], R)  ∈ [0,π]
    //   seq_expert[t]  = brain_experts_[tokens[t] % n_brain_nodes_]    ∈ [0, 60] (or -1)
    //   Uploads both buffers. Called once per training step when think_bias_=true.
    void  buildThinkDepth(const std::vector<int32_t>& seq, int think_token_id, int end_token_id);

    // Phase 1 path
    void  adamStepGPU(bool upload_cpu_grads = true);
    void  dispatchAdam(AdamParam& p, float bc1, float bc2, bool upload_cpu_grads);
    void  syncWeightsToCPU();  // readback w_buf → cpu_w_owned for all params (Phase 3 save)
    float forwardBackwardCPU(const std::vector<int32_t>& seq, bool accumulate_grads);
    void  zeroCPUGrads();

    // Phase 3 path
    float gpuForwardBackward(const std::vector<int32_t>& seq, float inv_batch);
    void  zeroGPUGrads();
    float readbackLoss();

    // GPU dispatch helpers
    ComPtr<ID3D11UnorderedAccessView> makeUAV(ID3D11Buffer* buf, uint32_t first, uint32_t n);
    ComPtr<ID3D11ShaderResourceView>  makeSRV(ID3D11Buffer* buf, uint32_t first, uint32_t n);
    ComPtr<ID3D11ShaderResourceView>  makeSRVi(ID3D11Buffer* buf, uint32_t first, uint32_t n); // int32
    void  setCB(const void* data, uint32_t bytes);  // writes gen_cb_ and binds slot 0
    void  clearViews(uint32_t nuav, uint32_t nsrv);

    // Weight SRV lookup (by param name)
    ID3D11ShaderResourceView* wSRV(const std::string& name);
    ID3D11Buffer*              wBuf(const std::string& name);

    const float* w(const std::string& name) const;

    ComPtr<ID3D11Buffer> createBuffer(uint32_t bytes, bool uav, bool staging = false);
    ComPtr<ID3D11Buffer> createAndUpload(const float* data, uint32_t numel);
    ComPtr<ID3D11Buffer> createIntBuffer(uint32_t n_ints, bool uav);
    std::vector<float>   readbackBuffer(ID3D11Buffer* buf, uint32_t numel);
    void  uploadToBuffer(ID3D11Buffer* buf, const float* data, uint32_t numel);
};
