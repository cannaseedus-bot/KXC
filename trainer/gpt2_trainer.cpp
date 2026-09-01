#define NOMINMAX
// gpt2_trainer.cpp — GPT-2 trainer: CPU forward/backward + GPU Adam
//
// Forward pass runs on CPU using weights kept in system RAM (the safetensors blob).
// Backward pass is exact backprop through all layers.
// Adam weight update runs on GPU via cs_adam_ compute shader.
// Loss is real cross-entropy from actual token targets.

#include "gpt2_trainer.h"
#include "../pi_kuhul/KuhulPhysics.h"    // adaptive gradient-gravity controller (GPT2_ADAPTIVE_CLIP)
#include "../pi_kuhul/DirectXMathAVX2.h" // FMA3 (_mm256_fmadd_ps) + F16C for cosine-sim & Q8 hot-swap
#include "d3d11_engine.h"   // trainer-local copy (xvm-d3d12 D3D11Engine with rawCtx/rawDevice); repo src/ has an incompatible variant
#include <nlohmann/json.hpp>
#include <d3dcompiler.h>
#include <windows.h>
#include <filesystem>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <random>
#include <unordered_map>

using json = nlohmann::json;
#pragma comment(lib, "d3dcompiler")

// Shader lookup must not depend on the caller's current directory. The
// trainer is launched by shells, dashboards, and runtime hosts from several
// different working directories, so resolve packaged sources from the EXE.
static std::filesystem::path trainerExeDir() {
    char buffer[MAX_PATH]{};
    DWORD n = GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return std::filesystem::current_path();
    return std::filesystem::path(buffer).parent_path();
}

static std::filesystem::path resolveShader(const std::string& requested) {
    const std::filesystem::path name = std::filesystem::path(requested).filename();
    const auto exe = trainerExeDir();
    const std::filesystem::path candidates[] = {
        std::filesystem::path(requested),
        exe / ".." / "shaders" / name,
        exe / "shaders" / name,
        std::filesystem::current_path() / "trainer" / "build" / "shaders" / name,
        std::filesystem::current_path() / "dist" / "kuhul-runtime-v1" / "shaders" / name
    };
    for (const auto& candidate : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec)) return candidate;
    }
    return candidates[1];
}

static const bool kHasAVX2 = DirectX::AVX2::XMVerifyAVX2Support();

// ── math helpers ─────────────────────────────────────────────────────────────

static inline float gelu_fwd(float x) {
    const float k = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(k));
}

static inline float gelu_bwd(float x) {
    const float k  = 0.7978845608f * (x + 0.044715f * x * x * x);
    const float t  = tanhf(k);
    return 0.5f*(1.f+t) + 0.5f*x*(1.f-t*t)*0.7978845608f*(1.f+3.f*0.044715f*x*x);
}

static void layernorm_fwd(const float* x, const float* g, const float* b,
                           uint32_t N, float eps,
                           float* y, float* xhat, float& mean_out, float& inv_std_out) {
    float mean = 0.f;
    for (uint32_t i = 0; i < N; ++i) mean += x[i];
    mean /= N;
    float var = 0.f;
    for (uint32_t i = 0; i < N; ++i) { float d = x[i]-mean; var += d*d; }
    var /= N;
    float inv_std = 1.f / sqrtf(var + eps);
    mean_out = mean; inv_std_out = inv_std;
    for (uint32_t i = 0; i < N; ++i) {
        xhat[i] = (x[i]-mean)*inv_std;
        y[i]    = g[i]*xhat[i] + b[i];
    }
}

// dy: upstream gradient [N]; x: input [N]; xhat: saved normalized [N]
// g: gamma [N]; inv_std: saved from forward
// dx, dg, db: output gradients [N]  (accumulated += )
static void layernorm_bwd(const float* dy, const float* xhat, const float* g,
                           uint32_t N, float inv_std,
                           float* dx, float* dg, float* db) {
    float dot = 0.f, sum_dy = 0.f;
    for (uint32_t i = 0; i < N; ++i) { dot += dy[i]*g[i]*xhat[i]; sum_dy += dy[i]*g[i]; }
    for (uint32_t i = 0; i < N; ++i) {
        dx[i] += inv_std * (dy[i]*g[i] - (sum_dy + xhat[i]*dot)/N);
        dg[i] += dy[i]*xhat[i];
        db[i] += dy[i];
    }
}

// C = A @ B  (M×K) × (K×N) → (M×N)
static void matmul(const float* A, const float* B, float* C, uint32_t M, uint32_t K, uint32_t N) {
    for (uint32_t i = 0; i < M; ++i)
        for (uint32_t j = 0; j < N; ++j) {
            float acc = 0.f;
            for (uint32_t k = 0; k < K; ++k) acc += A[i*K+k] * B[k*N+j];
            C[i*N+j] = acc;
        }
}

// dA += dC @ B.T;  dB += A.T @ dC
static void matmul_bwd(const float* dC, const float* A, const float* B,
                        float* dA, float* dB,
                        uint32_t M, uint32_t K, uint32_t N) {
    // dA[i,k] += sum_j dC[i,j] * B[k,j]
    for (uint32_t i = 0; i < M; ++i)
        for (uint32_t k = 0; k < K; ++k) {
            float acc = 0.f;
            for (uint32_t j = 0; j < N; ++j) acc += dC[i*N+j] * B[k*N+j];
            dA[i*K+k] += acc;
        }
    // dB[k,j] += sum_i A[i,k] * dC[i,j]
    for (uint32_t k = 0; k < K; ++k)
        for (uint32_t j = 0; j < N; ++j) {
            float acc = 0.f;
            for (uint32_t i = 0; i < M; ++i) acc += A[i*K+k] * dC[i*N+j];
            dB[k*N+j] += acc;
        }
}

// bias backward: db[j] += sum_i dC[i,j]
static void bias_bwd(const float* dC, float* db, uint32_t M, uint32_t N) {
    for (uint32_t j = 0; j < N; ++j) {
        float acc = 0.f;
        for (uint32_t i = 0; i < M; ++i) acc += dC[i*N+j];
        db[j] += acc;
    }
}

// ── constructor ───────────────────────────────────────────────────────────────

GPT2Trainer::GPT2Trainer(D3D11Engine* d11) : d11_(d11) {}
GPT2Trainer::~GPT2Trainer() {}

// ── buffer helpers ────────────────────────────────────────────────────────────

ComPtr<ID3D11Buffer> GPT2Trainer::createBuffer(uint32_t bytes, bool uav, bool staging) {
    auto* dev = d11_->rawDevice();
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = bytes;
    if (staging) {
        desc.Usage = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    } else {
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (uav) desc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(float);
    }
    ComPtr<ID3D11Buffer> buf;
    dev->CreateBuffer(&desc, nullptr, &buf);
    return buf;
}

ComPtr<ID3D11Buffer> GPT2Trainer::createAndUpload(const float* data, uint32_t numel) {
    auto* dev = d11_->rawDevice();
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = numel * sizeof(float);
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = sizeof(float);
    D3D11_SUBRESOURCE_DATA init{};
    init.pSysMem = data;
    ComPtr<ID3D11Buffer> buf;
    dev->CreateBuffer(&desc, data ? &init : nullptr, &buf);
    return buf;
}

std::vector<float> GPT2Trainer::readbackBuffer(ID3D11Buffer* buf, uint32_t numel) {
    auto* dev = d11_->rawDevice();
    auto* ctx = d11_->rawCtx();
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = numel * sizeof(float);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    // Must match source buffer's MiscFlags for CopyResource to succeed
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = sizeof(float);
    ComPtr<ID3D11Buffer> staging;
    dev->CreateBuffer(&desc, nullptr, &staging);
    if (!staging) return std::vector<float>(numel, 0.f);
    ctx->CopyResource(staging.Get(), buf);
    D3D11_MAPPED_SUBRESOURCE mapped{};
    ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    std::vector<float> out(numel, 0.f);
    if (mapped.pData) std::memcpy(out.data(), mapped.pData, numel * sizeof(float));
    ctx->Unmap(staging.Get(), 0);
    return out;
}

// Upload a CPU float array into an existing GPU buffer (dynamic update path)
void GPT2Trainer::uploadToBuffer(ID3D11Buffer* buf, const float* data, uint32_t numel) {
    auto* ctx = d11_->rawCtx();
    auto* dev = d11_->rawDevice();
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = numel * sizeof(float);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = sizeof(float);
    ComPtr<ID3D11Buffer> staging;
    dev->CreateBuffer(&desc, nullptr, &staging);
    if (!staging) return;
    D3D11_MAPPED_SUBRESOURCE ms{};
    if (SUCCEEDED(ctx->Map(staging.Get(), 0, D3D11_MAP_WRITE, 0, &ms)) && ms.pData) {
        std::memcpy(ms.pData, data, numel * sizeof(float));
        ctx->Unmap(staging.Get(), 0);
        ctx->CopyResource(buf, staging.Get());
    }
}

// ── shader loading ────────────────────────────────────────────────────────────

static ComPtr<ID3D11ComputeShader> compileCS(ID3D11Device* dev,
                                              const std::string& path,
                                              const char* entry) {
    ComPtr<ID3DBlob> blob, err;
    const auto resolved = resolveShader(path);
    const auto wide = resolved.wstring();
    HRESULT hr = D3DCompileFromFile(
        wide.c_str(),
        nullptr, nullptr, entry, "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &err);
    if (FAILED(hr)) {
        if (err) std::cerr << "[trainer] shader error in " << resolved.string() << ": "
                           << (char*)err->GetBufferPointer() << "\n";
        return nullptr;
    }
    ComPtr<ID3D11ComputeShader> cs;
    dev->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(),
                             nullptr, &cs);
    return cs;
}

bool GPT2Trainer::loadShaders() {
    auto* dev = d11_->rawDevice();
    const std::string base = "shaders/";

    cs_adam_ = compileCS(dev, base + "gpt2_adam.hlsl",         "CSMain");
    if (!cs_adam_) { std::cerr << "[trainer] gpt2_adam.hlsl failed\n"; return false; }

    if (cfg_.use_gpu_fwd) {
        cs_embed_fwd_       = compileCS(dev, base + "gpt2_embed_fwd.hlsl",      "CSMain");
        cs_lnorm_fwd_       = compileCS(dev, base + "gpt2_layernorm_fwd.hlsl",  "CSMain");
        cs_matmul_fwd_      = compileCS(dev, base + "gpt2_matmul_fwd.hlsl",     "CSMain");
        cs_matmul_fwd_transb_= compileCS(dev, base + "gpt2_matmul_fwd.hlsl",   "CSMain_transB");
        cs_attn_fwd_        = compileCS(dev, base + "gpt2_attn_fwd.hlsl",       "CSMain");
        cs_attn_qk_dot_     = compileCS(dev, base + "gpt2_attn_qk_dot_.hlsl",  "CSMain");
        cs_attn_softmax_    = compileCS(dev, base + "gpt2_attn_softmax_.hlsl", "CSMain");
        cs_gelu_fwd_        = compileCS(dev, base + "gpt2_gelu_fwd.hlsl",       "CSMain");
        cs_resadd_add3_     = compileCS(dev, base + "gpt2_residual_add.hlsl",   "CSMain_add3");
        cs_resadd_addto_    = compileCS(dev, base + "gpt2_residual_add.hlsl",   "CSMain_addto");
        cs_loss_            = compileCS(dev, base + "gpt2_loss.hlsl",            "CSMain");
        cs_lnorm_bwd_       = compileCS(dev, base + "gpt2_layernorm_bwd.hlsl",  "CSMain");
        cs_lnorm_bwd_params_= compileCS(dev, base + "gpt2_layernorm_bwd.hlsl",  "CSMain_params");
        cs_gelu_bwd_        = compileCS(dev, base + "gpt2_gelu_bwd.hlsl",       "CSMain");
        cs_attn_bwd_dvdp_   = compileCS(dev, base + "gpt2_attn_bwd.hlsl",       "CSMain_dVdP");
        cs_attn_bwd_dq_     = compileCS(dev, base + "gpt2_attn_bwd.hlsl",       "CSMain_dQ");
        cs_attn_bwd_dk_     = compileCS(dev, base + "gpt2_attn_bwd.hlsl",       "CSMain_dK");
        cs_matmul_bwd_dA_   = compileCS(dev, base + "gpt2_matmul_bwd.hlsl",    "CSMain_dA");
        cs_matmul_bwd_dB_   = compileCS(dev, base + "gpt2_matmul_bwd.hlsl",    "CSMain_dB");
        cs_embed_bwd_       = compileCS(dev, base + "gpt2_embed_bwd.hlsl",      "CSMain");
        cs_bias_bwd_        = compileCS(dev, base + "gpt2_bias_bwd.hlsl",       "CSMain");

        bool ok = cs_embed_fwd_ && cs_lnorm_fwd_ && cs_matmul_fwd_ && cs_matmul_fwd_transb_
               && cs_attn_fwd_  && cs_attn_qk_dot_ && cs_attn_softmax_
               && cs_gelu_fwd_  && cs_resadd_add3_ && cs_resadd_addto_
               && cs_loss_      && cs_lnorm_bwd_  && cs_lnorm_bwd_params_ && cs_gelu_bwd_
               && cs_attn_bwd_dvdp_ && cs_attn_bwd_dq_ && cs_attn_bwd_dk_
               && cs_matmul_bwd_dA_ && cs_matmul_bwd_dB_ && cs_embed_bwd_ && cs_bias_bwd_;
        if (!ok) { std::cerr << "[trainer] one or more Phase-3 shaders failed\n"; return false; }

        // Optional π-nary + brain think-bias shader (gated by GPT2_THINK_BIAS env flag)
        think_bias_ = std::getenv("GPT2_THINK_BIAS") != nullptr;
        if (think_bias_) {
            cs_kuhul_think_bias_ = compileCS(dev, base + "gpt2_kuhul_think_bias.hlsl", "CSMain");
            if (!cs_kuhul_think_bias_) {
                std::cerr << "[trainer] gpt2_kuhul_think_bias.hlsl failed — think-bias disabled\n";
                think_bias_ = false;
            } else {
                std::cerr << "[trainer] π-nary + brain think-bias shader ready\n";
            }
        }

        // Optional per-layer gravity well sync (gated by GPT2_GRAVITY_SYNC env flag)
        gravity_sync_ = std::getenv("GPT2_GRAVITY_SYNC") != nullptr;
        if (gravity_sync_) {
            cs_gravity_field_layer_ = compileCS(dev, base + "cs_gravity_field_layer_.hlsl", "CSMain");
            if (!cs_gravity_field_layer_) {
                std::cerr << "[trainer] cs_gravity_field_layer_.hlsl failed — gravity sync disabled\n";
                gravity_sync_ = false;
            } else {
                std::cerr << "[trainer] gravity field layer shader ready\n";
            }
        }

        // Optional TENSOR FOLD CORE — XVM linear cluster dispatch (GPT2_TENSOR_FOLD env)
        tensor_fold_core_ = std::getenv("GPT2_TENSOR_FOLD") != nullptr;
        if (tensor_fold_core_) {
            cs_bone_argsort_        = compileCS(dev, base + "cs_bone_argsort_.hlsl",        "CSMain");
            cs_fold_kernel_compute_ = compileCS(dev, base + "cs_fold_kernel_compute_.hlsl", "CSMain");
            if (!cs_bone_argsort_ || !cs_fold_kernel_compute_) {
                std::cerr << "[trainer] TENSOR FOLD CORE shaders failed — disabled\n";
                tensor_fold_core_ = false;
            } else {
                loadFoldKernels();
                std::cerr << "[trainer] TENSOR FOLD CORE ready ("
                          << fold_dispatch_table_.size() << " fold kernels, "
                          << N_FOLD_CLUSTERS << " cluster slots)\n";
            }
        }
    }

    // Persistent cbuffers
    auto makeCB = [&](uint32_t bytes) -> ComPtr<ID3D11Buffer> {
        D3D11_BUFFER_DESC d{};
        d.ByteWidth = bytes; d.Usage = D3D11_USAGE_DYNAMIC;
        d.BindFlags = D3D11_BIND_CONSTANT_BUFFER; d.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        ComPtr<ID3D11Buffer> b; dev->CreateBuffer(&d, nullptr, &b); return b;
    };
    adam_cb_          = makeCB(48);  // AdamCB: 7 floats + numel + stride_x + 3 pad = 48 bytes
    gen_cb_           = makeCB(64);
    think_bias_cb_    = makeCB(48);  // ThinkBiasCB: 2 uint + 10 float = 48 bytes (3 × vec4, D3D11 aligned)
    gravity_layer_cb_ = makeCB(16);  // GravityLayerCB: seq_len, n_head, layer_idx, gravity_scale
    if (tensor_fold_core_) {
        bone_sort_cb_    = makeCB(16); // BoneSortCB:    seq_len, n_clusters, 2 pad
        fold_dispatch_cb_= makeCB(16); // FoldDispatchCB: seq_len, n_head, cluster_start, gravity_scale
    }

    std::cerr << "[trainer] shaders compiled — GPU Adam"
              << (cfg_.use_gpu_fwd ? " + Phase-3 pipeline" : "") << " enabled\n";
    return true;
}

// ── weight loading ────────────────────────────────────────────────────────────

struct STMeta {
    std::string dtype;
    std::vector<uint64_t> shape;
    uint64_t data_start, data_end;
    uint64_t size() const { return data_end - data_start; }
    uint64_t n_floats() const {
        uint64_t n = 1; for (auto d : shape) n *= d; return n;
    }
};

static std::unordered_map<std::string, STMeta>
parse_st(const uint8_t* hdr, uint64_t hlen) {
    auto j = json::parse(std::string(reinterpret_cast<const char*>(hdr), hlen));
    std::unordered_map<std::string, STMeta> out;
    for (auto& [k, v] : j.items()) {
        if (k == "__metadata__") continue;
        STMeta m;
        m.dtype = v.value("dtype", "F32");
        if (v.contains("shape"))
            for (auto& d : v["shape"]) m.shape.push_back(d.get<uint64_t>());
        if (v.contains("data_offsets")) {
            m.data_start = v["data_offsets"][0].get<uint64_t>();
            m.data_end   = v["data_offsets"][1].get<uint64_t>();
        }
        out[k] = m;
    }
    return out;
}

bool GPT2Trainer::loadWeights(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "[trainer] cannot open: " << path << "\n"; return false; }
    size_t fsz = f.tellg(); f.seekg(0);

    uint64_t hlen = 0; f.read((char*)&hlen, 8);
    std::vector<uint8_t> hbuf(hlen); f.read((char*)hbuf.data(), hlen);
    weight_blob_.resize(fsz - 8 - hlen);
    f.read((char*)weight_blob_.data(), weight_blob_.size());
    f.close();

    // Parse header JSON directly for __metadata__ (parse_st skips it)
    {
        auto j_full = json::parse(std::string(reinterpret_cast<const char*>(hbuf.data()), hlen));
        if (j_full.contains("__metadata__")) {
            auto& jm = j_full["__metadata__"];
            // kuhul_brain_fold: {"assignments": {"<THINK>": 7, ...}, ...}
            if (jm.contains("kuhul_brain_fold")) {
                try {
                    auto fold_j = json::parse(jm["kuhul_brain_fold"].get<std::string>());
                    if (fold_j.contains("assignments")) {
                        // Map token name → open token ID
                        static const std::unordered_map<std::string, int32_t> kuhul_id_map = {
                            {"<AGENT>",50260},{"</AGENT>",50261},
                            {"<THINK>",50262},{"</THINK>",50263},
                            {"<TOOL_CALL>",50264},{"</TOOL_CALL>",50265},
                            {"<INSTRUCT>",50266},{"</INSTRUCT>",50267},
                            {"<USER>",50268},{"</USER>",50269},
                        };
                        for (auto& [tok_name, cluster_id] : fold_j["assignments"].items()) {
                            auto it = kuhul_id_map.find(tok_name);
                            if (it != kuhul_id_map.end())
                                kuhul_fold_experts_[it->second] = cluster_id.get<int32_t>();
                        }
                        std::cerr << "[trainer] kuhul fold→expert map loaded ("
                                  << kuhul_fold_experts_.size() << " entries)\n";
                    }
                } catch (...) {}
            }
        }
    }

    auto meta = parse_st(hbuf.data(), hlen);
    std::cerr << "[trainer] loading " << meta.size() << " tensors\n";

    // Auto-detect architecture from tensor shapes
    uint32_t max_layer = 0;
    for (auto& [k, m] : meta) {
        if (k.find("transformer.h.") == 0) {
            uint32_t idx = std::stoul(k.substr(14));
            max_layer = std::max(max_layer, idx + 1);
        }
    }
    model_cfg_.n_layer = max_layer > 0 ? max_layer : model_cfg_.n_layer;

    // Detect n_embd + vocab from wte/lm_head tensor size
    // Shape may be missing (empty []) in old saves; derive from byte size instead.
    auto wte_it = meta.find("transformer.wte.weight");
    if (wte_it == meta.end()) wte_it = meta.find("lm_head.weight");
    if (wte_it != meta.end()) {
        uint64_t n_floats_wte = wte_it->second.size() / sizeof(float);
        if (wte_it->second.shape.size() == 2) {
            model_cfg_.vocab_size = (uint32_t)wte_it->second.shape[0];
            model_cfg_.n_embd     = (uint32_t)wte_it->second.shape[1];
        } else {
            // Infer from size: n_embd is either 768 or 1024; vocab = n_floats / n_embd
            model_cfg_.n_embd = (n_floats_wte % 1024 == 0 && n_floats_wte / 1024 > 1000)
                                  ? 1024 : 768;
            model_cfg_.vocab_size = (uint32_t)(n_floats_wte / model_cfg_.n_embd);
        }
        model_cfg_.n_head     = model_cfg_.n_embd / 64;
        model_cfg_.d_head     = 64;
        model_cfg_.d_ff       = 4 * model_cfg_.n_embd;
        model_cfg_.attn_scale = 1.f / sqrtf(64.f);
    }

    // Build CPU weight tables — no GPU upload (CPU Adam needs no round-trip)
    params_.reserve(meta.size());
    for (auto& [k, m] : meta) {
        if (k == "__metadata__") continue;
        const bool is_f16 = (m.dtype == "F16");
        uint32_t n_floats = is_f16
            ? (uint32_t)(m.size() / sizeof(uint16_t))
            : (uint32_t)(m.size() / sizeof(float));
        if (n_floats == 0) continue;

        AdamParam p;
        p.name  = k;
        p.numel = n_floats;
        p.shape = m.shape;  // preserve tensor shape for save
        if (is_f16) {
            // F16 tensor: expand to float32 using F16C stream converter
            p.cpu_w_owned.resize(n_floats);
            const auto* half_src = reinterpret_cast<const DirectX::PackedVector::HALF*>(
                weight_blob_.data() + m.data_start);
            if (kHasAVX2) {
                DirectX::AVX2::XMConvertHalfToFloatStream(
                    p.cpu_w_owned.data(), sizeof(float),
                    half_src, sizeof(DirectX::PackedVector::HALF),
                    n_floats);
            } else {
                for (uint32_t i = 0; i < n_floats; ++i)
                    p.cpu_w_owned[i] = DirectX::PackedVector::XMConvertHalfToFloat(half_src[i]);
            }
        } else {
            const float* cpu_ptr = reinterpret_cast<const float*>(weight_blob_.data() + m.data_start);
            p.cpu_w_owned.assign(cpu_ptr, cpu_ptr + n_floats);
        }
        p.cpu_w  = p.cpu_w_owned.data();
        p.cpu_g.assign(n_floats, 0.f);

        // Upload weights to GPU; zero-init grad/moment buffers
        p.w_buf = createAndUpload(p.cpu_w_owned.data(), n_floats);
        std::vector<float> zeros(n_floats, 0.f);
        p.g_buf = createAndUpload(zeros.data(), n_floats);
        p.m_buf = createAndUpload(zeros.data(), n_floats);
        p.v_buf = createAndUpload(zeros.data(), n_floats);

        params_.push_back(std::move(p));
    }

    // Build lookup maps (after all push_backs so no reallocation invalidates anything)
    for (uint32_t i = 0; i < (uint32_t)params_.size(); ++i) {
        param_idx_[params_[i].name] = i;
        cpu_weights_[params_[i].name] = { params_[i].cpu_w, params_[i].numel };
        // Build SRV for each weight buffer (used in Phase 3 forward pass)
        param_srv_[params_[i].name] = makeSRV(params_[i].w_buf.Get(), 0, params_[i].numel);
    }

    // Free original blob — weights now live in cpu_w_owned
    weight_blob_.clear();
    weight_blob_.shrink_to_fit();

    std::cerr << "[trainer] " << params_.size() << " param tensors in CPU RAM\n";
    return true;
}

// ── random weight init (from scratch, no safetensors) ────────────────────────

bool GPT2Trainer::initWeightsFromScratch() {
    const auto& c  = model_cfg_;
    const uint32_t V  = c.vocab_size, E = c.n_embd, T = c.n_ctx;
    const uint32_t F  = c.d_ff, NL = c.n_layer;

    // GPT-2 uses std=0.02 for embeddings, std=0.02/sqrt(2*NL) for c_proj residual path
    std::mt19937 rng(42);
    auto normal = [&](float std_) {
        std::normal_distribution<float> d(0.f, std_);
        return [d, &rng]() mutable { return d(rng); };
    };
    const float emb_std  = 0.02f;
    const float res_std  = 0.02f / std::sqrt(2.f * (float)NL);  // residual path scaling
    const float init_std = 0.02f;

    auto make_param = [&](std::string name,
                          std::vector<size_t> shape,
                          uint32_t numel,
                          std::function<float()> gen,
                          bool zero = false) {
        AdamParam p;
        p.name  = std::move(name);
        p.shape = std::move(shape);
        p.numel = numel;
        p.cpu_w_owned.resize(numel);
        if (zero) std::fill(p.cpu_w_owned.begin(), p.cpu_w_owned.end(), 0.f);
        else       std::generate(p.cpu_w_owned.begin(), p.cpu_w_owned.end(), gen);
        p.cpu_w = p.cpu_w_owned.data();
        p.cpu_g.assign(numel, 0.f);
        std::vector<float> zeros(numel, 0.f);
        p.w_buf = createAndUpload(p.cpu_w_owned.data(), numel);
        p.g_buf = createAndUpload(zeros.data(), numel);
        p.m_buf = createAndUpload(zeros.data(), numel);
        p.v_buf = createAndUpload(zeros.data(), numel);
        params_.push_back(std::move(p));
    };

    auto rn_emb  = normal(emb_std);
    auto rn_init = normal(init_std);
    auto rn_res  = normal(res_std);
    auto rn_1    = []() { return 1.f; };
    auto rn_0    = []() { return 0.f; };

    // Embeddings
    make_param("transformer.wte.weight", {V, E}, V * E, rn_emb);
    make_param("transformer.wpe.weight", {T, E}, T * E, rn_emb);

    for (uint32_t l = 0; l < NL; ++l) {
        std::string lp = "transformer.h." + std::to_string(l) + ".";
        // LN1
        make_param(lp + "ln_1.weight", {E},         E,       rn_1);
        make_param(lp + "ln_1.bias",   {E},         E,       rn_0, true);
        // Attention QKV + projection
        make_param(lp + "attn.c_attn.weight", {E, 3*E}, E * 3*E, rn_init);
        make_param(lp + "attn.c_attn.bias",   {3*E},    3*E,     rn_0, true);
        make_param(lp + "attn.c_proj.weight", {E, E},   E * E,   rn_res);
        make_param(lp + "attn.c_proj.bias",   {E},      E,       rn_0, true);
        // LN2
        make_param(lp + "ln_2.weight", {E},         E,       rn_1);
        make_param(lp + "ln_2.bias",   {E},         E,       rn_0, true);
        // MLP
        make_param(lp + "mlp.c_fc.weight",   {E, F},   E * F,   rn_init);
        make_param(lp + "mlp.c_fc.bias",     {F},      F,       rn_0, true);
        make_param(lp + "mlp.c_proj.weight", {F, E},   F * E,   rn_res);
        make_param(lp + "mlp.c_proj.bias",   {E},      E,       rn_0, true);
    }

    // Final LN
    make_param("transformer.ln_f.weight", {E}, E, rn_1);
    make_param("transformer.ln_f.bias",   {E}, E, rn_0, true);

    // LM head (tied to wte — same data pointer in HF, but we keep separate for optimizer)
    make_param("lm_head.weight", {V, E}, V * E, [&](){
        // Copy wte values so tied embeddings start identical
        static uint32_t idx = 0;
        return (idx < V*E) ? params_[0].cpu_w_owned[idx++] : 0.f;
    });

    // Build index + SRV map
    for (uint32_t i = 0; i < (uint32_t)params_.size(); ++i) {
        param_idx_[params_[i].name]   = i;
        cpu_weights_[params_[i].name] = { params_[i].cpu_w, params_[i].numel };
        param_srv_[params_[i].name]   = makeSRV(params_[i].w_buf.Get(), 0, params_[i].numel);
    }

    std::cerr << "[trainer] scratch init: " << params_.size() << " params"
              << " (" << c.preset_name_str() << " shape)\n";
    return true;
}

// ── int32 buffer helper ───────────────────────────────────────────────────────

ComPtr<ID3D11Buffer> GPT2Trainer::createIntBuffer(uint32_t n_ints, bool uav) {
    auto* dev = d11_->rawDevice();
    D3D11_BUFFER_DESC d{};
    d.ByteWidth           = n_ints * sizeof(int32_t);
    d.Usage               = D3D11_USAGE_DEFAULT;
    d.BindFlags           = D3D11_BIND_SHADER_RESOURCE;
    if (uav) d.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
    d.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    d.StructureByteStride = sizeof(int32_t);
    ComPtr<ID3D11Buffer> buf;
    dev->CreateBuffer(&d, nullptr, &buf);
    return buf;
}

// ── view helpers ──────────────────────────────────────────────────────────────

ComPtr<ID3D11UnorderedAccessView>
GPT2Trainer::makeUAV(ID3D11Buffer* buf, uint32_t first, uint32_t n) {
    D3D11_UNORDERED_ACCESS_VIEW_DESC d{};
    d.Format              = DXGI_FORMAT_UNKNOWN;
    d.ViewDimension       = D3D11_UAV_DIMENSION_BUFFER;
    d.Buffer.FirstElement = first;
    d.Buffer.NumElements  = n;
    ComPtr<ID3D11UnorderedAccessView> v;
    d11_->rawDevice()->CreateUnorderedAccessView(buf, &d, &v);
    return v;
}

ComPtr<ID3D11ShaderResourceView>
GPT2Trainer::makeSRV(ID3D11Buffer* buf, uint32_t first, uint32_t n) {
    D3D11_SHADER_RESOURCE_VIEW_DESC d{};
    d.Format              = DXGI_FORMAT_UNKNOWN;
    // D3D11_SRV_DIMENSION_BUFFER correctly handles FirstElement offset for structured buffers.
    // D3D11_SRV_DIMENSION_BUFFEREX silently ignores FirstElement on Intel HD 4600 driver.
    d.ViewDimension       = D3D11_SRV_DIMENSION_BUFFER;
    d.Buffer.FirstElement = first;
    d.Buffer.NumElements  = n;
    ComPtr<ID3D11ShaderResourceView> v;
    d11_->rawDevice()->CreateShaderResourceView(buf, &d, &v);
    return v;
}

ComPtr<ID3D11ShaderResourceView>
GPT2Trainer::makeSRVi(ID3D11Buffer* buf, uint32_t first, uint32_t n) {
    // For int32 structured buffers
    D3D11_SHADER_RESOURCE_VIEW_DESC d{};
    d.Format              = DXGI_FORMAT_UNKNOWN;
    d.ViewDimension       = D3D11_SRV_DIMENSION_BUFFEREX;
    d.BufferEx.FirstElement = first;
    d.BufferEx.NumElements  = n;
    ComPtr<ID3D11ShaderResourceView> v;
    d11_->rawDevice()->CreateShaderResourceView(buf, &d, &v);
    return v;
}

void GPT2Trainer::setCB(const void* data, uint32_t bytes) {
    auto* ctx = d11_->rawCtx();
    D3D11_MAPPED_SUBRESOURCE ms{};
    ctx->Map(gen_cb_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &ms);
    std::memcpy(ms.pData, data, bytes);
    ctx->Unmap(gen_cb_.Get(), 0);
    ID3D11Buffer* cbs[] = { gen_cb_.Get() };
    ctx->CSSetConstantBuffers(0, 1, cbs);
}

void GPT2Trainer::clearViews(uint32_t nuav, uint32_t nsrv) {
    auto* ctx = d11_->rawCtx();
    static ID3D11UnorderedAccessView* nullUAV[8] = {};
    static ID3D11ShaderResourceView*  nullSRV[8] = {};
    if (nuav) ctx->CSSetUnorderedAccessViews(0, nuav, nullUAV, nullptr);
    if (nsrv) ctx->CSSetShaderResources(0, nsrv, nullSRV);
}

ID3D11ShaderResourceView* GPT2Trainer::wSRV(const std::string& name) {
    auto it = param_srv_.find(name);
    return it != param_srv_.end() ? it->second.Get() : nullptr;
}

ID3D11Buffer* GPT2Trainer::wBuf(const std::string& name) {
    auto it = param_idx_.find(name);
    return it != param_idx_.end() ? params_[it->second].g_buf.Get() : nullptr;
    // Returns g_buf intentionally — callers use wBuf for grad accumulation target
}

// ── working buffer allocation ─────────────────────────────────────────────────

bool GPT2Trainer::allocWorkingBuffers() {
    if (!cfg_.use_gpu_fwd) return true;  // Phase 1: no activation buffers needed

    const uint32_t S  = cfg_.block_size;
    const uint32_t E  = model_cfg_.n_embd;
    const uint32_t H  = model_cfg_.n_head;
    const uint32_t F  = model_cfg_.d_ff;
    const uint32_t V  = model_cfg_.vocab_size;
    const uint32_t NL = model_cfg_.n_layer;
    max_S_ = S;

    auto fb = [&](uint32_t n) { return createBuffer(n * sizeof(float), /*uav=*/true); };

    // Per-layer buffers — one D3D11 buffer per layer, each starting at offset 0.
    // This avoids the Intel HD 4600 SRV.FirstElement bug (driver ignores non-zero FirstElement).
    h_buf_.resize(NL + 1);
    for (uint32_t l = 0; l <= NL; ++l) h_buf_[l] = fb(S * E);
    qkv_buf_.resize(NL);
    xhat_ln1_buf_.resize(NL); xhat_ln2_buf_.resize(NL);
    inv_std_ln1_.resize(NL);  inv_std_ln2_.resize(NL);
    P_buf_.resize(NL);
    ln1_y_buf_.resize(NL);    ln2_y_buf_.resize(NL);
    attn_out_buf_.resize(NL); mlp_pre_buf_.resize(NL); mlp_gelu_buf_.resize(NL);
    for (uint32_t l = 0; l < NL; ++l) {
        qkv_buf_[l]      = fb(S * 3 * E);
        xhat_ln1_buf_[l] = fb(S * E);
        xhat_ln2_buf_[l] = fb(S * E);
        inv_std_ln1_[l]  = fb(S);
        inv_std_ln2_[l]  = fb(S);
        P_buf_[l]        = fb(H * S * S);
        ln1_y_buf_[l]    = fb(S * E);
        ln2_y_buf_[l]    = fb(S * E);
        attn_out_buf_[l] = fb(S * E);
        mlp_pre_buf_[l]  = fb(S * F);
        mlp_gelu_buf_[l] = fb(S * F);
    }
    xhat_lnf_buf_ = fb(S * E);
    inv_std_lnf_  = fb(S);
    lnf_y_buf_    = fb(S * E);
    logits_buf_   = fb(V);
    dlogits_buf_  = fb(V);
    loss_buf_     = fb(1);
    loss_all_     = fb(S);   // [S] per-position loss (fullseq)
    loss_staging_ = createBuffer(sizeof(float), false, /*staging=*/true);
    dh_buf_       = fb(S * E);
    d_qkv_buf_   = fb(S * 3*E);
    d_mlp_buf_   = fb(S * F);
    dP_tmp_buf_  = fb(S * S);      // per-head (processed sequentially)
    dot_row_buf_ = fb(S);          // per-head (processed sequentially)
    tokens_buf_   = createIntBuffer(S, /*uav=*/false);

    if (think_bias_ || gravity_sync_) {
        // think_depth_buf_: [S] float — geodesic arc depth ∈ [0,π], 0 outside <THINK>
        think_depth_buf_      = fb(S);
        // seq_bone_ids_buf_:     [S*4] int32 — 4 LBS bone IDs per token (innermost fold → base hash)
        // seq_bone_weights_buf_: [S*4] float — blend weights, sum=1.0 per token
        seq_bone_ids_buf_     = createIntBuffer(S * 4, /*uav=*/false);
        seq_bone_weights_buf_ = fb(S * 4);
        std::cerr << "[trainer] think-bias/gravity buffers allocated (think_depth + 4-bone LBS, S=" << S << ")\n";
    }
    if (gravity_sync_) {
        D3D11_QUERY_DESC qd{};
        qd.Query = D3D11_QUERY_EVENT;
        d11_->rawDevice()->CreateQuery(&qd, &gravity_sync_query_);
    }
    if (tensor_fold_core_) {
        sorted_token_idx_buf_ = createIntBuffer(S,               /*uav=*/true);
        cluster_start_buf_    = createIntBuffer(N_FOLD_CLUSTERS, /*uav=*/true);
        cluster_count_buf_    = createIntBuffer(N_FOLD_CLUSTERS, /*uav=*/true);
        cpu_sorted_idx_.resize(S);
        cpu_cluster_start_.assign(N_FOLD_CLUSTERS, 0);
        cpu_cluster_count_.assign(N_FOLD_CLUSTERS, 0);
        std::cerr << "[trainer] TENSOR FOLD CORE buffers allocated ("
                  << N_FOLD_CLUSTERS << " cluster slots)\n";
    }

    std::cerr << "[trainer] GPU activation buffers allocated (" << S << " tokens)\n";
    return true;
}

// ── brain expert routing ───────────────────────────────────────────────────────

bool GPT2Trainer::loadBrainExperts(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "[trainer] brain experts not found: " << path << "\n"; return false; }
    f.seekg(0, std::ios::end);
    auto sz = (size_t)f.tellg(); f.seekg(0);
    n_brain_nodes_ = (int)(sz / sizeof(int32_t));
    brain_experts_.resize(n_brain_nodes_);
    f.read(reinterpret_cast<char*>(brain_experts_.data()), sz);
    std::cerr << "[trainer] brain experts loaded: " << n_brain_nodes_
              << " nodes from " << path << "\n";
    return true;
}

// ── TENSOR FOLD CORE kernel loader ────────────────────────────────────────────
//
// Scans ../shaders/fold_kernels/ for fold_NNN.hlsl (cluster-specific overrides).
// The default COMPUTE_FOLD kernel is registered for all clusters that have no override.
// Pressure weight comes from Fold2DCompiler pressure table (COMPUTE rank-2 = 2.0).

bool GPT2Trainer::loadFoldKernels() {
    auto* dev = d11_->rawDevice();
    const std::string base    = "../shaders/";
    const std::string fk_dir  = base + "fold_kernels/";

    // Default COMPUTE_FOLD kernel — already compiled as cs_fold_kernel_compute_
    // Register it for every cluster slot (cluster-specific files can override below)
    for (int32_t c = 0; c < (int32_t)N_FOLD_CLUSTERS; ++c)
        fold_dispatch_table_[c] = { cs_fold_kernel_compute_, 2.0f, false };

    // Overlay cluster-specific kernels from fold_kernels/fold_NNN.hlsl
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((fk_dir + "fold_???.hlsl").c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            std::string fname = fd.cFileName;
            if (fname.size() >= 12 && fname.substr(0, 5) == "fold_") {
                try {
                    int32_t cid = std::stoi(fname.substr(5, 3));
                    if (cid >= 0 && cid < (int32_t)N_FOLD_CLUSTERS) {
                        auto cs = compileCS(dev, fk_dir + fname, "CSMain");
                        if (cs) {
                            fold_dispatch_table_[cid] = { cs, 2.0f, false };
                            std::cerr << "[trainer] fold kernel override: cluster " << cid
                                      << " ← " << fname << "\n";
                        }
                    }
                } catch(...) {}
            }
        } while (FindNextFileA(h, &fd));
        FindClose(h);
    }
    return !fold_dispatch_table_.empty();
}

// Build think_depth[S] + seq_expert[S] from the current token sequence.
//
// FOLD architecture:
//   FOLD  = any KUHUL structural span  <X>…</X>   (e.g. <THINK>, <AGENT>, <TOOL_CALL>)
//   NODE  = each token array position inside the fold
//
// seq_expert[t]  — FOLD-level routing (not node-level token hash):
//   INSIDE a fold  → expert cluster assigned to that fold type by kuhul_fold_experts_
//   OUTSIDE folds  → brain_experts_[tokens[t] % n_brain_nodes_]  (node-level fallback)
//   Innermost fold wins when folds nest (e.g. <THINK> inside <AGENT>).
//
// think_depth[t]:
//   Inside <THINK>…</THINK> only: arccos( dot(wte[think_open], wte[tok]) / (|a|·|b|) )
//   Other fold interiors: 0.0 (expert routing still applies, but no arc bias)
//   Outside all folds: 0.0
void GPT2Trainer::buildThinkDepth(const std::vector<int32_t>& seq,
                                   int think_token_id, int end_token_id) {
    const uint32_t S   = (uint32_t)seq.size();
    const uint32_t E   = model_cfg_.n_embd;
    const float*   wte = w("wte.weight");

    std::vector<float>   depth(S, 0.0f);
    // 4-bone LBS: interleaved layout [t0_b0,t0_b1,t0_b2,t0_b3, t1_b0,...]
    std::vector<int32_t> bone_ids(S * 4, -1);
    std::vector<float>   bone_wts(S * 4, 0.0f);

    auto is_fold_open  = [](int32_t tok) { return tok >= 50260 && tok <= 50269 && (tok & 1) == 0; };
    auto is_fold_close = [](int32_t tok) { return tok >= 50260 && tok <= 50269 && (tok & 1) == 1; };
    auto fold_open_of  = [](int32_t close_tok) { return close_tok & ~1; };

    struct FoldFrame {
        int32_t open_tok;
        int32_t expert_id;
        int     think_start;
    };
    std::vector<FoldFrame> fold_stack;

    for (uint32_t t = 0; t < S; ++t) {
        int32_t tok = seq[t];

        // ── Fold stack bookkeeping ────────────────────────────────────────────
        // For close tokens, snapshot the frame BEFORE popping (close belongs to closing fold)
        FoldFrame closing_frame{ -1, -1, -1 };
        bool is_open  = is_fold_open(tok);
        bool is_close = is_fold_close(tok);

        if (is_open) {
            int32_t fold_expert = -1;
            if (n_brain_nodes_ > 0) {
                auto it = kuhul_fold_experts_.find(tok);
                fold_expert = (it != kuhul_fold_experts_.end())
                    ? it->second
                    : brain_experts_[(uint32_t)tok % (uint32_t)n_brain_nodes_];
            }
            fold_stack.push_back({ tok, fold_expert, tok == think_token_id ? (int)t : -1 });
        }
        else if (is_close) {
            int32_t expected_open = fold_open_of(tok);
            for (int fi = (int)fold_stack.size() - 1; fi >= 0; --fi) {
                if (fold_stack[fi].open_tok == expected_open) {
                    closing_frame = fold_stack[fi];
                    fold_stack.erase(fold_stack.begin() + fi);
                    break;
                }
            }
        }

        // ── think_depth (regular nodes inside <THINK> only) ──────────────────
        if (!is_open && !is_close && wte && !fold_stack.empty()) {
            for (int fi = (int)fold_stack.size() - 1; fi >= 0; --fi) {
                if (fold_stack[fi].open_tok == think_token_id && fold_stack[fi].think_start >= 0) {
                    const float* a = wte + (uint32_t)seq[fold_stack[fi].think_start] * E;
                    const float* b = wte + (uint32_t)tok * E;
                    double dot_ab = 0.0, norm_a = 0.0, norm_b = 0.0;
                    if (kHasAVX2 && E >= 8) {
                        __m256 vdot = _mm256_setzero_ps();
                        __m256 vna  = _mm256_setzero_ps();
                        __m256 vnb  = _mm256_setzero_ps();
                        uint32_t d = 0;
                        for (; d + 8 <= E; d += 8) {
                            __m256 va = _mm256_loadu_ps(a + d);
                            __m256 vb = _mm256_loadu_ps(b + d);
                            vdot = _mm256_fmadd_ps(va, vb, vdot);
                            vna  = _mm256_fmadd_ps(va, va, vna);
                            vnb  = _mm256_fmadd_ps(vb, vb, vnb);
                        }
                        // Reduce 8-lane → scalar via 128-bit hadd chain
                        auto hsum256 = [](const __m256& v) -> float {
                            __m128 lo  = _mm256_castps256_ps128(v);
                            __m128 hi  = _mm256_extractf128_ps(v, 1);
                            __m128 s   = _mm_add_ps(lo, hi);
                            s = _mm_hadd_ps(s, s);
                            s = _mm_hadd_ps(s, s);
                            return _mm_cvtss_f32(s);
                        };
                        dot_ab = (double)hsum256(vdot);
                        norm_a = (double)hsum256(vna);
                        norm_b = (double)hsum256(vnb);
                        for (; d < E; ++d) {  // scalar tail (E=1024: never reached)
                            dot_ab += (double)a[d] * b[d];
                            norm_a += (double)a[d] * a[d];
                            norm_b += (double)b[d] * b[d];
                        }
                    } else {
                        for (uint32_t d = 0; d < E; ++d) {
                            dot_ab += (double)a[d] * b[d];
                            norm_a += (double)a[d] * a[d];
                            norm_b += (double)b[d] * b[d];
                        }
                    }
                    double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
                    depth[t] = (denom > 1e-9)
                        ? (float)std::acos(std::max(-1.0, std::min(1.0, dot_ab / denom)))
                        : 0.0f;
                    break;
                }
            }
        }

        // ── 4-bone LBS assignment ─────────────────────────────────────────────
        // Bone priority:
        //   D==0 (outside all folds):          [hash,  -1,   -1,   -1]  [1.00, 0,    0,    0   ]
        //   D==1 (1 fold):                     [fold0, hash, -1,   -1]  [0.75, 0.25, 0,    0   ]
        //   D==2 (2 nested folds):             [fold1, fold0,hash, -1]  [0.60, 0.25, 0.15, 0   ]
        //   D>=3 (3+ nested folds):            [fold2, fold1,fold0,hash] [0.50, 0.25, 0.15, 0.10]
        // Close tokens use the fold being closed as their primary bone.
        int32_t hash_expert = -1;
        if (n_brain_nodes_ > 0 && tok >= 0 && tok < 50270)
            hash_expert = brain_experts_[(uint32_t)tok % (uint32_t)n_brain_nodes_];

        int32_t* bids = bone_ids.data() + t * 4;
        float*   bwts = bone_wts.data() + t * 4;

        if (is_close) {
            // Close token: primary = fold being closed, secondary = parent fold (if any)
            bids[0] = closing_frame.expert_id;
            bids[1] = !fold_stack.empty() ? fold_stack.back().expert_id : hash_expert;
            bids[2] = hash_expert;
            bids[3] = -1;
            if (!fold_stack.empty()) {
                bwts[0] = 0.60f; bwts[1] = 0.25f; bwts[2] = 0.15f; bwts[3] = 0.0f;
            } else {
                bwts[0] = 0.75f; bwts[1] = 0.25f; bwts[2] = 0.0f;  bwts[3] = 0.0f;
                bids[1] = hash_expert; bids[2] = -1;
            }
        }
        else {
            int D = (int)fold_stack.size();
            if (D == 0) {
                bids[0] = hash_expert; bwts[0] = 1.0f;
                bids[1] = -1;          bwts[1] = 0.0f;
                bids[2] = -1;          bwts[2] = 0.0f;
                bids[3] = -1;          bwts[3] = 0.0f;
            } else if (D == 1) {
                bids[0] = fold_stack[D-1].expert_id; bwts[0] = 0.75f;
                bids[1] = hash_expert;                bwts[1] = 0.25f;
                bids[2] = -1;                         bwts[2] = 0.0f;
                bids[3] = -1;                         bwts[3] = 0.0f;
            } else if (D == 2) {
                bids[0] = fold_stack[D-1].expert_id; bwts[0] = 0.60f;
                bids[1] = fold_stack[D-2].expert_id; bwts[1] = 0.25f;
                bids[2] = hash_expert;                bwts[2] = 0.15f;
                bids[3] = -1;                         bwts[3] = 0.0f;
            } else {
                bids[0] = fold_stack[D-1].expert_id; bwts[0] = 0.50f;
                bids[1] = fold_stack[D-2].expert_id; bwts[1] = 0.25f;
                bids[2] = fold_stack[D-3].expert_id; bwts[2] = 0.15f;
                bids[3] = hash_expert;                bwts[3] = 0.10f;
            }
        }
    }

    // ── CPU cluster sort (TENSOR FOLD CORE) ──────────────────────────────────
    // Counting sort on primary bone cluster ID → linear cluster groups.
    // Matches the GPU cs_bone_argsort_ result exactly (same algorithm, CPU mirror).
    if (tensor_fold_core_) {
        std::fill(cpu_cluster_count_.begin(), cpu_cluster_count_.end(), 0u);
        for (uint32_t t = 0; t < S; ++t) {
            auto cid = (uint32_t)std::max(0, std::min((int32_t)N_FOLD_CLUSTERS - 1, bone_ids[t * 4]));
            cpu_cluster_count_[cid]++;
        }
        cpu_cluster_start_[0] = 0;
        for (uint32_t c = 1; c < N_FOLD_CLUSTERS; ++c)
            cpu_cluster_start_[c] = cpu_cluster_start_[c - 1] + cpu_cluster_count_[c - 1];
        cpu_sorted_idx_.resize(S);
        std::vector<uint32_t> scatter(cpu_cluster_start_);
        for (uint32_t t = 0; t < S; ++t) {
            auto cid = (uint32_t)std::max(0, std::min((int32_t)N_FOLD_CLUSTERS - 1, bone_ids[t * 4]));
            cpu_sorted_idx_[scatter[cid]++] = t;
        }
    }

    // ── Upload think_depth (float) ────────────────────────────────────────────
    uploadToBuffer(think_depth_buf_.Get(), depth.data(), S);

    // ── Upload bone_ids (int32) via staging ───────────────────────────────────
    {
        auto* ctx = d11_->rawCtx();
        auto* dev = d11_->rawDevice();
        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth            = S * 4 * sizeof(int32_t);
        desc.Usage                = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags       = D3D11_CPU_ACCESS_WRITE;
        desc.MiscFlags            = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride  = sizeof(int32_t);
        ComPtr<ID3D11Buffer> staging;
        dev->CreateBuffer(&desc, nullptr, &staging);
        if (staging) {
            D3D11_MAPPED_SUBRESOURCE ms{};
            if (SUCCEEDED(ctx->Map(staging.Get(), 0, D3D11_MAP_WRITE, 0, &ms)) && ms.pData) {
                std::memcpy(ms.pData, bone_ids.data(), S * 4 * sizeof(int32_t));
                ctx->Unmap(staging.Get(), 0);
                ctx->CopyResource(seq_bone_ids_buf_.Get(), staging.Get());
            }
        }
    }

    // ── Upload bone_weights (float) ───────────────────────────────────────────
    uploadToBuffer(seq_bone_weights_buf_.Get(), bone_wts.data(), S * 4);
}

// ── init ──────────────────────────────────────────────────────────────────────

bool GPT2Trainer::init(const TrainerConfig& cfg) {
    cfg_ = cfg;
    if (const char* w = std::getenv("GPT2_WARMUP")) warmup_steps_ = std::atoi(w);
    if (const char* c = std::getenv("GPT2_CLIP")) grad_clip_ = (float)std::atof(c);
    adaptive_clip_ = std::getenv("GPT2_ADAPTIVE_CLIP") != nullptr;
    if (adaptive_clip_) { physics_ = new KuhulPhysicsSolver(0.5f, 5.0f);
        std::cerr << "[trainer] adaptive grad-clip (KuhulPhysics) enabled\n"; }
    else std::cerr << "[trainer] grad-clip fixed = " << grad_clip_ << "\n";
    fullseq_ = std::getenv("GPT2_FULLSEQ") != nullptr;
    if (!loadShaders()) return false;

    if (!cfg_.preset_name.empty()) {
        model_cfg_ = GPT2Config::from_preset(cfg_.preset_name);
        if (!initWeightsFromScratch()) return false;
    } else {
        if (!loadWeights(cfg_.model_path)) return false;
    }

    if (!allocWorkingBuffers()) return false;

    // Brain expert routing: load brain2/experts.bin if think_bias_ is active
    if (think_bias_) {
        const char* bpath = std::getenv("GPT2_BRAIN_EXPERTS");
        std::string experts_path = bpath ? bpath : "../../brain2/experts.bin";
        loadBrainExperts(experts_path);  // non-fatal: brain_experts_ stays empty on failure
    }
    std::cerr << "[trainer] init complete — " << params_.size() << " params"
              << ", n_layer=" << model_cfg_.n_layer
              << ", n_embd="  << model_cfg_.n_embd
              << ", n_head="  << model_cfg_.n_head << "\n";
    return true;
}

// ── weight accessor helper ─────────────────────────────────────────────────────

const float* GPT2Trainer::w(const std::string& name) const {
    auto it = cpu_weights_.find(name);
    if (it == cpu_weights_.end()) return nullptr;
    return it->second.first;
}

// Build mask: 1.0 for positions inside <INSTRUCT>...</INSTRUCT> spans, 0.0 for <USER> spans.
// Returns all-ones when completion_only is false (no masking).
static std::vector<float> buildCompletionMask(const std::vector<int32_t>& seq, bool completion_only) {
    std::vector<float> mask(seq.size(), 1.0f);
    if (!completion_only) return mask;
    bool in_user     = false;
    bool in_instruct = false;
    for (size_t t = 0; t < seq.size(); ++t) {
        int32_t tok = seq[t];
        if      (tok == 50268) { in_user = true;     in_instruct = false; } // <USER>
        else if (tok == 50269) { in_user = false; }                          // </USER>
        else if (tok == 50266) { in_instruct = true;  in_user = false; }    // <INSTRUCT>
        else if (tok == 50267) { in_instruct = false; }                      // </INSTRUCT>
        if (tok == 50268 || tok == 50269 || tok == 50266 || tok == 50267) {
            mask[t] = 0.0f;
            continue;
        }
        mask[t] = in_instruct ? 1.0f : 0.0f;
    }
    return mask;
}

// ── CPU forward+backward for one sequence ─────────────────────────────────────

float GPT2Trainer::forwardBackwardCPU(const std::vector<int32_t>& seq,
                                       bool accumulate_grads) {
    const uint32_t S  = (uint32_t)seq.size();
    const uint32_t E  = model_cfg_.n_embd;
    const uint32_t H  = model_cfg_.n_head;
    const uint32_t D  = model_cfg_.d_head;   // 64
    const uint32_t F  = model_cfg_.d_ff;     // 4*E
    const uint32_t V  = model_cfg_.vocab_size;
    const uint32_t NL = model_cfg_.n_layer;
    const float    eps = 1e-5f;
    const float    scale = model_cfg_.attn_scale;

    // ── activations (forward) ────────────────────────────────────────────────
    std::vector<float> hidden(S*E);
    // Per-layer saved activations (for backward)
    struct LayerActs {
        std::vector<float> h_in;        // hidden before this layer [S*E]
        std::vector<float> ln1_y, ln1_xhat; float ln1_inv_std[1024]; float ln1_mean[1024];
        std::vector<float> qkv;         // [S, 3E]
        std::vector<float> attn_out;    // [S, E] — post attention + c_proj
        std::vector<float> h_mid;       // hidden after attn residual [S*E]
        std::vector<float> ln2_y, ln2_xhat; float ln2_inv_std[1024]; float ln2_mean[1024];
        std::vector<float> mlp_pre;     // pre-GELU [S*F]
        std::vector<float> mlp_gelu;    // post-GELU [S*F]
        std::vector<float> mlp_out;     // [S*E]
        // Per-head attention softmax weights P [H][S*S]
        std::vector<std::vector<float>> P;
        // Per-head Q, K, V slices [H][S*D]
        std::vector<std::vector<float>> Qh, Kh, Vh, Ah;
    };
    std::vector<LayerActs> acts(NL);
    for (auto& a : acts) {
        a.ln1_y.resize(S*E); a.ln1_xhat.resize(S*E);
        a.ln2_y.resize(S*E); a.ln2_xhat.resize(S*E);
        a.qkv.resize(S*3*E);
        a.attn_out.resize(S*E);
        a.h_in.resize(S*E); a.h_mid.resize(S*E);
        a.mlp_pre.resize(S*F); a.mlp_gelu.resize(S*F); a.mlp_out.resize(S*E);
        a.P.resize(H, std::vector<float>(S*S, 0.f));
        a.Qh.resize(H, std::vector<float>(S*D, 0.f));
        a.Kh.resize(H, std::vector<float>(S*D, 0.f));
        a.Vh.resize(H, std::vector<float>(S*D, 0.f));
        a.Ah.resize(H, std::vector<float>(S*D, 0.f));
    }
    std::vector<float> ln_f_y(S*E), ln_f_xhat(S*E);
    float ln_f_inv_std[1024], ln_f_mean[1024];

    // ── Embedding ────────────────────────────────────────────────────────────
    // wte may be stored as lm_head.weight (tied embedding)
    const float* wte = w("transformer.wte.weight");
    if (!wte) wte = w("lm_head.weight");
    const float* wpe = w("transformer.wpe.weight");
    if (!wte || !wpe) { std::cerr << "[trainer] missing embeddings\n"; return -1.f; }
    // Guard: token ids must be < vocab_size, else the wte read/write below is a
    // heap OOB (SIGSEGV on the CPU path). A data/model vocab mismatch triggers it.
    static bool warned_oob_tok = false;
    for (uint32_t i = 0; i < S; ++i) {
        if ((uint32_t)seq[i] >= V) {
            if (!warned_oob_tok) {
                std::cerr << "[trainer] WARNING: token id " << seq[i]
                          << " >= vocab_size " << V
                          << " (data/model vocab mismatch) — clamping reads, skipping OOB grads\n";
                warned_oob_tok = true;
            }
        }
    }
    for (uint32_t i = 0; i < S; ++i) {
        const uint32_t tok = std::min((uint32_t)seq[i], V - 1);
        for (uint32_t d = 0; d < E; ++d)
            hidden[i*E+d] = wte[tok*E+d] + wpe[i*E+d];
    }

    // ── Transformer layers ───────────────────────────────────────────────────
    for (uint32_t l = 0; l < NL; ++l) {
        auto& a = acts[l];
        a.h_in = hidden;
        const std::string pfx = "transformer.h." + std::to_string(l) + ".";

        const float* ln1_g = w(pfx+"ln_1.weight"), *ln1_b = w(pfx+"ln_1.bias");
        const float* ln2_g = w(pfx+"ln_2.weight"), *ln2_b = w(pfx+"ln_2.bias");
        const float* cattn_w = w(pfx+"attn.c_attn.weight"); // [E, 3E]
        const float* cattn_b = w(pfx+"attn.c_attn.bias");   // [3E]
        const float* cproj_w = w(pfx+"attn.c_proj.weight"); // [E, E]
        const float* cproj_b = w(pfx+"attn.c_proj.bias");   // [E]
        const float* fc_w    = w(pfx+"mlp.c_fc.weight");    // [E, F]
        const float* fc_b    = w(pfx+"mlp.c_fc.bias");      // [F]
        const float* fc2_w   = w(pfx+"mlp.c_proj.weight");  // [F, E]
        const float* fc2_b   = w(pfx+"mlp.c_proj.bias");    // [E]

        // LayerNorm 1
        for (uint32_t i = 0; i < S; ++i)
            layernorm_fwd(&hidden[i*E], ln1_g, ln1_b, E, eps,
                          &a.ln1_y[i*E], &a.ln1_xhat[i*E],
                          a.ln1_mean[i], a.ln1_inv_std[i]);

        // QKV projection: [S,E] @ [E,3E] + [3E] → [S,3E]
        matmul(a.ln1_y.data(), cattn_w, a.qkv.data(), S, E, 3*E);
        for (uint32_t i = 0; i < S; ++i)
            for (uint32_t d = 0; d < 3*E; ++d) a.qkv[i*3*E+d] += cattn_b[d];

        // Split QKV into per-head slices and run attention
        std::fill(a.attn_out.begin(), a.attn_out.end(), 0.f);
        for (uint32_t h = 0; h < H; ++h) {
            // Extract Q,K,V for head h
            for (uint32_t i = 0; i < S; ++i) {
                for (uint32_t d = 0; d < D; ++d) {
                    a.Qh[h][i*D+d] = a.qkv[i*3*E + h*D + d];
                    a.Kh[h][i*D+d] = a.qkv[i*3*E + E + h*D + d];
                    a.Vh[h][i*D+d] = a.qkv[i*3*E + 2*E + h*D + d];
                }
            }
            // Causal attention
            const float* Q = a.Qh[h].data(), *K = a.Kh[h].data(), *V = a.Vh[h].data();
            float* Ah = a.Ah[h].data();
            float* P  = a.P[h].data();
            for (uint32_t i = 0; i < S; ++i) {
                // scores
                float mx = -1e30f;
                for (uint32_t j = 0; j <= i; ++j) {
                    float dot = 0.f;
                    for (uint32_t d = 0; d < D; ++d) dot += Q[i*D+d]*K[j*D+d];
                    P[i*S+j] = dot * scale;
                    mx = std::max(mx, P[i*S+j]);
                }
                for (uint32_t j = i+1; j < S; ++j) P[i*S+j] = -1e30f;
                float sum = 0.f;
                for (uint32_t j = 0; j <= i; ++j) { P[i*S+j] = expf(P[i*S+j]-mx); sum += P[i*S+j]; }
                for (uint32_t j = 0; j <= i; ++j) P[i*S+j] /= sum;
                for (uint32_t j = i+1; j < S; ++j) P[i*S+j] = 0.f;
                // weighted V
                for (uint32_t d = 0; d < D; ++d) {
                    float acc = 0.f;
                    for (uint32_t j = 0; j < S; ++j) acc += P[i*S+j]*V[j*D+d];
                    Ah[i*D+d] = acc;
                }
            }
            // Scatter head output into attn_concat
            for (uint32_t i = 0; i < S; ++i)
                for (uint32_t d = 0; d < D; ++d)
                    a.attn_out[i*E + h*D + d] = Ah[i*D+d];  // concat
        }
        // c_proj: [S,E] @ [E,E] + [E] → [S,E]
        std::vector<float> attn_proj(S*E, 0.f);
        matmul(a.attn_out.data(), cproj_w, attn_proj.data(), S, E, E);
        for (uint32_t i = 0; i < S; ++i)
            for (uint32_t d = 0; d < E; ++d) {
                attn_proj[i*E+d] += cproj_b[d];
                hidden[i*E+d] += attn_proj[i*E+d];  // residual
            }
        a.h_mid = hidden;

        // LayerNorm 2
        for (uint32_t i = 0; i < S; ++i)
            layernorm_fwd(&hidden[i*E], ln2_g, ln2_b, E, eps,
                          &a.ln2_y[i*E], &a.ln2_xhat[i*E],
                          a.ln2_mean[i], a.ln2_inv_std[i]);

        // MLP c_fc: [S,E]@[E,F]+[F] → [S,F], then GELU
        matmul(a.ln2_y.data(), fc_w, a.mlp_pre.data(), S, E, F);
        for (uint32_t i = 0; i < S; ++i)
            for (uint32_t d = 0; d < F; ++d) {
                a.mlp_pre[i*F+d] += fc_b[d];
                a.mlp_gelu[i*F+d] = gelu_fwd(a.mlp_pre[i*F+d]);
            }

        // MLP c_proj: [S,F]@[F,E]+[E] → [S,E]
        matmul(a.mlp_gelu.data(), fc2_w, a.mlp_out.data(), S, F, E);
        for (uint32_t i = 0; i < S; ++i)
            for (uint32_t d = 0; d < E; ++d) {
                a.mlp_out[i*E+d] += fc2_b[d];
                hidden[i*E+d] += a.mlp_out[i*E+d];  // residual
            }
    }

    // Final LayerNorm
    const float* ln_f_g = w("transformer.ln_f.weight");
    const float* ln_f_b = w("transformer.ln_f.bias");
    for (uint32_t i = 0; i < S; ++i)
        layernorm_fwd(&hidden[i*E], ln_f_g, ln_f_b, E, eps,
                      &ln_f_y[i*E], &ln_f_xhat[i*E],
                      ln_f_mean[i], ln_f_inv_std[i]);

    // LM Head: logits = hidden[S-2] @ wte.T  predicts token at position S-1 (next-token target).
    // Use S-2 so the hidden state has NOT seen token[S-1] yet (causal mask: h[t] sees tokens 0..t).
    // Using S-1 would let the model read its own input token → trivial loss≈0.
    if (S < 2) return 0.f;
    const uint32_t last = S - 2;
    std::vector<float> logits(V, 0.f);
    for (uint32_t v = 0; v < V; ++v) {
        float dot = 0.f;
        for (uint32_t d = 0; d < E; ++d) dot += ln_f_y[last*E+d] * wte[v*E+d];
        logits[v] = dot;
    }

    // ── Backward pass ────────────────────────────────────────────────────────

    float total_loss = 0.f;
    int32_t target = -1;
    if (S >= 2) target = seq[S-1];
    if (target >= (int)V) target = (int)V - 1;   // guard OOB target (vocab mismatch)

    if (target < 0 || !accumulate_grads) return total_loss;

    // completion_only: skip gradient if the target position is not inside an INSTRUCT span
    if (cfg_.completion_only) {
        auto mask = buildCompletionMask(seq, true);
        uint32_t target_pos = S - 1;  // position we're predicting
        if (target_pos < mask.size() && mask[target_pos] < 0.5f)
            return 0.f;
    }

    // Stable softmax for loss + lm_head grad
    {
        float mx = *std::max_element(logits.begin(), logits.end());
        std::vector<float> probs(V);
        float sum_exp = 0.f;
        for (uint32_t v = 0; v < V; ++v) { probs[v] = expf(logits[v]-mx); sum_exp += probs[v]; }
        for (uint32_t v = 0; v < V; ++v) probs[v] /= sum_exp;

        total_loss = -logf(probs[target] + 1e-10f);

        std::vector<float> dlogits(V);
        for (uint32_t v = 0; v < V; ++v) dlogits[v] = probs[v];
        dlogits[target] -= 1.f;

        // lm_head weight gradient
        auto it = param_idx_.find("transformer.wte.weight");
        if (it == param_idx_.end()) it = param_idx_.find("lm_head.weight");
        if (it != param_idx_.end()) {
            auto& p = params_[it->second];
            for (uint32_t v = 0; v < V; ++v)
                for (uint32_t d = 0; d < E; ++d)
                    p.cpu_g[v*E+d] += dlogits[v] * ln_f_y[last*E+d];
        }

        // dh[last,:] = wte.T @ dlogits; other positions start at 0
        // (full S*E tensor — attention backward will spread grads to earlier positions)
        // dh allocated below
    }

    // Full gradient tensor [S*E]; fill dh[last,:] from lm_head grad
    std::vector<float> dh(S*E, 0.f);
    {
        float mx = *std::max_element(logits.begin(), logits.end());
        std::vector<float> probs(V);
        float sum_exp = 0.f;
        for (uint32_t v = 0; v < V; ++v) { probs[v] = expf(logits[v]-mx); sum_exp += probs[v]; }
        for (uint32_t v = 0; v < V; ++v) probs[v] /= sum_exp;
        probs[target] -= 1.f;  // dlogits = probs - onehot
        for (uint32_t d = 0; d < E; ++d) {
            float acc = 0.f;
            for (uint32_t v = 0; v < V; ++v) acc += wte[v*E+d] * probs[v];
            dh[last*E+d] = acc;
        }
    }

    // Final LayerNorm backward (all positions; only last has non-zero upstream)
    {
        auto& p_lng = params_[param_idx_["transformer.ln_f.weight"]];
        auto& p_lnb = params_[param_idx_["transformer.ln_f.bias"]];
        std::vector<float> new_dh(S*E, 0.f);
        for (uint32_t s = 0; s < S; ++s) {
            layernorm_bwd(&dh[s*E], &ln_f_xhat[s*E], ln_f_g,
                          E, ln_f_inv_std[s],
                          &new_dh[s*E], p_lng.cpu_g.data(), p_lnb.cpu_g.data());
        }
        dh = std::move(new_dh);
    }

    // ── Layer loop: NL-1 down to 0 ───────────────────────────────────────────
    for (int l = (int)NL - 1; l >= 0; --l) {
        auto& a = acts[(uint32_t)l];
        const std::string pfx = "transformer.h." + std::to_string(l) + ".";

        auto& p_ln2_g   = params_[param_idx_[pfx+"ln_2.weight"]];
        auto& p_ln2_b   = params_[param_idx_[pfx+"ln_2.bias"]];
        auto& p_fc_w    = params_[param_idx_[pfx+"mlp.c_fc.weight"]];
        auto& p_fc_b    = params_[param_idx_[pfx+"mlp.c_fc.bias"]];
        auto& p_fc2_w   = params_[param_idx_[pfx+"mlp.c_proj.weight"]];
        auto& p_fc2_b   = params_[param_idx_[pfx+"mlp.c_proj.bias"]];
        auto& p_ln1_g   = params_[param_idx_[pfx+"ln_1.weight"]];
        auto& p_ln1_b   = params_[param_idx_[pfx+"ln_1.bias"]];
        auto& p_cattn_w = params_[param_idx_[pfx+"attn.c_attn.weight"]];
        auto& p_cattn_b = params_[param_idx_[pfx+"attn.c_attn.bias"]];
        auto& p_cproj_w = params_[param_idx_[pfx+"attn.c_proj.weight"]];
        auto& p_cproj_b = params_[param_idx_[pfx+"attn.c_proj.bias"]];

        const float* fc2_w_p   = w(pfx+"mlp.c_proj.weight");
        const float* fc_w_p    = w(pfx+"mlp.c_fc.weight");
        const float* cproj_w_p = w(pfx+"attn.c_proj.weight");
        const float* cattn_w_p = w(pfx+"attn.c_attn.weight");

        // ── MLP backward ─────────────────────────────────────────────────────
        // h_out = h_mid + mlp_out  →  d_mlp_out = dh,  d_h_mid_residual = dh
        std::vector<float> d_mlp_out(S*E);
        std::memcpy(d_mlp_out.data(), dh.data(), S*E*sizeof(float));

        // c_proj: mlp_gelu[S,F] @ fc2_w[F,E] = mlp_out[S,E]
        std::vector<float> d_mlp_gelu(S*F, 0.f);
        matmul_bwd(d_mlp_out.data(), a.mlp_gelu.data(), fc2_w_p,
                   d_mlp_gelu.data(), p_fc2_w.cpu_g.data(), S, F, E);
        bias_bwd(d_mlp_out.data(), p_fc2_b.cpu_g.data(), S, E);

        // GELU
        std::vector<float> d_mlp_pre(S*F);
        for (uint32_t s = 0; s < S; ++s)
            for (uint32_t f = 0; f < F; ++f)
                d_mlp_pre[s*F+f] = d_mlp_gelu[s*F+f] * gelu_bwd(a.mlp_pre[s*F+f]);

        // c_fc: ln2_y[S,E] @ fc_w[E,F] = mlp_pre[S,F]
        std::vector<float> d_ln2_y(S*E, 0.f);
        matmul_bwd(d_mlp_pre.data(), a.ln2_y.data(), fc_w_p,
                   d_ln2_y.data(), p_fc_w.cpu_g.data(), S, E, F);
        bias_bwd(d_mlp_pre.data(), p_fc_b.cpu_g.data(), S, F);

        // LN2 backward per position → d_h_mid
        std::vector<float> d_h_mid(S*E, 0.f);
        for (uint32_t s = 0; s < S; ++s) {
            layernorm_bwd(&d_ln2_y[s*E], &a.ln2_xhat[s*E], w(pfx+"ln_2.weight"),
                          E, a.ln2_inv_std[s],
                          &d_h_mid[s*E], p_ln2_g.cpu_g.data(), p_ln2_b.cpu_g.data());
        }
        // MLP residual
        for (uint32_t i = 0; i < S*E; ++i) d_h_mid[i] += dh[i];

        // ── Attention backward ────────────────────────────────────────────────
        // h_mid = h_in + attn_proj  →  d_attn_proj = d_h_mid,  d_h_in_residual = d_h_mid
        std::vector<float> d_attn_proj(S*E);
        std::memcpy(d_attn_proj.data(), d_h_mid.data(), S*E*sizeof(float));

        // c_proj: attn_out[S,E] @ cproj_w[E,E] = attn_proj[S,E]
        std::vector<float> d_attn_concat(S*E, 0.f);
        matmul_bwd(d_attn_proj.data(), a.attn_out.data(), cproj_w_p,
                   d_attn_concat.data(), p_cproj_w.cpu_g.data(), S, E, E);
        bias_bwd(d_attn_proj.data(), p_cproj_b.cpu_g.data(), S, E);

        // Per-head attention backward → d_qkv[S,3E]
        std::vector<float> d_qkv(S*3*E, 0.f);
        for (uint32_t h = 0; h < H; ++h) {
            const float* Qh = a.Qh[h].data();
            const float* Kh = a.Kh[h].data();
            const float* Vh = a.Vh[h].data();
            const float* Ph = a.P[h].data();

            // Gather d_Ah from concat gradient
            std::vector<float> d_Ah(S*D, 0.f);
            for (uint32_t s = 0; s < S; ++s)
                for (uint32_t d = 0; d < D; ++d)
                    d_Ah[s*D+d] = d_attn_concat[s*E + h*D + d];

            // d_V[j,d] = sum_{i>=j} P[i,j] * d_Ah[i,d]
            std::vector<float> d_Vh(S*D, 0.f);
            for (uint32_t i = 0; i < S; ++i)
                for (uint32_t j = 0; j <= i; ++j) {
                    const float pij = Ph[i*S+j];
                    for (uint32_t d = 0; d < D; ++d)
                        d_Vh[j*D+d] += pij * d_Ah[i*D+d];
                }

            // Softmax backward → d_Q, d_K
            std::vector<float> d_Qh(S*D, 0.f), d_Kh(S*D, 0.f);
            for (uint32_t i = 0; i < S; ++i) {
                // d_P[i,j] = d_Ah[i,:] @ Vh[j,:]  and softmax backward
                float dot_i = 0.f;
                std::vector<float> d_P_row(i+1);
                for (uint32_t j = 0; j <= i; ++j) {
                    float dp = 0.f;
                    for (uint32_t d = 0; d < D; ++d) dp += d_Ah[i*D+d] * Vh[j*D+d];
                    d_P_row[j] = dp;
                    dot_i += Ph[i*S+j] * dp;
                }
                // d_score[i,j] = scale * P[i,j] * (d_P[i,j] - dot_i)
                for (uint32_t j = 0; j <= i; ++j) {
                    const float ds = scale * Ph[i*S+j] * (d_P_row[j] - dot_i);
                    for (uint32_t d = 0; d < D; ++d) d_Qh[i*D+d] += ds * Kh[j*D+d];
                    for (uint32_t d = 0; d < D; ++d) d_Kh[j*D+d] += ds * Qh[i*D+d];
                }
            }

            // Scatter Q,K,V grads into d_qkv
            for (uint32_t s = 0; s < S; ++s)
                for (uint32_t d = 0; d < D; ++d) {
                    d_qkv[s*3*E + h*D + d]       += d_Qh[s*D+d];
                    d_qkv[s*3*E + E + h*D + d]   += d_Kh[s*D+d];
                    d_qkv[s*3*E + 2*E + h*D + d] += d_Vh[s*D+d];
                }
        }

        // c_attn bias backward
        bias_bwd(d_qkv.data(), p_cattn_b.cpu_g.data(), S, 3*E);

        // c_attn: ln1_y[S,E] @ cattn_w[E,3E] = qkv[S,3E]
        std::vector<float> d_ln1_y(S*E, 0.f);
        matmul_bwd(d_qkv.data(), a.ln1_y.data(), cattn_w_p,
                   d_ln1_y.data(), p_cattn_w.cpu_g.data(), S, E, 3*E);

        // LN1 backward per position → d_h_in (attn path only)
        std::vector<float> d_h_in(S*E, 0.f);
        for (uint32_t s = 0; s < S; ++s) {
            layernorm_bwd(&d_ln1_y[s*E], &a.ln1_xhat[s*E], w(pfx+"ln_1.weight"),
                          E, a.ln1_inv_std[s],
                          &d_h_in[s*E], p_ln1_g.cpu_g.data(), p_ln1_b.cpu_g.data());
        }

        // New dh: residual (d_h_mid) + attn path (d_h_in)
        for (uint32_t i = 0; i < S*E; ++i) dh[i] = d_h_mid[i] + d_h_in[i];
    }

    // ── Embedding gradients ───────────────────────────────────────────────────
    {
        auto it_wpe = param_idx_.find("transformer.wpe.weight");
        auto it_wte = param_idx_.find("transformer.wte.weight");
        if (it_wte == param_idx_.end()) it_wte = param_idx_.find("lm_head.weight");

        if (it_wpe != param_idx_.end()) {
            auto& p_wpe = params_[it_wpe->second];
            for (uint32_t s = 0; s < S; ++s)
                for (uint32_t d = 0; d < E; ++d)
                    p_wpe.cpu_g[s*E+d] += dh[s*E+d];
        }
        if (it_wte != param_idx_.end()) {
            auto& p_wte = params_[it_wte->second];
            for (uint32_t s = 0; s < S; ++s) {
                if ((uint32_t)seq[s] >= V) continue;   // guard OOB token (heap-safe)
                const uint64_t base = (uint64_t)(uint32_t)seq[s] * E;
                if (base + E > p_wte.cpu_g.size()) continue;  // guard: param smaller than V*E
                for (uint32_t d = 0; d < E; ++d)
                    p_wte.cpu_g[base + d] += dh[s*E+d];
            }
        }
    }

    return total_loss;
}

// ── zero CPU gradients ─────────────────────────────────────────────────────────

void GPT2Trainer::zeroCPUGrads() {
    for (auto& p : params_)
        std::fill(p.cpu_g.begin(), p.cpu_g.end(), 0.f);
}

// ── zero GPU gradient buffers (Phase 3) ───────────────────────────────────────

void GPT2Trainer::zeroGPUGrads() {
    auto* ctx = d11_->rawCtx();
    auto* dev = d11_->rawDevice();
    static const float zero = 0.f;
    for (auto& p : params_) {
        // Fill g_buf with zeros via a 1-element staging upload repeated via ClearUnorderedAccessViewFloat
        UINT val[4] = {0,0,0,0};
        auto uav = makeUAV(p.g_buf.Get(), 0, p.numel);
        ctx->ClearUnorderedAccessViewUint(uav.Get(), val);
    }
    // Also zero dh and working grad buffers
    auto zeroF = [&](ID3D11Buffer* buf, uint32_t n) {
        auto uav = makeUAV(buf, 0, n);
        UINT val[4] = {};
        ctx->ClearUnorderedAccessViewUint(uav.Get(), val);
    };
    const uint32_t S=max_S_, E=model_cfg_.n_embd, F=model_cfg_.d_ff;
    zeroF(dh_buf_.Get(),      S*E);
    zeroF(d_qkv_buf_.Get(),  S*3*E);
    zeroF(d_mlp_buf_.Get(),  S*F);
    zeroF(dP_tmp_buf_.Get(), S*S);
    zeroF(dot_row_buf_.Get(),S);
    zeroF(logits_buf_.Get(), model_cfg_.vocab_size);
}

float GPT2Trainer::readbackLoss() {
    auto* ctx = d11_->rawCtx();
    auto* dev = d11_->rawDevice();
    // Use a structured staging buffer to ensure CopyResource compatibility
    D3D11_BUFFER_DESC sd{};
    sd.ByteWidth = sizeof(float);
    sd.Usage = D3D11_USAGE_STAGING;
    sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    sd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    sd.StructureByteStride = sizeof(float);
    ComPtr<ID3D11Buffer> st;
    dev->CreateBuffer(&sd, nullptr, &st);
    if (!st) return 0.f;
    ctx->CopyResource(st.Get(), loss_buf_.Get());
    D3D11_MAPPED_SUBRESOURCE ms{};
    ctx->Map(st.Get(), 0, D3D11_MAP_READ, 0, &ms);
    float v = ms.pData ? *static_cast<const float*>(ms.pData) : 0.f;
    ctx->Unmap(st.Get(), 0);
    return v;
}

// ── GPU forward+backward (Phase 3) ────────────────────────────────────────────

// Helper: get weight (w_buf) SRV — distinct from wSRV which is cached at init
// wSRV() already does this; just use wSRV() everywhere.

float GPT2Trainer::gpuForwardBackward(const std::vector<int32_t>& seq, float inv_batch) {
    auto* ctx = d11_->rawCtx();
    auto* dev = d11_->rawDevice();
    const uint32_t S  = (uint32_t)seq.size();
    const uint32_t E  = model_cfg_.n_embd;
    const uint32_t H  = model_cfg_.n_head;
    const uint32_t D  = model_cfg_.d_head;
    const uint32_t F  = model_cfg_.d_ff;
    const uint32_t V  = model_cfg_.vocab_size;
    const uint32_t NL = model_cfg_.n_layer;
    const float    eps   = 1e-5f;
    const float    scale = model_cfg_.attn_scale;

    // Helper: zero a structured buffer range via ClearUnorderedAccessViewUint
    auto zero = [&](ID3D11Buffer* b, uint32_t first, uint32_t n) {
        auto u = makeUAV(b, first, n);
        UINT z[4] = {};
        ctx->ClearUnorderedAccessViewUint(u.Get(), z);
    };

    // ── Upload tokens ─────────────────────────────────────────────────────────
    {
        D3D11_BUFFER_DESC sd{};
        sd.ByteWidth = S * sizeof(int32_t);
        sd.Usage = D3D11_USAGE_STAGING;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        sd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        sd.StructureByteStride = sizeof(int32_t);
        ComPtr<ID3D11Buffer> st; dev->CreateBuffer(&sd, nullptr, &st);
        if (st) {
            D3D11_MAPPED_SUBRESOURCE ms{};
            ctx->Map(st.Get(), 0, D3D11_MAP_WRITE, 0, &ms);
            if (ms.pData) std::memcpy(ms.pData, seq.data(), S * sizeof(int32_t));
            ctx->Unmap(st.Get(), 0);
            ctx->CopyResource(tokens_buf_.Get(), st.Get());
        }
    }

    // Build think_depth + seq_bone once per sequence (before the layer loop)
    if (think_bias_ || gravity_sync_ || tensor_fold_core_)
        buildThinkDepth(seq, /*think_token_id=*/50262, /*end_token_id=*/50263);

    // GPU argsort: build XVM linear cluster layout (once per sequence, reused every layer)
    if (tensor_fold_core_) {
        struct { uint32_t seq_len, n_clusters, pad0, pad1; } bc{ S, N_FOLD_CLUSTERS, 0, 0 };
        D3D11_MAPPED_SUBRESOURCE ms{};
        ctx->Map(bone_sort_cb_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &ms);
        if (ms.pData) std::memcpy(ms.pData, &bc, sizeof(bc));
        ctx->Unmap(bone_sort_cb_.Get(), 0);
        ID3D11Buffer* bscb = bone_sort_cb_.Get();
        ctx->CSSetConstantBuffers(0, 1, &bscb);
        ctx->CSSetShader(cs_bone_argsort_.Get(), nullptr, 0);
        auto sv0 = makeSRV(seq_bone_ids_buf_.Get(), 0, S * 4);  // bone_ids as SRV
        ID3D11ShaderResourceView*  srvsA[1] = { sv0.Get() };
        ctx->CSSetShaderResources(0, 1, srvsA);
        auto uv0 = makeUAV(sorted_token_idx_buf_.Get(), 0, S);
        auto uv1 = makeUAV(cluster_start_buf_.Get(),    0, N_FOLD_CLUSTERS);
        auto uv2 = makeUAV(cluster_count_buf_.Get(),    0, N_FOLD_CLUSTERS);
        ID3D11UnorderedAccessView* uvsA[3] = { uv0.Get(), uv1.Get(), uv2.Get() };
        ctx->CSSetUnorderedAccessViews(0, 3, uvsA, nullptr);
        ctx->Dispatch(1, 1, 1);
        clearViews(3, 1);
    }

    // ══════════════════════════ FORWARD PASS ═════════════════════════════════

    // 1. Embed: wte[tokens] + wpe → h_buf_[0, S, E]
    {
        struct { uint32_t S, E, pad[2]; } p{S, E};
        setCB(&p, 16);
        ctx->CSSetShader(cs_embed_fwd_.Get(), nullptr, 0);
        auto tsrv = makeSRVi(tokens_buf_.Get(), 0, S);
        ID3D11ShaderResourceView* srvs[3] = {
            tsrv.Get(), wSRV("transformer.wte.weight"), wSRV("transformer.wpe.weight")
        };
        ctx->CSSetShaderResources(0, 3, srvs);
        zero(h_buf_[0].Get(), 0, S * E);
        auto uv0 = makeUAV(h_buf_[0].Get(), 0, S * E);
        ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
        ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
        ctx->Dispatch(S, 1, 1); clearViews(1, 3);
    }

    // 2. Layer loop
    // Note: all per-layer buffers are separate per-layer D3D11 buffers starting at offset 0.
    // This sidesteps the Intel HD 4600 SRV.FirstElement bug (non-zero FirstElement ignored).
    for (uint32_t l = 0; l < NL; ++l) {
        const std::string pfx = "transformer.h." + std::to_string(l) + ".";

        // LN1: h_buf_[l] → ln1_y_buf_[l], xhat_ln1_buf_[l], inv_std_ln1_[l]
        {
            struct { uint32_t E, S; float eps; uint32_t pad; } p{E, S, eps};
            setCB(&p, 16);
            ctx->CSSetShader(cs_lnorm_fwd_.Get(), nullptr, 0);
            auto sv0 = makeSRV(h_buf_[l].Get(), 0, S * E);
            ID3D11ShaderResourceView* srvs[3] = {
                sv0.Get(), wSRV(pfx + "ln_1.weight"), wSRV(pfx + "ln_1.bias")
            };
            ctx->CSSetShaderResources(0, 3, srvs);
            auto uv0 = makeUAV(ln1_y_buf_[l].Get(),    0, S * E);
            auto uv1 = makeUAV(xhat_ln1_buf_[l].Get(), 0, S * E);
            auto uv2 = makeUAV(inv_std_ln1_[l].Get(),  0, S);
            ID3D11UnorderedAccessView* uvs[3] = { uv0.Get(), uv1.Get(), uv2.Get() };
            ctx->CSSetUnorderedAccessViews(0, 3, uvs, nullptr);
            ctx->Dispatch(S, 1, 1); clearViews(3, 3);
        }

        // QKV projection: ln1_y_buf_[l] → qkv_buf_[l]
        {
            struct { uint32_t M, K, N, use_bias, A_off, B_off; uint32_t pad[2]; } p{S, E, 3*E, 1, 0, 0};
            setCB(&p, 32);
            ctx->CSSetShader(cs_matmul_fwd_.Get(), nullptr, 0);
            auto sv0 = makeSRV(ln1_y_buf_[l].Get(), 0, S * E);
            ID3D11ShaderResourceView* srvs[3] = {
                sv0.Get(), wSRV(pfx + "attn.c_attn.weight"), wSRV(pfx + "attn.c_attn.bias")
            };
            ctx->CSSetShaderResources(0, 3, srvs);
            zero(qkv_buf_[l].Get(), 0, S * 3 * E);
            auto uv0 = makeUAV(qkv_buf_[l].Get(), 0, S * 3 * E);
            ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
            ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
            ctx->Dispatch((3 * E + 15) / 16, (S + 15) / 16, 1); clearViews(1, 3);
        }

        // Attention pass 1: QK dot → P_buf_[l] raw logits (no softmax yet)
        {
            struct { uint32_t S, E, D; float scale; } p{S, E, D, scale};
            setCB(&p, 16);
            ctx->CSSetShader(cs_attn_qk_dot_.Get(), nullptr, 0);
            auto sv0 = makeSRV(qkv_buf_[l].Get(), 0, S * 3 * E);
            ID3D11ShaderResourceView*  srvs[1] = { sv0.Get() };
            ctx->CSSetShaderResources(0, 1, srvs);
            auto uv0 = makeUAV(P_buf_[l].Get(), 0, H * S * S);
            ID3D11UnorderedAccessView* uvs[1]  = { uv0.Get() };
            ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
            ctx->Dispatch(H, 1, 1); clearViews(1, 1);
        }

        // Per-layer gravity field: LBS-overlap bias on P_buf_[l] (TRUE pre-softmax)
        if (gravity_sync_) {
            float gscale = physics_ ? physics_->GetAntigravityFloat() * 0.5f : 0.1f;
            struct { uint32_t seq_len, n_head, layer_idx; float gravity_scale; } gc{ S, H, l, gscale };
            D3D11_MAPPED_SUBRESOURCE ms{};
            ctx->Map(gravity_layer_cb_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &ms);
            if (ms.pData) std::memcpy(ms.pData, &gc, sizeof(gc));
            ctx->Unmap(gravity_layer_cb_.Get(), 0);
            ID3D11Buffer* gcb = gravity_layer_cb_.Get();
            ctx->CSSetConstantBuffers(0, 1, &gcb);
            ctx->CSSetShader(cs_gravity_field_layer_.Get(), nullptr, 0);
            auto sv0 = makeSRV (ln1_y_buf_[l].Get(),        0, S * E);
            auto sv1 = makeSRVi(seq_bone_ids_buf_.Get(),     0, S * 4);
            auto sv2 = makeSRV (seq_bone_weights_buf_.Get(), 0, S * 4);
            auto uv0 = makeUAV (P_buf_[l].Get(),             0, H * S * S);
            ID3D11ShaderResourceView*  srvs_g[3] = { sv0.Get(), sv1.Get(), sv2.Get() };
            ID3D11UnorderedAccessView* uvs_g [1] = { uv0.Get() };
            ctx->CSSetShaderResources(0, 3, srvs_g);
            ctx->CSSetUnorderedAccessViews(0, 1, uvs_g, nullptr);
            ctx->Dispatch(H, S, 1);
            ctx->End(gravity_sync_query_.Get());  // per-layer GPU timeline fence
            clearViews(1, 3);
        }

        // TENSOR FOLD CORE: per-cluster XVM dispatch (intra-cluster amplified gravity)
        if (tensor_fold_core_) {
            float gscale = physics_ ? physics_->GetAntigravityFloat() * 0.5f : 0.1f;
            auto sv_idx = makeSRV (sorted_token_idx_buf_.Get(), 0, S);
            auto sv_bid = makeSRV (seq_bone_ids_buf_.Get(),     0, S * 4);
            auto sv_bwt = makeSRV (seq_bone_weights_buf_.Get(), 0, S * 4);
            auto uv_p   = makeUAV (P_buf_[l].Get(),             0, H * S * S);
            ID3D11ShaderResourceView*  srvs_fc[3] = { sv_idx.Get(), sv_bid.Get(), sv_bwt.Get() };
            ID3D11UnorderedAccessView* uvs_fc [1] = { uv_p.Get() };
            ctx->CSSetShaderResources(0, 3, srvs_fc);
            ctx->CSSetUnorderedAccessViews(0, 1, uvs_fc, nullptr);

            for (uint32_t c = 0; c < N_FOLD_CLUSTERS; ++c) {
                if (cpu_cluster_count_[c] == 0) continue;
                auto it = fold_dispatch_table_.find((int32_t)c);
                if (it == fold_dispatch_table_.end()) continue;

                struct { uint32_t seq_len, n_head, cluster_start; float gravity_scale; }
                    fd{ S, H, cpu_cluster_start_[c], gscale * it->second.pressure };
                D3D11_MAPPED_SUBRESOURCE ms{};
                ctx->Map(fold_dispatch_cb_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &ms);
                if (ms.pData) std::memcpy(ms.pData, &fd, sizeof(fd));
                ctx->Unmap(fold_dispatch_cb_.Get(), 0);
                ID3D11Buffer* fdcb = fold_dispatch_cb_.Get();
                ctx->CSSetConstantBuffers(0, 1, &fdcb);
                ctx->CSSetShader(it->second.cs.Get(), nullptr, 0);
                ctx->Dispatch(H, cpu_cluster_count_[c], 1);
            }
            clearViews(1, 3);
        }

        // π-nary + brain think-bias: modify P_buf_[l] in-place (pre-softmax)
        if (think_bias_) {
            float ag = physics_ ? physics_->GetAntigravityFloat() : 0.3f;
            brain_scale_ = ag * 0.5f;
            struct {
                uint32_t seq_len, n_head;
                float g0, g1, g2, g3, g4;
                float antigravity_scale, brain_scale;
                float _pad0, _pad1, _pad2;  // 12 × 4 = 48 bytes, 3 × vec4 aligned
            } tb{ S, H, 0.5f, 0.2f, 0.7f, 0.3f, 0.1f, ag, brain_scale_, 0.f, 0.f, 0.f };
            D3D11_MAPPED_SUBRESOURCE ms{};
            ctx->Map(think_bias_cb_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &ms);
            if (ms.pData) std::memcpy(ms.pData, &tb, sizeof(tb));
            ctx->Unmap(think_bias_cb_.Get(), 0);
            ID3D11Buffer* cb0 = think_bias_cb_.Get();
            ctx->CSSetConstantBuffers(0, 1, &cb0);
            ctx->CSSetShader(cs_kuhul_think_bias_.Get(), nullptr, 0);
            auto uv0 = makeUAV(P_buf_[l].Get(),            0, H * S * S);
            auto sv0 = makeSRV (think_depth_buf_.Get(),     0, S);
            auto sv1 = makeSRVi(seq_bone_ids_buf_.Get(),    0, S * 4);
            auto sv2 = makeSRV (seq_bone_weights_buf_.Get(),0, S * 4);
            ID3D11UnorderedAccessView* uvs2[1] = { uv0.Get() };
            ID3D11ShaderResourceView*  srvs2[3]= { sv0.Get(), sv1.Get(), sv2.Get() };
            ctx->CSSetUnorderedAccessViews(0, 1, uvs2, nullptr);
            ctx->CSSetShaderResources(0, 3, srvs2);
            // Dispatch(H, S, 1): one group per head×query; 128 threads cover key dim
            // Note: only correct when S ≤ 128 (default block_size).
            ctx->Dispatch(H, S, 1);
            clearViews(1, 3);
        }

        // Attention pass 2: softmax(P_buf logits) + V accumulation → attn_out_buf_[l]
        {
            struct { uint32_t S, E, D; float scale; } p{S, E, D, scale};
            setCB(&p, 16);
            ctx->CSSetShader(cs_attn_softmax_.Get(), nullptr, 0);
            auto sv0 = makeSRV(qkv_buf_[l].Get(), 0, S * 3 * E);
            ID3D11ShaderResourceView*  srvs[1] = { sv0.Get() };
            ctx->CSSetShaderResources(0, 1, srvs);
            zero(attn_out_buf_[l].Get(), 0, S * E);
            auto uv0 = makeUAV(P_buf_[l].Get(),        0, H * S * S);
            auto uv1 = makeUAV(attn_out_buf_[l].Get(), 0, S * E);
            ID3D11UnorderedAccessView* uvs[2] = { uv0.Get(), uv1.Get() };
            ctx->CSSetUnorderedAccessViews(0, 2, uvs, nullptr);
            ctx->Dispatch(H, 1, 1); clearViews(2, 1);
        }

        // c_proj_attn: attn_out_buf_[l] → dh_buf_ (temp)
        {
            struct { uint32_t M, K, N, use_bias, A_off, B_off; uint32_t pad[2]; } p{S, E, E, 1, 0, 0};
            setCB(&p, 32);
            ctx->CSSetShader(cs_matmul_fwd_.Get(), nullptr, 0);
            auto sv0 = makeSRV(attn_out_buf_[l].Get(), 0, S * E);
            ID3D11ShaderResourceView* srvs[3] = {
                sv0.Get(), wSRV(pfx + "attn.c_proj.weight"), wSRV(pfx + "attn.c_proj.bias")
            };
            ctx->CSSetShaderResources(0, 3, srvs);
            zero(dh_buf_.Get(), 0, S * E);
            auto uv0 = makeUAV(dh_buf_.Get(), 0, S * E);
            ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
            ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
            ctx->Dispatch((E + 15) / 16, (S + 15) / 16, 1); clearViews(1, 3);
        }

        // First residual: h_buf_[l+1] = h_buf_[l] + dh_buf_
        {
            struct { uint32_t numel, pad[3]; } p{S * E};
            setCB(&p, 16);
            ctx->CSSetShader(cs_resadd_add3_.Get(), nullptr, 0);
            auto sv0 = makeSRV(h_buf_[l].Get(),   0, S * E);
            auto sv1 = makeSRV(dh_buf_.Get(),      0, S * E);
            ID3D11ShaderResourceView* srvs[2] = { sv0.Get(), sv1.Get() };
            ctx->CSSetShaderResources(0, 2, srvs);
            auto uv0 = makeUAV(h_buf_[l+1].Get(), 0, S * E);
            ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
            ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
            ctx->Dispatch((S * E + 255) / 256, 1, 1); clearViews(1, 2);
        }

        // LN2: h_buf_[l+1] → ln2_y_buf_[l], xhat_ln2_buf_[l], inv_std_ln2_[l]
        {
            struct { uint32_t E, S; float eps; uint32_t pad; } p{E, S, eps};
            setCB(&p, 16);
            ctx->CSSetShader(cs_lnorm_fwd_.Get(), nullptr, 0);
            auto sv0 = makeSRV(h_buf_[l+1].Get(), 0, S * E);
            ID3D11ShaderResourceView* srvs[3] = {
                sv0.Get(), wSRV(pfx + "ln_2.weight"), wSRV(pfx + "ln_2.bias")
            };
            ctx->CSSetShaderResources(0, 3, srvs);
            auto uv0 = makeUAV(ln2_y_buf_[l].Get(),    0, S * E);
            auto uv1 = makeUAV(xhat_ln2_buf_[l].Get(), 0, S * E);
            auto uv2 = makeUAV(inv_std_ln2_[l].Get(),  0, S);
            ID3D11UnorderedAccessView* uvs[3] = { uv0.Get(), uv1.Get(), uv2.Get() };
            ctx->CSSetUnorderedAccessViews(0, 3, uvs, nullptr);
            ctx->Dispatch(S, 1, 1); clearViews(3, 3);
        }

        // c_fc: ln2_y_buf_[l] → mlp_pre_buf_[l]
        {
            struct { uint32_t M, K, N, use_bias, A_off, B_off; uint32_t pad[2]; } p{S, E, F, 1, 0, 0};
            setCB(&p, 32);
            ctx->CSSetShader(cs_matmul_fwd_.Get(), nullptr, 0);
            auto sv0 = makeSRV(ln2_y_buf_[l].Get(), 0, S * E);
            ID3D11ShaderResourceView* srvs[3] = {
                sv0.Get(), wSRV(pfx + "mlp.c_fc.weight"), wSRV(pfx + "mlp.c_fc.bias")
            };
            ctx->CSSetShaderResources(0, 3, srvs);
            zero(mlp_pre_buf_[l].Get(), 0, S * F);
            auto uv0 = makeUAV(mlp_pre_buf_[l].Get(), 0, S * F);
            ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
            ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
            ctx->Dispatch((F + 15) / 16, (S + 15) / 16, 1); clearViews(1, 3);
        }

        // GELU: mlp_pre_buf_[l] → mlp_gelu_buf_[l]
        // Per-layer buffer starts at offset 0; x_in_offset=0 in CB.
        zero(mlp_gelu_buf_[l].Get(), 0, S * F);
        {
            struct { uint32_t numel, x_in_offset, pad[2]; } p{S * F, 0, {0,0}};
            setCB(&p, 16);
            ctx->CSSetShader(cs_gelu_fwd_.Get(), nullptr, 0);
            auto sv0 = makeSRV(mlp_pre_buf_[l].Get(), 0, S * F);
            ID3D11ShaderResourceView* srvs[1] = { sv0.Get() };
            ctx->CSSetShaderResources(0, 1, srvs);
            auto uv0 = makeUAV(mlp_gelu_buf_[l].Get(), 0, S * F);
            ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
            ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
            ctx->Dispatch((S * F + 255) / 256, 1, 1); clearViews(1, 1);
        }

        // c_proj_mlp: mlp_gelu_buf_[l] → dh_buf_ (temp)
        {
            struct { uint32_t M, K, N, use_bias, A_off, B_off; uint32_t pad[2]; } p{S, F, E, 1, 0, 0};
            setCB(&p, 32);
            ctx->CSSetShader(cs_matmul_fwd_.Get(), nullptr, 0);
            auto sv0 = makeSRV(mlp_gelu_buf_[l].Get(), 0, S * F);
            ID3D11ShaderResourceView* srvs[3] = {
                sv0.Get(), wSRV(pfx + "mlp.c_proj.weight"), wSRV(pfx + "mlp.c_proj.bias")
            };
            ctx->CSSetShaderResources(0, 3, srvs);
            zero(dh_buf_.Get(), 0, S * E);
            auto uv0 = makeUAV(dh_buf_.Get(), 0, S * E);
            ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
            ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
            ctx->Dispatch((E + 15) / 16, (S + 15) / 16, 1); clearViews(1, 3);
        }

        // Second residual: h_buf_[l+1] += dh_buf_
        {
            struct { uint32_t numel, pad[3]; } p{S * E};
            setCB(&p, 16);
            ctx->CSSetShader(cs_resadd_addto_.Get(), nullptr, 0);
            auto sv0 = makeSRV(dh_buf_.Get(), 0, S * E);
            ID3D11ShaderResourceView* srvs[1] = { sv0.Get() };
            ctx->CSSetShaderResources(0, 1, srvs);
            auto uv0 = makeUAV(h_buf_[l+1].Get(), 0, S * E);
            ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
            ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
            ctx->Dispatch((S * E + 255) / 256, 1, 1); clearViews(1, 1);
        }
    } // end layer loop

    // 3. Final LN: h_buf_[NL] → lnf_y_buf_[S,E], xhat_lnf, inv_std_lnf
    {
        struct { uint32_t E, S; float eps; uint32_t pad; } p{E, S, eps};
        setCB(&p, 16);
        ctx->CSSetShader(cs_lnorm_fwd_.Get(), nullptr, 0);
        auto sv0 = makeSRV(h_buf_[NL].Get(), 0, S * E);  // per-layer buffer at offset 0
        ID3D11ShaderResourceView* srvs[3] = {
            sv0.Get(), wSRV("transformer.ln_f.weight"), wSRV("transformer.ln_f.bias")
        };
        ctx->CSSetShaderResources(0, 3, srvs);
        auto uv0 = makeUAV(lnf_y_buf_.Get(),    0, S * E);
        auto uv1 = makeUAV(xhat_lnf_buf_.Get(), 0, S * E);
        auto uv2 = makeUAV(inv_std_lnf_.Get(),  0, S);
        ID3D11UnorderedAccessView* uvs[3] = { uv0.Get(), uv1.Get(), uv2.Get() };
        ctx->CSSetUnorderedAccessViews(0, 3, uvs, nullptr);
        ctx->Dispatch(S, 1, 1); clearViews(3, 3);
    }

    // 4. LM head: lnf_y_buf_[S-2, E] @ wte.T → logits_buf_[V]
    // Read from position S-2 so the hidden state has NOT seen token[S-1].
    // h[S-1] already contains token[S-1] in its residual — using it as input trivially gives loss≈0.
    // A_row_offset=last tells CSMain_transB which row of lnf_y to use (Intel SRV offset workaround).
    if (S < 2) return 0.f;
    const uint32_t last = S - 2;
    {
        struct { uint32_t M, K, N, use_bias, A_row_offset, B_row_offset; uint32_t pad[2]; }
            p{1, E, V, 0, last, 0};
        setCB(&p, 32);
        ctx->CSSetShader(cs_matmul_fwd_transb_.Get(), nullptr, 0);
        auto sv0 = makeSRV(lnf_y_buf_.Get(), 0, S * E);  // full buffer; shader uses A_row_offset
        ID3D11ShaderResourceView* srvs[2] = { sv0.Get(), wSRV("transformer.wte.weight") };
        ctx->CSSetShaderResources(0, 2, srvs);
        zero(logits_buf_.Get(), 0, V);
        auto uv0 = makeUAV(logits_buf_.Get(), 0, V);
        ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
        ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
        ctx->Dispatch((V + 15) / 16, 1, 1); clearViews(1, 2);
    }

    // 5. Loss + dlogits: logits → loss_buf_, dlogits_buf_
    // target = seq[last+1] = seq[S-1]: the next token after position last=S-2.
    const uint32_t target = (uint32_t)seq[last + 1];
    {
        struct { uint32_t V, tgt; float inv_b; uint32_t pad; } p{V, target, inv_batch, 0};
        setCB(&p, 16);
        ctx->CSSetShader(cs_loss_.Get(), nullptr, 0);
        auto sv0 = makeSRV(logits_buf_.Get(), 0, V);
        ID3D11ShaderResourceView* srvs[1] = { sv0.Get() };
        ctx->CSSetShaderResources(0, 1, srvs);
        auto uv0 = makeUAV(dlogits_buf_.Get(), 0, V);
        auto uv1 = makeUAV(loss_buf_.Get(),    0, 1);
        ID3D11UnorderedAccessView* uvs[2] = { uv0.Get(), uv1.Get() };
        ctx->CSSetUnorderedAccessViews(0, 2, uvs, nullptr);
        ctx->Dispatch(1, 1, 1); clearViews(2, 1);
    }

    // DBG: spot-check a few key hidden states and logits (env-gated: GPT2_DBG).
    // Off by default — these are per-position GPU->CPU readbacks in the hot path.
    static const bool s_dbg = std::getenv("GPT2_DBG") != nullptr;
    if (s_dbg) {
        auto checkBuf = [&](const char* lbl, ID3D11Buffer* b, uint32_t n) {
            auto v = readbackBuffer(b, n);
            float mx = 0.f; bool bad = false;
            for (uint32_t i = 0; i < n; ++i) {
                if (!std::isfinite(v[i])) { bad = true; break; }
                mx = std::max(mx, std::abs(v[i]));
            }
            std::cerr << "[dbg] " << lbl << ": max=" << mx << (bad?" NAN":" ok") << "\n";
            return bad;
        };
        checkBuf("h[0](embed)",    h_buf_[0].Get(),  S * E);
        checkBuf("h[1](l0 out)",   h_buf_[1].Get(),  S * E);
        checkBuf("h[NL](final)",   h_buf_[NL].Get(), S * E);
        checkBuf("logits",         logits_buf_.Get(), V);
        std::cerr << "[dbg] loss=" << readbackLoss() << " target=" << target << "\n";
    }

    // ══════════════════════════ BACKWARD PASS ════════════════════════════════

    // Helper: dispatch matmul_bwd_dA  (dA[M,K] += dC[M,N] @ B[K,N].T)
    // M,K,N cbuffer; SRVs: A(t0), B(t1), dC(t2); UAVs: dA(u0), dB_dummy(u1)
    auto mmBwdA = [&](uint32_t M, uint32_t K, uint32_t N,
                      ID3D11ShaderResourceView* A_srv, ID3D11ShaderResourceView* B_srv,
                      ID3D11ShaderResourceView* dC_srv,
                      ID3D11UnorderedAccessView* dA_uav) {
        struct { uint32_t M, K, N, pad; } p{M, K, N, 0};
        setCB(&p, 16);
        ctx->CSSetShader(cs_matmul_bwd_dA_.Get(), nullptr, 0);
        ID3D11ShaderResourceView* srvs[3] = { A_srv, B_srv, dC_srv };
        ctx->CSSetShaderResources(0, 3, srvs);
        // u0=dA (written), u1=dummy (not written by CSMain_dA)
        auto dummy_uav = makeUAV(loss_buf_.Get(), 0, 1);
        ID3D11UnorderedAccessView* uvs[2] = { dA_uav, dummy_uav.Get() };
        ctx->CSSetUnorderedAccessViews(0, 2, uvs, nullptr);
        ctx->Dispatch((K + 15) / 16, (M + 15) / 16, 1); clearViews(2, 3);
    };

    // Helper: dispatch matmul_bwd_dB  (dB[K,N] += A[M,K].T @ dC[M,N])
    auto mmBwdB = [&](uint32_t M, uint32_t K, uint32_t N,
                      ID3D11ShaderResourceView* A_srv,
                      ID3D11ShaderResourceView* dC_srv,
                      ID3D11UnorderedAccessView* dB_uav) {
        struct { uint32_t M, K, N, pad; } p{M, K, N, 0};
        setCB(&p, 16);
        ctx->CSSetShader(cs_matmul_bwd_dB_.Get(), nullptr, 0);
        // B is not needed by CSMain_dB but must bind something valid
        ID3D11ShaderResourceView* srvs[3] = { A_srv, nullptr, dC_srv };
        ctx->CSSetShaderResources(0, 3, srvs);
        auto dummy_uav = makeUAV(loss_buf_.Get(), 0, 1);
        ID3D11UnorderedAccessView* uvs[2] = { dummy_uav.Get(), dB_uav };
        ctx->CSSetUnorderedAccessViews(0, 2, uvs, nullptr);
        ctx->Dispatch((N + 15) / 16, (K + 15) / 16, 1); clearViews(2, 3);
    };

    // Helper: accumulate bias gradient: dbias[N] += sum_rows(dC[M,N])
    auto biasBwd = [&](uint32_t M, uint32_t N,
                       ID3D11ShaderResourceView* dC_srv,
                       ID3D11UnorderedAccessView* dbias_uav) {
        struct { uint32_t M, N, pad[2]; } p{M, N};
        setCB(&p, 16);
        ctx->CSSetShader(cs_bias_bwd_.Get(), nullptr, 0);
        ID3D11ShaderResourceView* srvs[1] = { dC_srv };
        ctx->CSSetShaderResources(0, 1, srvs);
        ID3D11UnorderedAccessView* uvs[1] = { dbias_uav };
        ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
        ctx->Dispatch((N + 255) / 256, 1, 1); clearViews(1, 1);
    };

    // Helper: layernorm backward
    // dout_srv → dh_buf_ (+=), dgamma, dbeta
    // dout_srv must NOT alias dh_buf_ — D3D11 silently nulls SRV when same buffer is UAV.
    auto lnBwd = [&](uint32_t xh_off, ID3D11Buffer* xhat_buf,
                     ID3D11ShaderResourceView* gamma_srv,
                     uint32_t is_off, ID3D11Buffer* inv_std_buf,
                     ID3D11ShaderResourceView* dout_srv,  // explicit: must differ from dh_buf_
                     ID3D11UnorderedAccessView* dgamma_uav,
                     ID3D11UnorderedAccessView* dbeta_uav) {
        struct { uint32_t E, S; float eps; uint32_t pad; } p{E, S, eps};
        setCB(&p, 16);

        // Pass 1: CSMain — compute dx (per-position, S groups). No dgamma/dbeta touch.
        ctx->CSSetShader(cs_lnorm_bwd_.Get(), nullptr, 0);
        auto sv0 = makeSRV(xhat_buf,    xh_off, S * E);
        auto sv3 = makeSRV(inv_std_buf, is_off, S);
        ID3D11ShaderResourceView* srvs[4] = { sv0.Get(), gamma_srv, dout_srv, sv3.Get() };
        ctx->CSSetShaderResources(0, 4, srvs);
        auto uv0 = makeUAV(dh_buf_.Get(), 0, S * E);  // dx (+=)
        ID3D11UnorderedAccessView* uvs1[3] = { uv0.Get(), nullptr, nullptr };
        ctx->CSSetUnorderedAccessViews(0, 3, uvs1, nullptr);
        ctx->Dispatch(S, 1, 1); clearViews(3, 4);

        // Pass 2: CSMain_params — accumulate dgamma/dbeta (per-dim, ceil(E/256) groups).
        // Each thread owns one j in [0,E), loops over all S positions. No race.
        ctx->CSSetShader(cs_lnorm_bwd_params_.Get(), nullptr, 0);
        auto sv0p = makeSRV(xhat_buf, xh_off, S * E);
        ID3D11ShaderResourceView* srvsp[4] = { sv0p.Get(), nullptr, dout_srv, nullptr };
        ctx->CSSetShaderResources(0, 4, srvsp);
        ID3D11UnorderedAccessView* uvs2[3] = { nullptr, dgamma_uav, dbeta_uav };
        ctx->CSSetUnorderedAccessViews(0, 3, uvs2, nullptr);
        ctx->Dispatch((E + 255) / 256, 1, 1); clearViews(3, 4);
    };

    // 6. LM head backward.
    //    fullseq_: loop ALL positions — every target (incl. phase/glyph tokens) gets gradient,
    //              dh_buf_ filled at all positions. Gradient mean-scaled by 1/S.
    //    else:     last-position only (original, VRAM-lean).
    auto& wte_p = params_[param_idx_.at("transformer.wte.weight")];
    const uint32_t VCHUNK = 2048;
    if (fullseq_) {
        zero(dh_buf_.Get(), 0, S * E);
        const float inv_bS = inv_batch / (float)S;
        for (uint32_t s = 0; s < S; ++s) {
            const uint32_t tgt = (uint32_t)seq[s];   // same convention as last-position path
            // LM head at s -> logits_buf_
            {
                struct { uint32_t M, K, N, use_bias, A_off, B_off; uint32_t pad[2]; } p{1, E, V, 0, s, 0};
                setCB(&p, 32);
                ctx->CSSetShader(cs_matmul_fwd_transb_.Get(), nullptr, 0);
                auto sv0 = makeSRV(lnf_y_buf_.Get(), 0, S * E);
                ID3D11ShaderResourceView* srvs[2] = { sv0.Get(), wSRV("transformer.wte.weight") };
                ctx->CSSetShaderResources(0, 2, srvs);
                zero(logits_buf_.Get(), 0, V);
                auto uv0 = makeUAV(logits_buf_.Get(), 0, V);
                ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
                ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
                ctx->Dispatch((V + 15) / 16, 1, 1); clearViews(1, 2);
            }
            // Loss + dlogits (target tgt, mean-scaled) -> dlogits_buf_, loss_buf_
            {
                struct { uint32_t V, tgt; float inv_b; uint32_t pad; } p{V, tgt, inv_bS, 0};
                setCB(&p, 16);
                ctx->CSSetShader(cs_loss_.Get(), nullptr, 0);
                auto sv0 = makeSRV(logits_buf_.Get(), 0, V);
                ID3D11ShaderResourceView* srvs[1] = { sv0.Get() };
                ctx->CSSetShaderResources(0, 1, srvs);
                auto uv0 = makeUAV(dlogits_buf_.Get(), 0, V);
                auto uv1 = makeUAV(loss_buf_.Get(), 0, 1);
                ID3D11UnorderedAccessView* uvs[2] = { uv0.Get(), uv1.Get() };
                ctx->CSSetUnorderedAccessViews(0, 2, uvs, nullptr);
                ctx->Dispatch(1, 1, 1); clearViews(2, 1);
            }
            // record per-position loss (no CPU stall): loss_buf_[0] -> loss_all_[s]
            { D3D11_BOX box{ 0, 0, 0, (UINT)sizeof(float), 1, 1 };
              ctx->CopySubresourceRegion(loss_all_.Get(), 0, (UINT)(s * sizeof(float)), 0, 0,
                                         loss_buf_.Get(), 0, &box); }
            // d_wte += dlogits @ lnf_y[s]  (chunked over V, B_off=s)
            {
                auto sv0 = makeSRV(dlogits_buf_.Get(), 0, V);
                auto sv1 = makeSRV(lnf_y_buf_.Get(),   0, S * E);
                for (uint32_t v0 = 0; v0 < V; v0 += VCHUNK) {
                    uint32_t vn = std::min(VCHUNK, V - v0);
                    struct { uint32_t M, K, N, use_bias, A_off, B_off; uint32_t pad[2]; } p{vn, 1, E, 0, v0, s};
                    setCB(&p, 32);
                    ctx->CSSetShader(cs_matmul_fwd_.Get(), nullptr, 0);
                    ID3D11ShaderResourceView* srvs[3] = { sv0.Get(), sv1.Get(), nullptr };
                    ctx->CSSetShaderResources(0, 3, srvs);
                    auto uv0 = makeUAV(wte_p.g_buf.Get(), v0 * E, vn * E);
                    ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
                    ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
                    ctx->Dispatch((E + 15) / 16, (vn + 15) / 16, 1); clearViews(1, 3);
                }
            }
            // d_lnf_y[s] = wte @ dlogits -> dh_buf_[s*E]
            {
                struct { uint32_t M, K, N, use_bias, A_off, B_off; uint32_t pad[2]; } p{1, V, E, 0, 0, 0};
                setCB(&p, 32);
                ctx->CSSetShader(cs_matmul_fwd_.Get(), nullptr, 0);
                auto sv0 = makeSRV(dlogits_buf_.Get(), 0, V);
                ID3D11ShaderResourceView* srvs[3] = { sv0.Get(), wSRV("transformer.wte.weight"), nullptr };
                ctx->CSSetShaderResources(0, 3, srvs);
                auto uv0 = makeUAV(dh_buf_.Get(), s * E, E);
                ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
                ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
                ctx->Dispatch((E + 15) / 16, 1, 1); clearViews(1, 3);
            }
        }
    } else {
    // 6a. d_wte += dlogits[V,1] @ lnf_y[last,E]   (outer product, M=V, K=1, N=E)
    {
        auto sv0 = makeSRV(dlogits_buf_.Get(), 0, V);
        auto sv1 = makeSRV(lnf_y_buf_.Get(),   0, S * E);
        for (uint32_t v0 = 0; v0 < V; v0 += VCHUNK) {
            uint32_t vn = std::min(VCHUNK, V - v0);
            struct { uint32_t M, K, N, use_bias, A_off, B_off; uint32_t pad[2]; }
                p{vn, 1, E, 0, v0, last};
            setCB(&p, 32);
            ctx->CSSetShader(cs_matmul_fwd_.Get(), nullptr, 0);
            ID3D11ShaderResourceView* srvs[3] = { sv0.Get(), sv1.Get(), nullptr };
            ctx->CSSetShaderResources(0, 3, srvs);
            auto uv0 = makeUAV(wte_p.g_buf.Get(), v0 * E, vn * E);
            ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
            ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
            ctx->Dispatch((E + 15) / 16, (vn + 15) / 16, 1); clearViews(1, 3);
        }
    }

    // 6b. d_lnf_y[last] = wte @ dlogits  → dh_buf_[last*E .. +E]; other positions 0
    zero(dh_buf_.Get(), 0, S * E);
    {
        struct { uint32_t M, K, N, use_bias, A_off, B_off; uint32_t pad[2]; } p{1, V, E, 0, 0, 0};
        setCB(&p, 32);
        ctx->CSSetShader(cs_matmul_fwd_.Get(), nullptr, 0);
        auto sv0 = makeSRV(dlogits_buf_.Get(), 0, V);
        ID3D11ShaderResourceView* srvs[3] = { sv0.Get(), wSRV("transformer.wte.weight"), nullptr };
        ctx->CSSetShaderResources(0, 3, srvs);
        auto uv0 = makeUAV(dh_buf_.Get(), last * E, E);
        ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
        ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
        ctx->Dispatch((E + 15) / 16, 1, 1); clearViews(1, 3);
    }
    }

    // 7. Final LN backward: d_lnf_y (in dh_buf_[last]) → dx into dh_buf_, d_gamma, d_beta
    // Fix: dh_buf_ was previously passed as both dout SRV and dx UAV — D3D11 silently
    // nulled the SRV, making this backward a no-op and causing gradient explosion.
    // Fix: copy d_lnf_y to lnf_y_buf_ (idle temp), zero dh_buf_, then use lnf_y_buf_ as dout.
    {
        ctx->CopyResource(lnf_y_buf_.Get(), dh_buf_.Get());  // save d_lnf_y to temp
        zero(dh_buf_.Get(), 0, S * E);                       // dx starts at 0
        auto dout_srv = makeSRV(lnf_y_buf_.Get(), 0, S * E); // dout = d_lnf_y (different buf)
        auto dg_uav = makeUAV(params_[param_idx_.at("transformer.ln_f.weight")].g_buf.Get(),
                              0, E);
        auto db_uav = makeUAV(params_[param_idx_.at("transformer.ln_f.bias")].g_buf.Get(),
                              0, E);
        lnBwd(0, xhat_lnf_buf_.Get(), wSRV("transformer.ln_f.weight"),
              0, inv_std_lnf_.Get(), dout_srv.Get(), dg_uav.Get(), db_uav.Get());
    }

    // 8. Layer backward loop (NL-1 downto 0)
    for (int32_t li = (int32_t)NL - 1; li >= 0; --li) {
        const uint32_t l   = (uint32_t)li;
        const std::string pfx = "transformer.h." + std::to_string(l) + ".";

        // === MLP BACKWARD ===
        // dh_buf_ = gradient into h[l+1]
        // Residual: d_h_mid = dh_buf_ (skip path), d_mlp_out = dh_buf_

        // 8a. d_c_proj_mlp_w += mlp_gelu[l].T @ dh_buf_   (dB: M=S, K=F, N=E)
        {
            auto A_srv  = makeSRV(mlp_gelu_buf_[l].Get(), 0, S * F);
            auto dC_srv = makeSRV(dh_buf_.Get(),       0,    S * E);
            auto dB_uav = makeUAV(params_[param_idx_.at(pfx + "mlp.c_proj.weight")].g_buf.Get(),
                                  0, F * E);
            mmBwdB(S, F, E, A_srv.Get(), dC_srv.Get(), dB_uav.Get());
        }

        // 8b. d_c_proj_mlp_b += sum_rows(dh_buf_)
        {
            auto dC_srv  = makeSRV(dh_buf_.Get(), 0, S * E);
            auto db_uav  = makeUAV(params_[param_idx_.at(pfx + "mlp.c_proj.bias")].g_buf.Get(),
                                   0, E);
            biasBwd(S, E, dC_srv.Get(), db_uav.Get());
        }

        // 8c. d_mlp_gelu = dh_buf_ @ c_proj_mlp_w.T   (dA: M=S, K=F, N=E → d_mlp_buf_)
        zero(d_mlp_buf_.Get(), 0, S * F);
        {
            auto B_srv  = makeSRV(params_[param_idx_.at(pfx + "mlp.c_proj.weight")].w_buf.Get(),
                                  0, F * E);
            auto dC_srv = makeSRV(dh_buf_.Get(), 0, S * E);
            auto dA_uav = makeUAV(d_mlp_buf_.Get(), 0, S * F);
            mmBwdA(S, F, E, nullptr, B_srv.Get(), dC_srv.Get(), dA_uav.Get());
        }

        // 8d. GELU backward: d_mlp_pre = d_mlp_gelu * gelu'(mlp_pre[l])  (in-place)
        // Shader reads d_mlp_buf_ via RWStructuredBuffer u0 and multiplies in-place.
        // Previous binding of d_mlp_buf_ as both SRV t1 and UAV u0 caused D3D11 to
        // silently null the SRV, making the backward a no-op. Fixed: u0 only.
        {
            struct { uint32_t numel, pad[3]; } p{S * F};
            setCB(&p, 16);
            ctx->CSSetShader(cs_gelu_bwd_.Get(), nullptr, 0);
            auto sv0 = makeSRV(mlp_pre_buf_[l].Get(), 0, S * F);  // pre-GELU values
            ID3D11ShaderResourceView* srvs[1] = { sv0.Get() };
            ctx->CSSetShaderResources(0, 1, srvs);
            auto uv0 = makeUAV(d_mlp_buf_.Get(), 0, S * F);  // in-place multiply
            ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
            ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
            ctx->Dispatch((S * F + 255) / 256, 1, 1); clearViews(1, 1);
        }

        // 8e. d_c_fc_w += ln2_y[l].T @ d_mlp_buf_   (dB: M=S, K=E, N=F)
        {
            auto A_srv  = makeSRV(ln2_y_buf_[l].Get(), 0, S * E);
            auto dC_srv = makeSRV(d_mlp_buf_.Get(), 0,     S * F);
            auto dB_uav = makeUAV(params_[param_idx_.at(pfx + "mlp.c_fc.weight")].g_buf.Get(),
                                  0, E * F);
            mmBwdB(S, E, F, A_srv.Get(), dC_srv.Get(), dB_uav.Get());
        }

        // 8f. d_c_fc_b += sum_rows(d_mlp_buf_)
        {
            auto dC_srv = makeSRV(d_mlp_buf_.Get(), 0, S * F);
            auto db_uav = makeUAV(params_[param_idx_.at(pfx + "mlp.c_fc.bias")].g_buf.Get(),
                                  0, F);
            biasBwd(S, F, dC_srv.Get(), db_uav.Get());
        }

        // 8g. d_ln2_y = d_mlp_buf_ @ c_fc_w.T   (dA: M=S, K=E, N=F)
        //     → d_qkv_buf_ first S*E elements (reused as temp)
        zero(d_qkv_buf_.Get(), 0, S * E);
        {
            auto B_srv  = makeSRV(params_[param_idx_.at(pfx + "mlp.c_fc.weight")].w_buf.Get(),
                                  0, E * F);
            auto dC_srv = makeSRV(d_mlp_buf_.Get(), 0, S * F);
            auto dA_uav = makeUAV(d_qkv_buf_.Get(), 0, S * E);
            mmBwdA(S, E, F, nullptr, B_srv.Get(), dC_srv.Get(), dA_uav.Get());
        }
        // d_qkv_buf_[0..S*E] now holds d_ln2_y

        // 8h. LN2 backward: d_ln2_y (in d_qkv_buf_) is stored as dout in…
        //     We need to swap: currently dh_buf_ is the gradient we're propagating.
        //     LN bwd reads dout from dh_buf_ (hardcoded in lnBwd helper).
        //     So: first copy d_ln2_y to dh_buf_, then call lnBwd (which adds dx to dh_buf_).
        //     But lnBwd reads dout = current dh_buf_ (which is d_h_out/d_h_mid skip gradient).
        //     We need dout = d_ln2_y, not d_h_mid.
        //
        //     Fix: use a separate LN bwd call where dout = d_qkv_buf_[:S*E].
        //     The lnBwd helper is hardcoded to read dh_buf_ as dout — we need a different binding.
        //
        //     So: do it inline here rather than via lnBwd helper.
        {
            struct { uint32_t E, S; float eps; uint32_t pad; } p{E, S, eps};
            setCB(&p, 16);
            auto sv0 = makeSRV(xhat_ln2_buf_[l].Get(), 0, S * E);  // xhat
            auto sv1 = wSRV(pfx + "ln_2.weight");                   // gamma
            auto sv2 = makeSRV(d_qkv_buf_.Get(),     0, S * E);    // dout = d_ln2_y
            auto sv3 = makeSRV(inv_std_ln2_[l].Get(), 0, S);        // inv_std
            auto uv1 = makeUAV(params_[param_idx_.at(pfx + "ln_2.weight")].g_buf.Get(), 0, E);
            auto uv2 = makeUAV(params_[param_idx_.at(pfx + "ln_2.bias")  ].g_buf.Get(), 0, E);

            // Pass 1: dx only (no dgamma/dbeta to avoid race across S groups)
            ctx->CSSetShader(cs_lnorm_bwd_.Get(), nullptr, 0);
            ID3D11ShaderResourceView* srvs[4] = { sv0.Get(), sv1, sv2.Get(), sv3.Get() };
            ctx->CSSetShaderResources(0, 4, srvs);
            auto uv0 = makeUAV(dh_buf_.Get(), 0, S * E);  // dx += (adds to d_h_mid)
            ID3D11UnorderedAccessView* uvs1[3] = { uv0.Get(), nullptr, nullptr };
            ctx->CSSetUnorderedAccessViews(0, 3, uvs1, nullptr);
            ctx->Dispatch(S, 1, 1); clearViews(3, 4);

            // Pass 2: dgamma/dbeta, race-free (one thread per dim j, loops over S)
            ctx->CSSetShader(cs_lnorm_bwd_params_.Get(), nullptr, 0);
            ID3D11ShaderResourceView* srvsp[4] = { sv0.Get(), nullptr, sv2.Get(), nullptr };
            ctx->CSSetShaderResources(0, 4, srvsp);
            ID3D11UnorderedAccessView* uvs2[3] = { nullptr, uv1.Get(), uv2.Get() };
            ctx->CSSetUnorderedAccessViews(0, 3, uvs2, nullptr);
            ctx->Dispatch((E + 255) / 256, 1, 1); clearViews(3, 4);
        }
        // dh_buf_ = d_h_mid  (gradient into h after first residual)

        // === ATTENTION BACKWARD ===

        // 8i. d_c_proj_attn_w += attn_out[l].T @ dh_buf_   (dB: M=S, K=E, N=E)
        {
            auto A_srv  = makeSRV(attn_out_buf_[l].Get(), 0, S * E);
            auto dC_srv = makeSRV(dh_buf_.Get(),       0,    S * E);
            auto dB_uav = makeUAV(params_[param_idx_.at(pfx + "attn.c_proj.weight")].g_buf.Get(),
                                  0, E * E);
            mmBwdB(S, E, E, A_srv.Get(), dC_srv.Get(), dB_uav.Get());
        }

        // 8j. d_c_proj_attn_b += sum_rows(dh_buf_)
        {
            auto dC_srv = makeSRV(dh_buf_.Get(), 0, S * E);
            auto db_uav = makeUAV(params_[param_idx_.at(pfx + "attn.c_proj.bias")].g_buf.Get(),
                                  0, E);
            biasBwd(S, E, dC_srv.Get(), db_uav.Get());
        }

        // 8k. d_attn_out = dh_buf_ @ c_proj_attn_w.T   (dA: M=S, K=E, N=E)
        //     → lnf_y_buf_ (reused as temp, forward done)
        zero(lnf_y_buf_.Get(), 0, S * E);
        {
            auto B_srv  = makeSRV(params_[param_idx_.at(pfx + "attn.c_proj.weight")].w_buf.Get(),
                                  0, E * E);
            auto dC_srv = makeSRV(dh_buf_.Get(), 0, S * E);
            auto dA_uav = makeUAV(lnf_y_buf_.Get(), 0, S * E);
            mmBwdA(S, E, E, nullptr, B_srv.Get(), dC_srv.Get(), dA_uav.Get());
        }
        // lnf_y_buf_ = d_attn_out [S, E]

        // 8l. Attention backward (3-pass per head, writes to d_qkv_buf_ [S,3E])
        zero(d_qkv_buf_.Get(), 0, S * 3 * E);
        for (uint32_t h = 0; h < H; ++h) {
            // Zero per-head temps
            zero(dP_tmp_buf_.Get(), 0, S * S);
            zero(dot_row_buf_.Get(), 0, S);

            // P_head_offset = h*S*S: shader reads P[P_head_offset + i*S + j] from full P_buf_[l].
            // Intel HD 4600 SRV.FirstElement bug: can't use makeSRV(P_buf_[l], h*S*S, S*S).
            struct { uint32_t S, D, E, h; float scale; uint32_t P_head_offset; uint32_t pad[2]; }
                abp{S, D, E, h, scale, h * S * S};
            setCB(&abp, 32);

            auto qkv_srv  = makeSRV(qkv_buf_[l].Get(), 0, S * 3 * E);
            auto P_srv    = makeSRV(P_buf_[l].Get(),    0, H * S * S);  // full; head via CB
            auto dout_srv = makeSRV(lnf_y_buf_.Get(),  0, S * E);
            auto dqkv_uav = makeUAV(d_qkv_buf_.Get(),  0,                S * 3 * E);
            auto dP_uav   = makeUAV(dP_tmp_buf_.Get(), 0,                S * S);
            auto dr_uav   = makeUAV(dot_row_buf_.Get(), 0,               S);

            // Pass 1: dV and dP_tmp
            ctx->CSSetShader(cs_attn_bwd_dvdp_.Get(), nullptr, 0);
            {
                ID3D11ShaderResourceView* srvs[3] = { qkv_srv.Get(), P_srv.Get(), dout_srv.Get() };
                ctx->CSSetShaderResources(0, 3, srvs);
                ID3D11UnorderedAccessView* uvs[3] = { dqkv_uav.Get(), dP_uav.Get(), dr_uav.Get() };
                ctx->CSSetUnorderedAccessViews(0, 3, uvs, nullptr);
            }
            ctx->Dispatch(S, 1, 1); clearViews(3, 3);

            // Pass 2: dQ and dot_row
            setCB(&abp, 32);
            ctx->CSSetShader(cs_attn_bwd_dq_.Get(), nullptr, 0);
            {
                ID3D11ShaderResourceView* srvs[3] = { qkv_srv.Get(), P_srv.Get(), dout_srv.Get() };
                ctx->CSSetShaderResources(0, 3, srvs);
                ID3D11UnorderedAccessView* uvs[3] = { dqkv_uav.Get(), dP_uav.Get(), dr_uav.Get() };
                ctx->CSSetUnorderedAccessViews(0, 3, uvs, nullptr);
            }
            ctx->Dispatch(S, 1, 1); clearViews(3, 3);

            // Pass 3: dK
            setCB(&abp, 32);
            ctx->CSSetShader(cs_attn_bwd_dk_.Get(), nullptr, 0);
            {
                ID3D11ShaderResourceView* srvs[3] = { qkv_srv.Get(), P_srv.Get(), dout_srv.Get() };
                ctx->CSSetShaderResources(0, 3, srvs);
                ID3D11UnorderedAccessView* uvs[3] = { dqkv_uav.Get(), dP_uav.Get(), dr_uav.Get() };
                ctx->CSSetUnorderedAccessViews(0, 3, uvs, nullptr);
            }
            ctx->Dispatch(S, 1, 1); clearViews(3, 3);
        }
        // d_qkv_buf_ [S, 3E] now holds dQ, dK, dV (interleaved)

        // 8m. d_c_attn_w += ln1_y[l].T @ d_qkv_buf_   (dB: M=S, K=E, N=3E)
        {
            auto A_srv  = makeSRV(ln1_y_buf_[l].Get(), 0, S * E);
            auto dC_srv = makeSRV(d_qkv_buf_.Get(),  0,     S * 3 * E);
            auto dB_uav = makeUAV(params_[param_idx_.at(pfx + "attn.c_attn.weight")].g_buf.Get(),
                                  0, E * 3 * E);
            mmBwdB(S, E, 3 * E, A_srv.Get(), dC_srv.Get(), dB_uav.Get());
        }

        // 8n. d_c_attn_b += sum_rows(d_qkv_buf_)
        {
            auto dC_srv = makeSRV(d_qkv_buf_.Get(), 0, S * 3 * E);
            auto db_uav = makeUAV(params_[param_idx_.at(pfx + "attn.c_attn.bias")].g_buf.Get(),
                                  0, 3 * E);
            biasBwd(S, 3 * E, dC_srv.Get(), db_uav.Get());
        }

        // 8o. d_ln1_y = d_qkv_buf_ @ c_attn_w.T   (dA: M=S, K=E, N=3E)
        //     → lnf_y_buf_ (reused again as temp)
        zero(lnf_y_buf_.Get(), 0, S * E);
        {
            auto B_srv  = makeSRV(params_[param_idx_.at(pfx + "attn.c_attn.weight")].w_buf.Get(),
                                  0, E * 3 * E);
            auto dC_srv = makeSRV(d_qkv_buf_.Get(), 0, S * 3 * E);
            auto dA_uav = makeUAV(lnf_y_buf_.Get(), 0, S * E);
            mmBwdA(S, E, 3 * E, nullptr, B_srv.Get(), dC_srv.Get(), dA_uav.Get());
        }
        // lnf_y_buf_ = d_ln1_y [S, E]

        // 8p. LN1 backward: dout = d_ln1_y (in lnf_y_buf_), dx += to dh_buf_
        {
            struct { uint32_t E, S; float eps; uint32_t pad; } p{E, S, eps};
            setCB(&p, 16);
            auto sv0 = makeSRV(xhat_ln1_buf_[l].Get(), 0, S * E);  // xhat
            auto sv1 = wSRV(pfx + "ln_1.weight");                   // gamma
            auto sv2 = makeSRV(lnf_y_buf_.Get(),       0, S * E);  // dout = d_ln1_y (temp)
            auto sv3 = makeSRV(inv_std_ln1_[l].Get(),  0, S);       // inv_std
            auto uv1 = makeUAV(params_[param_idx_.at(pfx + "ln_1.weight")].g_buf.Get(), 0, E);
            auto uv2 = makeUAV(params_[param_idx_.at(pfx + "ln_1.bias")  ].g_buf.Get(), 0, E);

            // Pass 1: dx only
            ctx->CSSetShader(cs_lnorm_bwd_.Get(), nullptr, 0);
            ID3D11ShaderResourceView* srvs[4] = { sv0.Get(), sv1, sv2.Get(), sv3.Get() };
            ctx->CSSetShaderResources(0, 4, srvs);
            auto uv0 = makeUAV(dh_buf_.Get(), 0, S * E);  // dx += (d_h[l])
            ID3D11UnorderedAccessView* uvs1[3] = { uv0.Get(), nullptr, nullptr };
            ctx->CSSetUnorderedAccessViews(0, 3, uvs1, nullptr);
            ctx->Dispatch(S, 1, 1); clearViews(3, 4);

            // Pass 2: dgamma/dbeta, race-free
            ctx->CSSetShader(cs_lnorm_bwd_params_.Get(), nullptr, 0);
            ID3D11ShaderResourceView* srvsp[4] = { sv0.Get(), nullptr, sv2.Get(), nullptr };
            ctx->CSSetShaderResources(0, 4, srvsp);
            ID3D11UnorderedAccessView* uvs2[3] = { nullptr, uv1.Get(), uv2.Get() };
            ctx->CSSetUnorderedAccessViews(0, 3, uvs2, nullptr);
            ctx->Dispatch((E + 255) / 256, 1, 1); clearViews(3, 4);
        }
        // dh_buf_ = d_h[l]  (gradient into layer l input)
    } // end layer backward loop

    // 9. Embedding backward: dh_buf_ → d_wte (+=), d_wpe (+=)
    {
        struct { uint32_t S, E, pad[2]; } p{S, E};
        setCB(&p, 16);
        ctx->CSSetShader(cs_embed_bwd_.Get(), nullptr, 0);
        auto tsrv = makeSRVi(tokens_buf_.Get(), 0, S);
        auto dh_srv = makeSRV(dh_buf_.Get(), 0, S * E);
        ID3D11ShaderResourceView* srvs[2] = { tsrv.Get(), dh_srv.Get() };
        ctx->CSSetShaderResources(0, 2, srvs);
        auto uv0 = makeUAV(params_[param_idx_.at("transformer.wte.weight")].g_buf.Get(),
                            0, params_[param_idx_.at("transformer.wte.weight")].numel);
        auto uv1 = makeUAV(params_[param_idx_.at("transformer.wpe.weight")].g_buf.Get(),
                            0, params_[param_idx_.at("transformer.wpe.weight")].numel);
        ID3D11UnorderedAccessView* uvs[2] = { uv0.Get(), uv1.Get() };
        ctx->CSSetUnorderedAccessViews(0, 2, uvs, nullptr);
        ctx->Dispatch(E, 1, 1); clearViews(2, 2);
    }

    if (fullseq_) {
        auto lv = readbackBuffer(loss_all_.Get(), S);   // S per-position losses
        float sum = 0.f, ts = 0.f, ps = 0.f; uint32_t tn = 0, pn = 0;
        for (uint32_t s = 0; s < S; ++s) {
            sum += lv[s];
            if ((uint32_t)seq[s] >= 50257u) { ps += lv[s]; ++pn; }   // glyph/phase (ext vocab)
            else                            { ts += lv[s]; ++tn; }   // GPT-2 text
        }
        std::cerr << "[split] text=" << (tn ? ts/tn : 0.f)
                  << " phase=" << (pn ? ps/pn : 0.f)
                  << " total=" << sum/(float)S << "\n";
        return sum / (float)S;
    }
    return readbackLoss();
}


// ── GPU Adam optimizer ────────────────────────────────────────────────────────

void GPT2Trainer::dispatchAdam(AdamParam& p, float bc1, float bc2, bool upload_cpu_grads) {
    auto* ctx = d11_->rawCtx();
    auto* dev = d11_->rawDevice();

    // Upload CPU gradients → GPU g_buf (Phase 1 path only)
    if (upload_cpu_grads) {
        D3D11_BUFFER_DESC sd{};
        sd.ByteWidth      = p.numel * sizeof(float);
        sd.Usage          = D3D11_USAGE_STAGING;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        ComPtr<ID3D11Buffer> staging;
        dev->CreateBuffer(&sd, nullptr, &staging);
        D3D11_MAPPED_SUBRESOURCE ms{};
        ctx->Map(staging.Get(), 0, D3D11_MAP_WRITE, 0, &ms);
        std::memcpy(ms.pData, p.cpu_g.data(), p.numel * sizeof(float));
        ctx->Unmap(staging.Get(), 0);
        ctx->CopyResource(p.g_buf.Get(), staging.Get());
    }

    // 2D dispatch to stay within D3D11's 65535 per-dimension limit.
    // stride_x = X dimension; total groups = X * Y >= ceil(numel/256).
    const uint32_t tg = (p.numel + 255) / 256;
    const uint32_t disp_x = std::min(tg, (uint32_t)32768);  // well under D3D11's 65535 limit
    const uint32_t disp_y = (tg + disp_x - 1) / disp_x;

    // Update cbuffer: lr, beta1, beta2, eps, wd, bc1, bc2, numel, stride_x, pad[3]
    struct AdamCB { float lr, beta1, beta2, eps, wd, bc1, bc2;
                    uint32_t numel, stride_x; float grad_clip; uint32_t pad[2]; };
    // Linear LR warmup over the first warmup_steps_ steps (train-from-zero stability).
    float lr_eff = cfg_.lr;
    if (warmup_steps_ > 0 && step_ < warmup_steps_)
        lr_eff = cfg_.lr * (float)(step_ + 1) / (float)warmup_steps_;
    AdamCB cb{ lr_eff, cfg_.beta1, cfg_.beta2, cfg_.eps, cfg_.weight_decay,
               bc1, bc2, p.numel, disp_x, grad_clip_ };
    D3D11_MAPPED_SUBRESOURCE ms{};
    ctx->Map(adam_cb_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &ms);
    std::memcpy(ms.pData, &cb, sizeof(cb));
    ctx->Unmap(adam_cb_.Get(), 0);

    // Bind: cbuffer + 4 UAVs (w, g, m, v)
    auto makeUAV = [&](ID3D11Buffer* buf) -> ComPtr<ID3D11UnorderedAccessView> {
        D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};
        ud.Format              = DXGI_FORMAT_UNKNOWN;
        ud.ViewDimension       = D3D11_UAV_DIMENSION_BUFFER;
        ud.Buffer.NumElements  = p.numel;
        ComPtr<ID3D11UnorderedAccessView> uav;
        dev->CreateUnorderedAccessView(buf, &ud, &uav);
        return uav;
    };
    auto uav_w = makeUAV(p.w_buf.Get());
    auto uav_g = makeUAV(p.g_buf.Get());
    auto uav_m = makeUAV(p.m_buf.Get());
    auto uav_v = makeUAV(p.v_buf.Get());

    ID3D11Buffer* cbs[] = { adam_cb_.Get() };
    ctx->CSSetConstantBuffers(0, 1, cbs);
    ID3D11UnorderedAccessView* uavs[] = { uav_w.Get(), uav_g.Get(), uav_m.Get(), uav_v.Get() };
    ctx->CSSetUnorderedAccessViews(0, 4, uavs, nullptr);
    ctx->CSSetShader(cs_adam_.Get(), nullptr, 0);
    ctx->Dispatch(disp_x, disp_y, 1);

    // Sync after every dispatch using Map-based readback (EVENT queries unreliable on
    // Intel HD 4600 CS). CopyResource + Map stalls the CPU until the GPU is idle,
    // preventing the TDR that fires when 317 dispatches queue without a sync.
    if (loss_staging_ && loss_buf_) {   // GPU-fwd path only; CPU path syncs via weight readback below
        ctx->CopyResource(loss_staging_.Get(), loss_buf_.Get());
        D3D11_MAPPED_SUBRESOURCE mr{};
        if (SUCCEEDED(ctx->Map(loss_staging_.Get(), 0, D3D11_MAP_READ, 0, &mr)))
            ctx->Unmap(loss_staging_.Get(), 0);
    }

    // Readback updated weights → cpu_w_owned (needed for CPU forward pass in Phase 1)
    if (upload_cpu_grads) {
        D3D11_BUFFER_DESC sd{};
        sd.ByteWidth      = p.numel * sizeof(float);
        sd.Usage          = D3D11_USAGE_STAGING;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Buffer> staging;
        dev->CreateBuffer(&sd, nullptr, &staging);
        ctx->CopyResource(staging.Get(), p.w_buf.Get());
        D3D11_MAPPED_SUBRESOURCE mr{};
        ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mr);
        std::memcpy(p.cpu_w_owned.data(), mr.pData, p.numel * sizeof(float));
        ctx->Unmap(staging.Get(), 0);
    }
}

void GPT2Trainer::adamStepGPU(bool upload_cpu_grads) {
    step_++;
    beta1_t_ *= cfg_.beta1;
    beta2_t_ *= cfg_.beta2;
    const float bc1 = 1.f / (1.f - beta1_t_);
    const float bc2 = 1.f / (1.f - beta2_t_);

    auto* ctx = d11_->rawCtx();
    auto* dev = d11_->rawDevice();

    // Pre-Adam device check (no GPU sync needed — reflects already-known state)
    {
        HRESULT dr = dev->GetDeviceRemovedReason();
        if (dr != S_OK)
            std::cerr << "[adam] device already lost BEFORE Adam step=" << step_
                      << " reason=0x" << std::hex << (uint32_t)dr << std::dec << "\n";
    }

    for (auto& p : params_) {
        if (upload_cpu_grads) {
            // Phase 1: skip params with no CPU gradient
            bool has_grad = false;
            for (float g : p.cpu_g) { if (g != 0.f) { has_grad = true; break; } }
            if (!has_grad) continue;
        }
        dispatchAdam(p, bc1, bc2, upload_cpu_grads);
    }

    // Unbind UAVs
    ID3D11UnorderedAccessView* nullUAVs[4] = {};
    ctx->CSSetUnorderedAccessViews(0, 4, nullUAVs, nullptr);

    // Post-Adam GPU event flush — forces any TDR or GPU error to surface
    {
        ID3D11Query* q = nullptr;
        D3D11_QUERY_DESC qd{ D3D11_QUERY_EVENT, 0 };
        HRESULT cqhr = dev->CreateQuery(&qd, &q);
        if (SUCCEEDED(cqhr) && q) {
            ctx->End(q);
            BOOL done = FALSE;
            HRESULT hr;
            while ((hr = ctx->GetData(q, &done, sizeof(done), 0)) == S_FALSE) {}
            q->Release();
            if (FAILED(hr) || !done) {
                HRESULT dr = dev->GetDeviceRemovedReason();
                std::cerr << "[adam] GPU event FAILED step=" << step_
                          << " hr=0x" << std::hex << (uint32_t)hr
                          << " device_removed=0x" << (uint32_t)dr << std::dec << "\n";
            } else {
                if (step_ <= 3)
                    std::cerr << "[adam] GPU event OK step=" << step_ << "\n";
            }
        } else {
            HRESULT dr = dev->GetDeviceRemovedReason();
            std::cerr << "[adam] CreateQuery failed step=" << step_
                      << " cqhr=0x" << std::hex << (uint32_t)cqhr
                      << " device_removed=0x" << (uint32_t)dr << std::dec << "\n";
        }
    }
}

void GPT2Trainer::syncWeightsToCPU() {
    auto* ctx = d11_->rawCtx();
    auto* dev = d11_->rawDevice();

    // Free all optimizer buffers (g, m, v) to reclaim ~4 GB of GPU memory before
    // allocating staging buffers for weight readback.  m_buf and v_buf are
    // reallocated as zeros below — one momentum-reset step is acceptable.
    for (auto& p : params_) { p.g_buf.Reset(); p.m_buf.Reset(); p.v_buf.Reset(); }

    // GPU sync: wait for all work to finish and free deferred resources
    {
        ID3D11Query* qEvt = nullptr;
        D3D11_QUERY_DESC qd{ D3D11_QUERY_EVENT, 0 };
        dev->CreateQuery(&qd, &qEvt);
        if (qEvt) {
            ctx->End(qEvt);
            BOOL done = FALSE;
            while (S_OK != ctx->GetData(qEvt, &done, sizeof(done), 0) || !done) {}
            qEvt->Release();
        }
    }

    // Probe: can we allocate even 4 bytes?
    {
        D3D11_BUFFER_DESC sd{}; sd.ByteWidth = 4; sd.Usage = D3D11_USAGE_STAGING;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Buffer> probe;
        HRESULT hr = dev->CreateBuffer(&sd, nullptr, &probe);
        std::cerr << "[sync] 4-byte probe hr=0x" << std::hex << (uint32_t)hr
                  << (probe ? " ok" : " NULL") << std::dec << "\n";
    }

    // Sort by size: reuse one staging buffer per unique param size.
    std::vector<uint32_t> order(params_.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return params_[a].numel < params_[b].numel;
    });

    ComPtr<ID3D11Buffer> staging;
    uint32_t staging_numel = 0;
    uint32_t synced = 0, skipped = 0;

    for (uint32_t idx : order) {
        auto& p = params_[idx];
        if (p.cpu_w_owned.empty() || !p.w_buf) continue;

        if (p.numel != staging_numel) {
            staging.Reset();
            D3D11_BUFFER_DESC sd{};
            sd.ByteWidth = p.numel * sizeof(float);
            sd.Usage = D3D11_USAGE_STAGING;
            sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            // No MiscFlags/StructureByteStride — plain staging matches dispatchAdam readback
            // and avoids E_INVALIDARG from STRUCTURED+STAGING on some drivers.
            if (FAILED(dev->CreateBuffer(&sd, nullptr, &staging)) || !staging) {
                ++skipped; staging_numel = 0; continue;
            }
            staging_numel = p.numel;
        }

        ctx->CopyResource(staging.Get(), p.w_buf.Get());
        D3D11_MAPPED_SUBRESOURCE mr{};
        if (SUCCEEDED(ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mr)) && mr.pData) {
            std::memcpy(p.cpu_w_owned.data(), mr.pData, p.numel * sizeof(float));
            ctx->Unmap(staging.Get(), 0);
            ++synced;
        }
    }
    staging.Reset();

    // Reallocate zeroed g/m/v buffers so next training step works.
    // Momentum state (m, v) is reset to zero — this causes one sub-optimal Adam step.
    std::vector<float> zeros;
    for (auto& p : params_) {
        if (zeros.size() < p.numel) zeros.assign(p.numel, 0.f);
        p.g_buf = createAndUpload(zeros.data(), p.numel);
        p.m_buf = createAndUpload(zeros.data(), p.numel);
        p.v_buf = createAndUpload(zeros.data(), p.numel);
    }

    std::cerr << "[trainer] syncWeightsToCPU: synced=" << synced
              << " skipped=" << skipped << "\n";
}

// ── train step ────────────────────────────────────────────────────────────────

float GPT2Trainer::train_step(const std::vector<std::vector<int32_t>>& batch) {
    float total_loss = 0.f;
    uint32_t count = 0;

    if (cfg_.use_gpu_fwd) {
        // Phase 3: full GPU forward+backward
        zeroGPUGrads();
        const float inv_batch = batch.empty() ? 1.f : 1.f / (float)batch.size();
        for (const auto& seq : batch) {
            if (seq.size() < 2) continue;
            float loss = gpuForwardBackward(seq, inv_batch);
            if (loss >= 0.f) { total_loss += loss; ++count; }
        }
        if (count > 0) total_loss /= count;

        // Adaptive gradient gravity (KuhulPhysics): real loss + REAL gradNorm (representative
        // layer ln_f.weight, read BEFORE Adam zeros grads) drive grad_clip_. logitMax=0 (Rule 2 off).
        if (adaptive_clip_ && physics_) {
            float gradNorm = 1.0f;
            auto it = param_idx_.find("transformer.ln_f.weight");
            if (it != param_idx_.end()) {
                auto& gp = params_[it->second];
                auto gv = readbackBuffer(gp.g_buf.Get(), gp.numel);
                double ss = 0.0; for (uint32_t i = 0; i < gp.numel; ++i) ss += (double)gv[i] * gv[i];
                gradNorm = (float)std::sqrt(ss);
            }
            physics_->Observe(total_loss, gradNorm, 0.0f);
            grad_clip_ = physics_->GetGradClipNorm();
        }

        // DBG: scan all g_buf for NaN/Inf before Adam (env-gated: GPT2_DBG, first step only)
        if (step_ == 0 && std::getenv("GPT2_DBG")) {  // identifies which param blows up
            for (auto& p : params_) {
                auto g = readbackBuffer(p.g_buf.Get(), p.numel);
                float gmax = 0.f; bool bad = false;
                for (uint32_t i = 0; i < p.numel; ++i) {
                    if (!std::isfinite(g[i])) { bad = true; break; }
                    gmax = std::max(gmax, std::abs(g[i]));
                }
                if (bad || gmax > 1e6f)
                    std::cerr << "[grad-dbg] " << p.name
                              << ": max=" << gmax << (bad?" INF/NAN":"") << "\n";
            }
        }

        adamStepGPU(false);  // grads already in g_buf, skip CPU upload
    } else {
        // Phase 1: CPU forward+backward, GPU Adam
        zeroCPUGrads();
        for (const auto& seq : batch) {
            if (seq.size() < 2) continue;
            float loss = forwardBackwardCPU(seq, true);
            if (loss >= 0.f) { total_loss += loss; ++count; }
        }
        if (count > 1) {
            const float inv = 1.f / count;
            for (auto& p : params_)
                for (float& g : p.cpu_g) g *= inv;
            total_loss /= count;
        }
        adamStepGPU(true);  // upload grads → GPU Adam → readback weights
    }

    std::cerr << "[trainer] step=" << step_ << " loss=" << total_loss << "\n";
    if (step_ % cfg_.log_every == 0)
        std::cerr << "[trainer] --- checkpoint step=" << step_ << " ---\n";
    if (step_ % cfg_.save_every == 0)
        save();

    return total_loss;
}

// ── training loop ─────────────────────────────────────────────────────────────

void GPT2Trainer::train() {
    std::ifstream df(cfg_.data_path, std::ios::binary);
    if (!df) { std::cerr << "[trainer] cannot open data: " << cfg_.data_path << "\n"; return; }

    // Auto-detect format:
    //   Headed: uint32 n_seq, uint32 seq_len, then n_seq*seq_len int32 tokens (tokenize_transitions --pack)
    //   Flat:   raw int32 token stream, chunked here by block_size (tokenize_transitions without --pack)
    df.seekg(0, std::ios::end);
    uint64_t file_bytes = (uint64_t)df.tellg();
    df.seekg(0, std::ios::beg);

    uint32_t n_seq = 0, seq_len = 0;
    df.read((char*)&n_seq, 4);
    df.read((char*)&seq_len, 4);
    const bool headed = (n_seq > 0 && seq_len > 1 && seq_len <= 8192 &&
                         (uint64_t)n_seq * seq_len * 4 + 8 == file_bytes);

    std::vector<std::vector<int32_t>> all_seqs;
    if (headed) {
        all_seqs.assign(n_seq, std::vector<int32_t>(seq_len));
        for (auto& seq : all_seqs) df.read((char*)seq.data(), seq_len * sizeof(int32_t));
        df.close();
        std::cerr << "[trainer] data: " << n_seq << " sequences x " << seq_len
                  << " tokens (headed format)\n";
    } else {
        // Flat format: seek back to 0, read everything, chunk by block_size
        df.seekg(0);
        uint32_t total_toks = (uint32_t)(file_bytes / sizeof(int32_t));
        std::vector<int32_t> flat(total_toks);
        df.read((char*)flat.data(), total_toks * sizeof(int32_t));
        df.close();
        uint32_t chunk = cfg_.block_size > 0 ? cfg_.block_size : 128;
        n_seq  = total_toks / chunk;  // drop incomplete final chunk
        seq_len = chunk;
        all_seqs.resize(n_seq, std::vector<int32_t>(chunk));
        for (uint32_t i = 0; i < n_seq; ++i)
            std::copy(flat.data() + i * chunk, flat.data() + (i + 1) * chunk, all_seqs[i].begin());
        std::cerr << "[trainer] data: " << total_toks << " tokens → " << n_seq
                  << " chunks x " << chunk << " (flat format)\n";
    }

    if (cfg_.completion_only)
        std::cerr << "[trainer] completion-only loss masking: <INSTRUCT>...</INSTRUCT> spans only\n";

    // Fixed-batch overfit diagnostic (GPT2_OVERFIT=1): reuse ONE batch every step and
    // print its loss each step. Loss falling ⇒ gradients are correct-direction (learns).
    if (std::getenv("GPT2_OVERFIT")) {
        std::vector<std::vector<int32_t>> fixed;
        for (uint32_t i = 0; i < cfg_.batch_size && i < n_seq; ++i) {
            auto seq = all_seqs[i];
            if (cfg_.block_size > 0 && seq.size() > cfg_.block_size) seq.resize(cfg_.block_size);
            fixed.push_back(std::move(seq));
        }
        std::cerr << "[overfit] fixed batch of " << fixed.size() << " seqs, "
                  << cfg_.max_steps << " steps\n";
        for (int s = 0; s < (int)cfg_.max_steps; ++s) {
            float loss = train_step(fixed);
            std::cerr << "[overfit] step=" << s << " loss=" << loss << "\n";
        }
        save();
        return;
    }

    std::mt19937 rng(42);
    std::vector<uint32_t> idx(n_seq);
    std::iota(idx.begin(), idx.end(), 0);

    if (cfg_.no_shuffle)
        std::cerr << "[trainer] shuffle disabled — training in dataset order (curriculum)\n";

    float running_loss = 0.f;
    int   running_n    = 0;

    for (int s = 0; s < (int)cfg_.max_steps; ++s) {
        if (!cfg_.no_shuffle && s % (int)n_seq == 0) std::shuffle(idx.begin(), idx.end(), rng);
        std::vector<std::vector<int32_t>> batch;
        for (uint32_t i = 0; i < cfg_.batch_size && i < n_seq; ++i) {
            auto seq = all_seqs[idx[(s * cfg_.batch_size + i) % n_seq]];
            if (cfg_.block_size > 0 && seq.size() > cfg_.block_size)
                seq.resize(cfg_.block_size);
            // completion-only: skip sequence if no INSTRUCT tokens
            if (cfg_.completion_only) {
                auto mask = buildCompletionMask(seq, true);
                float has_instruct = 0.f;
                for (float m : mask) has_instruct += m;
                if (has_instruct < 0.5f) continue;  // all-user sequence — skip
            }
            batch.push_back(std::move(seq));
        }
        if (batch.empty()) continue;

        float loss = train_step(batch);
        running_loss += loss;
        running_n++;

        bool do_log  = (running_n > 0 && s % (int)cfg_.log_every == 0);
        bool do_save = (cfg_.save_every > 0 && s > 0 && s % (int)cfg_.save_every == 0);

        if (do_log) {
            float avg = running_loss / running_n;
            std::cerr << "[train] step=" << s << "/" << cfg_.max_steps
                      << " loss=" << avg << "\n";
            running_loss = 0.f; running_n = 0;
        }
        if (do_save) {
            std::string ckpt = cfg_.output_path;
            auto ext = ckpt.rfind(".safetensors");
            if (ext != std::string::npos) ckpt.replace(ext, 12, "_step" + std::to_string(s) + ".safetensors");
            save(ckpt);
            std::cerr << "[train] checkpoint: " << ckpt << "\n";
        }
    }
    save();
}

// ── save ──────────────────────────────────────────────────────────────────────

bool GPT2Trainer::save(const std::string& override_path) {
    const std::string out_path = override_path.empty() ? cfg_.output_path : override_path;
    if (out_path.empty()) { std::cerr << "[trainer] no output path\n"; return false; }

    // Phase 3: weights live on GPU; sync to CPU before saving
    if (cfg_.use_gpu_fwd) syncWeightsToCPU();

    json header;
    std::vector<uint8_t> blob;

    for (auto& p : params_) {
        if (p.cpu_w_owned.empty()) continue;
        const float* src = p.cpu_w_owned.data();

        const uint64_t start = blob.size();
        const uint64_t nbytes = p.numel * sizeof(float);
        blob.resize(blob.size() + nbytes);
        std::memcpy(blob.data() + start, src, nbytes);
        header[p.name] = {
            {"dtype", "F32"},
            {"shape", p.shape},
            {"data_offsets", {start, start + nbytes}}
        };
    }

    std::string json_str = header.dump();
    while (json_str.size() % 8 != 0) json_str += ' ';
    uint64_t hlen = json_str.size();

    std::ofstream f(out_path, std::ios::binary);
    f.write((char*)&hlen, 8);
    f.write(json_str.data(), hlen);
    f.write((char*)blob.data(), blob.size());
    f.close();

    std::cerr << "[trainer] saved: " << out_path
              << " (" << (8 + hlen + blob.size()) / (1<<20) << " MB)\n";
    return true;
}
