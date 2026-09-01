// xshard_adapt.cpp — D3D11 host wrapper for xshard_adapt_fold.cso
//
// Safe default: dry-run. Pass --apply to commit adapted F32 shards in-place.

#include "d3d11_engine.h"
#include "xshard.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

struct AdaptParams {
    float lr = 1.0e-5f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1.0e-8f;
    float weight_decay = 0.0f;
    float bias_corr1 = 1.0f / (1.0f - 0.9f);
    float bias_corr2 = 1.0f / (1.0f - 0.999f);
    uint32_t numel = 0;
    uint32_t stride_x = 1;
    float grad_clip = 1.0f;
    float phase_angle = 0.0f;
    uint32_t fold_id = 0;
    float fold_gate = 1.0f;
    uint32_t update_mode = 0; // 0=Adam, 1=SGD
    uint32_t pad[2] = {0, 0};
};
static_assert(sizeof(AdaptParams) == 64, "AdaptParams must match HLSL cbuffer packing");

struct Options {
    std::string xshard_path;
    std::string shader_path = "../shaders/xshard_adapt_fold.cso";
    std::string fold_filter;
    std::string grad_file;
    std::string grad_dir;
    std::string grad_xshard;
    int max_shards = 1;
    float lr = 1.0e-5f;
    float grad_scale = 0.0f;
    float weight_decay = 0.0f;
    float fold_gate = 1.0f;
    std::string ledger_path;
    bool apply = false;
    bool force_warp = false;
    bool verbose = false;
    uint32_t update_mode = 0;
};

static void usage() {
    std::fprintf(stderr,
        "usage: xshard_adapt <file.xshard> [options]\n"
        "  --apply                 commit adapted shards in-place (default is dry-run)\n"
        "  --shader <path>          compiled xshard_adapt_fold.cso (default ../shaders/xshard_adapt_fold.cso)\n"
        "  --fold <phase>           only process one fold (Pop/Wo/Yax/Sek/Chen/Xul)\n"
        "  --max-shards <n>         max F32 shards to process (default 1)\n"
        "  --lr <v>                 learning rate (default 1e-5)\n"
        "  --grad-file <path>       F32 gradient file for a single processed shard\n"
        "  --grad-dir <dir>         F32 gradient files by seq/id: seq_N.f32 or <id>.grad.f32\n"
        "  --grad-xshard <path>     gradient XSHARD/1 container matched by seq/id/tensor\n"
        "  --grad-scale <v>         synthetic probe grad = weight * scale (default 0)\n"
        "  --weight-decay <v>       weight decay folded into gradient (default 0)\n"
        "  --fold-gate <v>          phase/fold gate multiplier (default 1)\n"
        "  --ledger <path>          JSONL checkpoint ledger (default <file>.adapt.jsonl)\n"
        "  --sgd                   use SGD instead of Adam\n"
        "  --warp                  force WARP device\n"
        "  --verbose               print adapter/device details\n"
    );
}

static bool parse_float(const char* s, float& out) {
    if (!s) return false;
    char* end = nullptr;
    const float v = std::strtof(s, &end);
    if (!end || *end != '\0' || !std::isfinite(v)) return false;
    out = v;
    return true;
}

static bool parse_int(const char* s, int& out) {
    if (!s) return false;
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0' || v < 1) return false;
    out = static_cast<int>(v);
    return true;
}

static bool parse_args(int argc, char** argv, Options& opt) {
    if (argc < 2) return false;
    opt.xshard_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--apply") == 0) opt.apply = true;
        else if (std::strcmp(a, "--warp") == 0) opt.force_warp = true;
        else if (std::strcmp(a, "--verbose") == 0) opt.verbose = true;
        else if (std::strcmp(a, "--sgd") == 0) opt.update_mode = 1;
        else if (std::strcmp(a, "--shader") == 0 && i + 1 < argc) opt.shader_path = argv[++i];
        else if (std::strcmp(a, "--ledger") == 0 && i + 1 < argc) opt.ledger_path = argv[++i];
        else if (std::strcmp(a, "--grad-file") == 0 && i + 1 < argc) opt.grad_file = argv[++i];
        else if (std::strcmp(a, "--grad-dir") == 0 && i + 1 < argc) opt.grad_dir = argv[++i];
        else if (std::strcmp(a, "--grad-xshard") == 0 && i + 1 < argc) opt.grad_xshard = argv[++i];
        else if (std::strcmp(a, "--fold") == 0 && i + 1 < argc) opt.fold_filter = argv[++i];
        else if (std::strcmp(a, "--max-shards") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], opt.max_shards)) return false;
        } else if (std::strcmp(a, "--lr") == 0 && i + 1 < argc) {
            if (!parse_float(argv[++i], opt.lr)) return false;
        } else if (std::strcmp(a, "--grad-scale") == 0 && i + 1 < argc) {
            if (!parse_float(argv[++i], opt.grad_scale)) return false;
        } else if (std::strcmp(a, "--weight-decay") == 0 && i + 1 < argc) {
            if (!parse_float(argv[++i], opt.weight_decay)) return false;
        } else if (std::strcmp(a, "--fold-gate") == 0 && i + 1 < argc) {
            if (!parse_float(argv[++i], opt.fold_gate)) return false;
        } else {
            return false;
        }
    }
    const int gradient_sources = (opt.grad_file.empty() ? 0 : 1) +
                                 (opt.grad_dir.empty() ? 0 : 1) +
                                 (opt.grad_xshard.empty() ? 0 : 1);
    if (gradient_sources > 1) return false;
    if (!opt.grad_file.empty() && opt.max_shards != 1) return false;
    if (opt.ledger_path.empty()) opt.ledger_path = opt.xshard_path + ".adapt.jsonl";
    return !opt.xshard_path.empty();
}

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    const auto n = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(n));
    if (!data.empty()) f.read(reinterpret_cast<char*>(data.data()), data.size());
    return data;
}

static bool read_float_file(const std::string& path, std::vector<float>& out, size_t expected_count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const auto n = f.tellg();
    if (n < 0 || static_cast<size_t>(n) != expected_count * sizeof(float)) return false;
    f.seekg(0, std::ios::beg);
    out.resize(expected_count);
    if (!out.empty()) f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(expected_count * sizeof(float)));
    return static_cast<bool>(f);
}

static std::string join_path(const std::string& dir, const std::string& leaf) {
    if (dir.empty()) return leaf;
    const char last = dir[dir.size() - 1];
    if (last == '\\' || last == '/') return dir + leaf;
    return dir + "\\" + leaf;
}

static std::string safe_id_filename(std::string s) {
    for (char& c : s) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.';
        if (!ok) c = '_';
    }
    return s;
}

static bool load_gradients_for_shard(
    const Options& opt,
    const XShardFile* grad_xshard,
    const XShardRecord& shard,
    std::vector<float>& grads,
    const std::vector<float>& weights,
    std::string& source) {

    if (!opt.grad_file.empty()) {
        source = opt.grad_file;
        return read_float_file(opt.grad_file, grads, weights.size());
    }

    if (grad_xshard) {
        const XShardRecord* g = nullptr;
        if (shard.seq >= 0 && shard.seq < grad_xshard->n_shards()) {
            const auto& by_seq = grad_xshard->shards()[shard.seq];
            if (by_seq.dtype == "F32" && by_seq.nbytes == static_cast<int64_t>(weights.size() * sizeof(float))) {
                g = &by_seq;
            }
        }
        if (!g && !shard.id.empty()) {
            const auto* by_id = grad_xshard->find_by_id(shard.id);
            if (by_id && by_id->dtype == "F32" && by_id->nbytes == static_cast<int64_t>(weights.size() * sizeof(float))) {
                g = by_id;
            }
        }
        if (!g && !shard.tensor_name.empty()) {
            const auto* by_tensor = grad_xshard->find_by_tensor(shard.tensor_name, shard.shard_index);
            if (by_tensor && by_tensor->dtype == "F32" && by_tensor->nbytes == static_cast<int64_t>(weights.size() * sizeof(float))) {
                g = by_tensor;
            }
        }
        source = opt.grad_xshard;
        if (!g) return false;
        grads.resize(weights.size());
        return grad_xshard->read_shard(g->seq, grads.data());
    }

    if (!opt.grad_dir.empty()) {
        char seq_name[64];
        std::snprintf(seq_name, sizeof(seq_name), "seq_%d.f32", shard.seq);
        const std::string candidates[] = {
            join_path(opt.grad_dir, seq_name),
            join_path(opt.grad_dir, safe_id_filename(shard.id) + ".grad.f32"),
            join_path(opt.grad_dir, safe_id_filename(shard.tensor_name) + ".grad.f32"),
        };
        for (const auto& candidate : candidates) {
            if (read_float_file(candidate, grads, weights.size())) {
                source = candidate;
                return true;
            }
        }
        source = opt.grad_dir;
        return false;
    }

    grads.assign(weights.size(), 0.0f);
    if (opt.grad_scale != 0.0f) {
        for (size_t i = 0; i < weights.size(); ++i) grads[i] = weights[i] * opt.grad_scale;
        source = "probe:weight*grad_scale";
    } else {
        source = "zero";
    }
    return true;
}

static uint32_t fold_id_for(const std::string& fold) {
    if (fold == "Pop") return 0;
    if (fold == "Wo") return 1;
    if (fold == "Yax") return 2;
    if (fold == "Sek") return 3;
    if (fold == "Chen") return 4;
    if (fold == "Xul") return 5;
    return 0;
}

static float phase_for(const XShardRecord& s) {
    if (std::isfinite(s.phase_angle) && s.phase_angle != 0.0f) return s.phase_angle;
    static constexpr float kPi3 = 1.0471975512f;
    return kPi3 * static_cast<float>(fold_id_for(s.fold));
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\\' || c == '"') {
            out.push_back('\\');
            out.push_back(c);
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

static void append_ledger(
    const Options& opt,
    const XShardRecord& shard,
    const char* status,
    const std::string& grad_source,
    float before,
    float after,
    size_t elements) {

    std::ofstream out(opt.ledger_path, std::ios::app | std::ios::binary);
    if (!out) return;
    out << "{\"op\":\"xshard.adapt\","
        << "\"mode\":\"" << (opt.apply ? "apply" : "dry-run") << "\","
        << "\"status\":\"" << status << "\","
        << "\"seq\":" << shard.seq << ","
        << "\"id\":\"" << json_escape(shard.id) << "\","
        << "\"tensor\":\"" << json_escape(shard.tensor_name) << "\","
        << "\"fold\":\"" << json_escape(shard.fold) << "\","
        << "\"dtype\":\"" << json_escape(shard.dtype) << "\","
        << "\"elements\":" << elements << ","
        << "\"lr\":" << opt.lr << ","
        << "\"grad_scale\":" << opt.grad_scale << ","
        << "\"grad_source\":\"" << json_escape(grad_source) << "\","
        << "\"weight_decay\":" << opt.weight_decay << ","
        << "\"fold_gate\":" << opt.fold_gate << ","
        << "\"update\":\"" << (opt.update_mode == 1 ? "sgd" : "adam") << "\","
        << "\"first_before\":" << before << ","
        << "\"first_after\":" << after
        << "}\n";
}

static HRESULT create_structured_buffer(
    ID3D11Device* dev,
    const void* data,
    UINT byte_width,
    UINT bind_flags,
    ID3D11Buffer** out) {

    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = std::max<UINT>(16, byte_width);
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = bind_flags;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = sizeof(float);

    D3D11_SUBRESOURCE_DATA init{};
    init.pSysMem = data;
    return dev->CreateBuffer(&desc, data ? &init : nullptr, out);
}

static HRESULT create_uav(ID3D11Device* dev, ID3D11Buffer* buf, UINT elements, ID3D11UnorderedAccessView** out) {
    D3D11_UNORDERED_ACCESS_VIEW_DESC desc{};
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    desc.Buffer.FirstElement = 0;
    desc.Buffer.NumElements = elements;
    return dev->CreateUnorderedAccessView(buf, &desc, out);
}

static bool dispatch_one(
    ID3D11Device* dev,
    ID3D11DeviceContext* ctx,
    ID3D11ComputeShader* cs,
    const XShardRecord& shard,
    std::vector<float>& weights,
    const std::vector<float>& grads,
    const Options& opt) {

    const UINT numel = static_cast<UINT>(weights.size());
    if (numel == 0) return false;

    std::vector<float> m(numel, 0.0f);
    std::vector<float> v(numel, 0.0f);

    ComPtr<ID3D11Buffer> w_buf, g_buf, m_buf, v_buf, cb, staging;
    ComPtr<ID3D11UnorderedAccessView> w_uav, g_uav, m_uav, v_uav;

    const UINT bytes = numel * sizeof(float);
    if (FAILED(create_structured_buffer(dev, weights.data(), bytes, D3D11_BIND_UNORDERED_ACCESS, w_buf.GetAddressOf()))) return false;
    if (FAILED(create_structured_buffer(dev, grads.data(), bytes, D3D11_BIND_UNORDERED_ACCESS, g_buf.GetAddressOf()))) return false;
    if (FAILED(create_structured_buffer(dev, m.data(), bytes, D3D11_BIND_UNORDERED_ACCESS, m_buf.GetAddressOf()))) return false;
    if (FAILED(create_structured_buffer(dev, v.data(), bytes, D3D11_BIND_UNORDERED_ACCESS, v_buf.GetAddressOf()))) return false;
    if (FAILED(create_uav(dev, w_buf.Get(), numel, w_uav.GetAddressOf()))) return false;
    if (FAILED(create_uav(dev, g_buf.Get(), numel, g_uav.GetAddressOf()))) return false;
    if (FAILED(create_uav(dev, m_buf.Get(), numel, m_uav.GetAddressOf()))) return false;
    if (FAILED(create_uav(dev, v_buf.Get(), numel, v_uav.GetAddressOf()))) return false;

    UINT groups = (numel + 255u) / 256u;
    UINT gx = std::min<UINT>(groups, 65535u);
    UINT gy = (groups + gx - 1u) / gx;

    AdaptParams params{};
    params.lr = opt.lr;
    params.weight_decay = opt.weight_decay;
    params.numel = numel;
    params.stride_x = gx;
    params.grad_clip = 1.0f;
    params.phase_angle = phase_for(shard);
    params.fold_id = fold_id_for(shard.fold);
    params.fold_gate = opt.fold_gate;
    params.update_mode = opt.update_mode;

    D3D11_BUFFER_DESC cb_desc{};
    cb_desc.ByteWidth = sizeof(AdaptParams);
    cb_desc.Usage = D3D11_USAGE_DEFAULT;
    cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA cb_init{};
    cb_init.pSysMem = &params;
    if (FAILED(dev->CreateBuffer(&cb_desc, &cb_init, cb.GetAddressOf()))) return false;

    D3D11_BUFFER_DESC st_desc{};
    st_desc.ByteWidth = bytes;
    st_desc.Usage = D3D11_USAGE_STAGING;
    st_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    st_desc.StructureByteStride = sizeof(float);
    st_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    if (FAILED(dev->CreateBuffer(&st_desc, nullptr, staging.GetAddressOf()))) return false;

    ID3D11UnorderedAccessView* uavs[] = { w_uav.Get(), g_uav.Get(), m_uav.Get(), v_uav.Get() };
    ctx->CSSetShader(cs, nullptr, 0);
    ctx->CSSetConstantBuffers(0, 1, cb.GetAddressOf());
    ctx->CSSetUnorderedAccessViews(0, 4, uavs, nullptr);
    ctx->Dispatch(gx, gy, 1);

    ID3D11UnorderedAccessView* null_uavs[4] = {};
    ID3D11Buffer* null_cb[1] = {};
    ctx->CSSetUnorderedAccessViews(0, 4, null_uavs, nullptr);
    ctx->CSSetConstantBuffers(0, 1, null_cb);
    ctx->CSSetShader(nullptr, nullptr, 0);
    ctx->CopyResource(staging.Get(), w_buf.Get());
    ctx->Flush();

    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) return false;
    std::memcpy(weights.data(), mapped.pData, bytes);
    ctx->Unmap(staging.Get(), 0);
    return true;
}

int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        usage();
        return 1;
    }

    XShardFile f;
    if (!f.open(opt.xshard_path.c_str()) || !f.is_valid()) {
        std::fprintf(stderr, "xshard_adapt error: invalid XSHARD file: %s\n", opt.xshard_path.c_str());
        return 1;
    }
    if (opt.apply && !f.reopen_write()) {
        std::fprintf(stderr, "xshard_adapt error: cannot reopen for write: %s\n", opt.xshard_path.c_str());
        return 1;
    }

    XShardFile grad_xshard_file;
    XShardFile* grad_xshard = nullptr;
    if (!opt.grad_xshard.empty()) {
        if (!grad_xshard_file.open(opt.grad_xshard.c_str()) || !grad_xshard_file.is_valid()) {
            std::fprintf(stderr, "xshard_adapt error: invalid gradient XSHARD file: %s\n", opt.grad_xshard.c_str());
            return 1;
        }
        grad_xshard = &grad_xshard_file;
    }

    auto bytecode = read_file(opt.shader_path);
    if (bytecode.empty()) {
        std::fprintf(stderr, "xshard_adapt error: shader not found or empty: %s\n", opt.shader_path.c_str());
        return 1;
    }

    D3D11Engine engine;
    if (!engine.init(opt.force_warp, opt.verbose)) {
        std::fprintf(stderr, "xshard_adapt error: D3D11 init failed: %s\n", engine.initReason().c_str());
        return 1;
    }

    ComPtr<ID3D11ComputeShader> cs;
    HRESULT hr = engine.rawDevice()->CreateComputeShader(bytecode.data(), bytecode.size(), nullptr, cs.GetAddressOf());
    if (FAILED(hr)) {
        std::fprintf(stderr, "xshard_adapt error: CreateComputeShader failed: 0x%08X\n", static_cast<unsigned>(hr));
        return 1;
    }

    int processed = 0;
    int skipped = 0;
    int errors = 0;

    for (const auto& shard : f.shards()) {
        if (processed >= opt.max_shards) break;
        if (!opt.fold_filter.empty() && shard.fold != opt.fold_filter) {
            ++skipped;
            continue;
        }
        if (shard.dtype != "F32" || shard.nbytes <= 0 || (shard.nbytes % 4) != 0) {
            ++skipped;
            continue;
        }

        std::vector<float> weights(static_cast<size_t>(shard.nbytes / 4));
        if (!f.read_shard(shard.seq, weights.data())) {
            std::fprintf(stderr, "xshard_adapt: read failed seq=%d id=%s\n", shard.seq, shard.id.c_str());
            if (opt.apply) f.mark_error(shard.seq);
            append_ledger(opt, shard, "read_error", "", 0.0f, 0.0f, 0);
            ++errors;
            continue;
        }

        std::vector<float> grads;
        std::string grad_source;
        if (!load_gradients_for_shard(opt, grad_xshard, shard, grads, weights, grad_source)) {
            std::fprintf(stderr, "xshard_adapt: gradient load failed seq=%d id=%s source=%s\n",
                shard.seq, shard.id.c_str(), grad_source.c_str());
            if (opt.apply) f.mark_error(shard.seq);
            append_ledger(opt, shard, "gradient_error", grad_source, weights.empty() ? 0.0f : weights[0], weights.empty() ? 0.0f : weights[0], weights.size());
            ++errors;
            continue;
        }

        const float before = weights.empty() ? 0.0f : weights[0];
        if (!dispatch_one(engine.rawDevice(), engine.rawCtx(), cs.Get(), shard, weights, grads, opt)) {
            std::fprintf(stderr, "xshard_adapt: dispatch failed seq=%d id=%s\n", shard.seq, shard.id.c_str());
            if (opt.apply) f.mark_error(shard.seq);
            append_ledger(opt, shard, "dispatch_error", grad_source, before, before, weights.size());
            ++errors;
            continue;
        }
        const float after = weights.empty() ? 0.0f : weights[0];

        if (opt.apply) {
            if (!f.commit_shard(shard.seq, weights.data(), shard.nbytes)) {
                std::fprintf(stderr, "xshard_adapt: commit failed seq=%d id=%s\n", shard.seq, shard.id.c_str());
                f.mark_error(shard.seq);
                append_ledger(opt, shard, "commit_error", grad_source, before, after, weights.size());
                ++errors;
                continue;
            }
        }

        append_ledger(opt, shard, opt.apply ? "committed" : "dry_run", grad_source, before, after, weights.size());
        std::printf("seq=%d fold=%s dtype=%s elems=%zu first=%g->%g grad=%s %s\n",
            shard.seq,
            shard.fold.c_str(),
            shard.dtype.c_str(),
            weights.size(),
            before,
            after,
            grad_source.c_str(),
            opt.apply ? "committed" : "dry-run");
        ++processed;
    }

    std::printf("xshard_adapt summary: processed=%d skipped=%d errors=%d mode=%s\n",
        processed, skipped, errors, opt.apply ? "apply" : "dry-run");
    if (processed > 0 || errors > 0) {
        std::printf("ledger=%s\n", opt.ledger_path.c_str());
    }
    return errors == 0 ? 0 : 2;
}
