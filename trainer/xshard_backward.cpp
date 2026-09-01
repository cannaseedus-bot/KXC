// xshard_backward.cpp — first-pass shard-resident gradient XSHARD producer
//
// This is the scheduler-facing bridge before the full model-specific backward
// pass lands. It streams a token/data bin to derive a deterministic batch signal,
// reads resident F32 shards, and emits a matching gradient XSHARD/1 container
// consumable by:
//   xshard_adapt model.xshard --grad-xshard gradients.xshard --apply

#include "xshard.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <windows.h>
#include <bcrypt.h>

#pragma comment(lib, "bcrypt")

static constexpr int ALIGNMENT = 64;
static constexpr uint16_t GRAD_FLAGS = XSHARD_FLAG_FOLD_TAGGED | XSHARD_FLAG_SUB_SHARDED | XSHARD_FLAG_STATE_MUTABLE;

struct Options {
    std::string model_xshard;
    std::string token_bin;
    std::string output;
    std::string fold_filter;
    int max_shards = 1;
    float grad_scale = 1.0e-3f;
    float weight_scale = 1.0e-4f;
};

struct GradShard {
    XShardRecord rec;
    std::vector<float> grads;
    std::string sha256;
    int64_t offset = 0;
    int64_t nbytes = 0;
};

static void usage() {
    std::fprintf(stderr,
        "usage: xshard_backward <model.xshard> --token-bin <tokens.bin> --output <gradients.xshard> [options]\n"
        "  --fold <phase>        only emit gradients for one fold\n"
        "  --max-shards <n>      max F32 shards to process (default 1)\n"
        "  --grad-scale <v>      token-signal gradient scale (default 1e-3)\n"
        "  --weight-scale <v>    weight-proportional gradient scale (default 1e-4)\n"
    );
}

static bool parse_float(const char* s, float& out) {
    char* end = nullptr;
    const float v = std::strtof(s, &end);
    if (!end || *end != '\0' || !std::isfinite(v)) return false;
    out = v;
    return true;
}

static bool parse_int(const char* s, int& out) {
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0' || v < 1) return false;
    out = static_cast<int>(v);
    return true;
}

static bool parse_args(int argc, char** argv, Options& opt) {
    if (argc < 2) return false;
    opt.model_xshard = argv[1];
    for (int i = 2; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--token-bin") == 0 && i + 1 < argc) opt.token_bin = argv[++i];
        else if (std::strcmp(a, "--output") == 0 && i + 1 < argc) opt.output = argv[++i];
        else if (std::strcmp(a, "--fold") == 0 && i + 1 < argc) opt.fold_filter = argv[++i];
        else if (std::strcmp(a, "--max-shards") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], opt.max_shards)) return false;
        } else if (std::strcmp(a, "--grad-scale") == 0 && i + 1 < argc) {
            if (!parse_float(argv[++i], opt.grad_scale)) return false;
        } else if (std::strcmp(a, "--weight-scale") == 0 && i + 1 < argc) {
            if (!parse_float(argv[++i], opt.weight_scale)) return false;
        } else {
            return false;
        }
    }
    return !opt.model_xshard.empty() && !opt.token_bin.empty() && !opt.output.empty();
}

static int64_t pad_up(int64_t n) {
    return ((n + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
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

static std::string sha256_hex(const void* data, size_t nbytes) {
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD cb = 0;
    DWORD hash_len = 0;
    std::string out(64, '0');

    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0) goto done;
    if (BCryptGetProperty(alg, BCRYPT_HASH_LENGTH, reinterpret_cast<PUCHAR>(&hash_len), sizeof(hash_len), &cb, 0) != 0) goto done;
    if (hash_len != 32) goto done;
    if (BCryptCreateHash(alg, &hash, nullptr, 0, nullptr, 0, 0) != 0) goto done;
    if (BCryptHashData(hash, reinterpret_cast<PUCHAR>(const_cast<void*>(data)), static_cast<ULONG>(nbytes), 0) != 0) goto done;
    {
        uint8_t digest[32]{};
        if (BCryptFinishHash(hash, digest, sizeof(digest), 0) != 0) goto done;
        static const char* hex = "0123456789abcdef";
        for (int i = 0; i < 32; ++i) {
            out[i * 2] = hex[digest[i] >> 4];
            out[i * 2 + 1] = hex[digest[i] & 0x0F];
        }
    }

done:
    if (hash) BCryptDestroyHash(hash);
    if (alg) BCryptCloseAlgorithmProvider(alg, 0);
    return out;
}

static float stream_token_signal(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::numeric_limits<float>::quiet_NaN();

    uint64_t count = 0;
    uint64_t mix = 1469598103934665603ull;
    uint8_t buf[64 * 1024];
    while (f) {
        f.read(reinterpret_cast<char*>(buf), sizeof(buf));
        const std::streamsize got = f.gcount();
        for (std::streamsize i = 0; i < got; ++i) {
            mix ^= buf[i];
            mix *= 1099511628211ull;
            ++count;
        }
    }
    if (count == 0) return 0.0f;
    const uint32_t low = static_cast<uint32_t>(mix & 0xFFFFFFFFu);
    return (static_cast<float>(low % 20001u) / 10000.0f) - 1.0f; // [-1, 1]
}

static std::string manifest_json(const std::vector<GradShard>& shards, int64_t state_start, int64_t data_start) {
    std::ostringstream os;
    os << "{\"@kind\":\"xshard/1-gradient\",\"version\":1,\"model\":\"gradient\",\"arch\":\"xshard-backward\","
       << "\"flags\":" << GRAD_FLAGS << ",\"alignment\":" << ALIGNMENT
       << ",\"n_shards\":" << shards.size()
       << ",\"state_start\":" << state_start
       << ",\"data_start\":" << data_start
       << ",\"shards\":[";
    for (size_t i = 0; i < shards.size(); ++i) {
        const auto& s = shards[i];
        if (i) os << ",";
        os << "{\"id\":\"" << json_escape(s.rec.id) << "\","
           << "\"seq\":" << s.rec.seq << ","
           << "\"tensor_name\":\"" << json_escape(s.rec.tensor_name) << "\","
           << "\"fold\":\"" << json_escape(s.rec.fold) << "\","
           << "\"phase_angle\":" << s.rec.phase_angle << ","
           << "\"shape\":[";
        for (size_t j = 0; j < s.rec.shape.size(); ++j) {
            if (j) os << ",";
            os << s.rec.shape[j];
        }
        os << "],\"dtype\":\"F32\","
           << "\"offset\":" << s.offset << ","
           << "\"nbytes\":" << s.nbytes << ","
           << "\"sha256\":\"" << s.sha256 << "\","
           << "\"shard_of\":\"" << json_escape(s.rec.shard_of.empty() ? s.rec.tensor_name : s.rec.shard_of) << "\","
           << "\"shard_index\":" << s.rec.shard_index << ","
           << "\"shard_count\":" << s.rec.shard_count;
        if (s.rec.shard_axis >= 0) {
            os << ",\"shard_axis\":" << s.rec.shard_axis
               << ",\"shard_slice\":[" << s.rec.shard_slice[0] << "," << s.rec.shard_slice[1] << "]";
        }
        os << "}";
    }
    os << "]}";
    return os.str();
}

static bool write_gradient_xshard(const std::string& path, std::vector<GradShard>& shards) {
    int64_t running = 0;
    for (auto& s : shards) {
        s.offset = running;
        s.nbytes = static_cast<int64_t>(s.grads.size() * sizeof(float));
        s.sha256 = sha256_hex(s.grads.data(), static_cast<size_t>(s.nbytes));
        running += pad_up(s.nbytes);
    }

    int64_t state_start = 0;
    int64_t data_start = 0;
    std::string manifest;
    for (int i = 0; i < 4; ++i) {
        manifest = manifest_json(shards, state_start, data_start);
        const int64_t next_state = pad_up(16 + static_cast<int64_t>(manifest.size()));
        const int64_t next_data = pad_up(next_state + static_cast<int64_t>(shards.size()));
        if (next_state == state_start && next_data == data_start) break;
        state_start = next_state;
        data_start = next_data;
    }
    manifest = manifest_json(shards, state_start, data_start);

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    const char magic[4] = {'X','S','H','D'};
    const uint16_t version = XSHARD_VERSION;
    const uint16_t flags = GRAD_FLAGS;
    const uint64_t manifest_len = static_cast<uint64_t>(manifest.size());
    out.write(magic, 4);
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
    out.write(reinterpret_cast<const char*>(&manifest_len), sizeof(manifest_len));
    out.write(manifest.data(), static_cast<std::streamsize>(manifest.size()));

    int64_t cur = 16 + static_cast<int64_t>(manifest.size());
    std::vector<char> zeros(static_cast<size_t>(std::max<int64_t>(ALIGNMENT, 1)), 0);
    if (state_start > cur) out.write(zeros.data(), static_cast<std::streamsize>(state_start - cur));
    for (size_t i = 0; i < shards.size(); ++i) {
        const uint8_t pending = XSHARD_STATE_PENDING;
        out.write(reinterpret_cast<const char*>(&pending), 1);
    }
    cur = state_start + static_cast<int64_t>(shards.size());
    if (data_start > cur) out.write(zeros.data(), static_cast<std::streamsize>(data_start - cur));

    for (const auto& s : shards) {
        const int64_t raw = static_cast<int64_t>(s.grads.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(s.grads.data()), static_cast<std::streamsize>(raw));
        const int64_t padded = pad_up(raw);
        if (padded > raw) out.write(zeros.data(), static_cast<std::streamsize>(padded - raw));
    }

    const uint32_t n = static_cast<uint32_t>(shards.size());
    const uint32_t footer_magic = XSHARD_MAGIC;
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(&footer_magic), sizeof(footer_magic));
    return static_cast<bool>(out);
}

int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        usage();
        return 1;
    }

    const float token_signal = stream_token_signal(opt.token_bin);
    if (!std::isfinite(token_signal)) {
        std::fprintf(stderr, "xshard_backward error: cannot stream token bin: %s\n", opt.token_bin.c_str());
        return 1;
    }

    XShardFile model;
    if (!model.open(opt.model_xshard.c_str()) || !model.is_valid()) {
        std::fprintf(stderr, "xshard_backward error: invalid model XSHARD: %s\n", opt.model_xshard.c_str());
        return 1;
    }

    std::vector<GradShard> out;
    int skipped = 0;
    for (const auto& shard : model.shards()) {
        if (static_cast<int>(out.size()) >= opt.max_shards) break;
        if (!opt.fold_filter.empty() && shard.fold != opt.fold_filter) {
            ++skipped;
            continue;
        }
        if (shard.dtype != "F32" || shard.nbytes <= 0 || (shard.nbytes % 4) != 0) {
            ++skipped;
            continue;
        }

        std::vector<float> weights(static_cast<size_t>(shard.nbytes / sizeof(float)));
        if (!model.read_shard(shard.seq, weights.data())) {
            std::fprintf(stderr, "xshard_backward: read failed seq=%d id=%s\n", shard.seq, shard.id.c_str());
            return 2;
        }

        GradShard g;
        g.rec = shard;
        g.grads.resize(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            const float phase = static_cast<float>((i % 31) - 15) / 15.0f;
            g.grads[i] = (weights[i] * opt.weight_scale) + (token_signal * phase * opt.grad_scale);
        }
        out.push_back(std::move(g));
    }

    if (out.empty()) {
        std::fprintf(stderr, "xshard_backward error: no F32 shards selected (skipped=%d)\n", skipped);
        return 1;
    }

    if (!write_gradient_xshard(opt.output, out)) {
        std::fprintf(stderr, "xshard_backward error: failed writing %s\n", opt.output.c_str());
        return 1;
    }

    std::printf("xshard_backward summary: gradients=%zu skipped=%d token_signal=%g output=%s\n",
        out.size(), skipped, token_signal, opt.output.c_str());
    return 0;
}
