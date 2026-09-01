#pragma once
// xshard.h — XSHARD/1 streaming tensor shard format
// Header-only reader/writer. Standard library only, no external dependencies.
//
// Usage (read):
//   XShardFile f;
//   f.open("model.xshard");
//   auto& shard = f.shards()[0];
//   std::vector<uint8_t> buf(shard.nbytes);
//   f.read_shard(0, buf.data());
//
// Usage (write-back after training):
//   f.reopen_write();
//   f.commit_shard(0, adapted_buf.data(), shard.nbytes);  // write + mark trained

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr uint32_t XSHARD_MAGIC         = 0x44485358u;  // "XSHD" LE
static constexpr uint16_t XSHARD_VERSION       = 1;
static constexpr int      XSHARD_ALIGNMENT     = 64;

// flags (bit field in header)
static constexpr uint16_t XSHARD_FLAG_FOLD_TAGGED   = 0x01;
static constexpr uint16_t XSHARD_FLAG_SUB_SHARDED   = 0x02;
static constexpr uint16_t XSHARD_FLAG_STATE_MUTABLE = 0x04;

// state byte values
static constexpr uint8_t XSHARD_STATE_PENDING = 0x00;
static constexpr uint8_t XSHARD_STATE_TRAINED = 0x01;
static constexpr uint8_t XSHARD_STATE_ERROR   = 0x02;
static constexpr uint8_t XSHARD_STATE_LOCKED  = 0xFF;

// ── Binary header (16 bytes, fixed) ──────────────────────────────────────────

#pragma pack(push, 1)
struct XShardBinHeader {
    uint8_t  magic[4];       // 'X','S','H','D'
    uint16_t version;        // 1 (u16 LE)
    uint16_t flags;          // XSHARD_FLAG_* bitmask (u16 LE)
    uint64_t manifest_len;   // byte count of JSON manifest (u64 LE)
};
static_assert(sizeof(XShardBinHeader) == 16, "XShardBinHeader must be 16 bytes");
#pragma pack(pop)

// ── Shard record (parsed from manifest JSON) ──────────────────────────────────

struct XShardRecord {
    std::string id;
    int         seq          = 0;
    std::string tensor_name;
    std::string fold;           // Pop/Wo/Yax/Sek/Chen/Xul
    float       phase_angle  = 0.0f;
    std::vector<int64_t> shape;
    std::string dtype;          // F32/BF16/F16/INT8/Q8_0/Q4_K
    int64_t     offset       = 0;    // byte offset from data_start
    int64_t     nbytes       = 0;
    std::string sha256;
    std::string shard_of;
    int         shard_index  = 0;
    int         shard_count  = 1;
    int         shard_axis   = -1;   // -1 = whole tensor (no sub-sharding)
    int         shard_slice[2] = {0, 0};
};

// ── Minimal JSON field scanner ────────────────────────────────────────────────
// Only handles compact JSON (no spaces) as written by safetensors_to_xshard.py.

namespace xshard_detail {

inline bool json_has(const std::string& j, const char* key) {
    return j.find(std::string("\"") + key + "\":") != std::string::npos;
}

inline int64_t json_int(const std::string& j, const char* key, size_t hint = 0) {
    std::string pat = std::string("\"") + key + "\":";
    auto pos = j.find(pat, hint);
    if (pos == std::string::npos) return 0;
    pos += pat.size();
    if (pos >= j.size()) return 0;
    bool neg = j[pos] == '-';
    if (neg) ++pos;
    int64_t v = 0;
    while (pos < j.size() && j[pos] >= '0' && j[pos] <= '9')
        v = v * 10 + (j[pos++] - '0');
    return neg ? -v : v;
}

inline std::string json_str(const std::string& j, const char* key, size_t hint = 0) {
    std::string pat = std::string("\"") + key + "\":\"";
    auto pos = j.find(pat, hint);
    if (pos == std::string::npos) return {};
    pos += pat.size();
    auto end = j.find('"', pos);
    if (end == std::string::npos) return {};
    return j.substr(pos, end - pos);
}

inline float json_float(const std::string& j, const char* key, size_t hint = 0) {
    std::string pat = std::string("\"") + key + "\":";
    auto pos = j.find(pat, hint);
    if (pos == std::string::npos) return 0.0f;
    pos += pat.size();
    try { return std::stof(j.substr(pos, 32)); } catch (...) { return 0.0f; }
}

// Parse int array "[a,b,c]" at key.
inline std::vector<int64_t> json_int_array(const std::string& j, const char* key, size_t hint = 0) {
    std::string pat = std::string("\"") + key + "\":[";
    auto pos = j.find(pat, hint);
    if (pos == std::string::npos) return {};
    pos += pat.size();
    std::vector<int64_t> out;
    while (pos < j.size() && j[pos] != ']') {
        if (j[pos] >= '0' && j[pos] <= '9') {
            int64_t v = 0;
            while (pos < j.size() && j[pos] >= '0' && j[pos] <= '9')
                v = v * 10 + (j[pos++] - '0');
            out.push_back(v);
        } else ++pos;
    }
    return out;
}

// Parse all shard records from manifest JSON.
inline std::vector<XShardRecord> json_shards(const std::string& j) {
    std::vector<XShardRecord> out;
    auto arr = j.find("\"shards\":[");
    if (arr == std::string::npos) return out;
    size_t p = arr + 9; // points at '['

    while (p < j.size()) {
        auto obj_start = j.find('{', p);
        if (obj_start == std::string::npos) break;

        // find matching '}'
        int depth = 0;
        size_t obj_end = obj_start;
        for (size_t i = obj_start; i < j.size(); ++i) {
            if (j[i] == '{') ++depth;
            else if (j[i] == '}' && --depth == 0) { obj_end = i; break; }
        }
        if (obj_end <= obj_start) break;

        const std::string o = j.substr(obj_start, obj_end - obj_start + 1);
        XShardRecord r;
        r.id          = json_str(o, "id");
        r.seq         = (int)json_int(o, "seq");
        r.tensor_name = json_str(o, "tensor_name");
        r.fold        = json_str(o, "fold");
        r.phase_angle = json_float(o, "phase_angle");
        r.dtype       = json_str(o, "dtype");
        r.offset      = json_int(o, "offset");
        r.nbytes      = json_int(o, "nbytes");
        r.sha256      = json_str(o, "sha256");
        r.shard_of    = json_str(o, "shard_of");
        r.shard_index = (int)json_int(o, "shard_index");
        r.shard_count = (int)std::max((int64_t)1, json_int(o, "shard_count"));
        r.shape       = json_int_array(o, "shape");

        // shard_axis: -1 means whole tensor (field absent = no sub-sharding)
        if (json_has(o, "shard_axis")) {
            r.shard_axis     = (int)json_int(o, "shard_axis");
            auto sl          = json_int_array(o, "shard_slice");
            r.shard_slice[0] = sl.size() > 0 ? (int)sl[0] : 0;
            r.shard_slice[1] = sl.size() > 1 ? (int)sl[1] : 0;
        }

        out.push_back(std::move(r));
        p = obj_end + 1;
        if (p < j.size() && j[p] == ']') break;
    }
    return out;
}

} // namespace xshard_detail

// ── XShardFile ────────────────────────────────────────────────────────────────

class XShardFile {
public:
    XShardFile()  = default;
    ~XShardFile() { close(); }

    XShardFile(const XShardFile&)            = delete;
    XShardFile& operator=(const XShardFile&) = delete;

    // Open for reading.  Call reopen_write() before any write operations.
    bool open(const char* path) {
        close();
        fp_ = std::fopen(path, "rb");
        if (!fp_) return false;
        if (!read_header()) { close(); return false; }
        path_       = path;
        write_mode_ = false;
        return true;
    }

    // Reopen the same file in r+b mode for in-place updates.
    bool reopen_write() {
        if (write_mode_) return true;
        std::fclose(fp_); fp_ = nullptr;
        fp_ = std::fopen(path_.c_str(), "r+b");
        if (!fp_) return false;
        write_mode_ = true;
        return true;
    }

    void close() {
        if (fp_) { std::fclose(fp_); fp_ = nullptr; }
        write_mode_ = false;
    }

    bool is_open()  const { return fp_ != nullptr; }
    bool is_valid() const { return is_open() && n_shards_ > 0; }

    // ── Manifest ──────────────────────────────────────────────────────────────

    const std::string&              manifest_json() const { return manifest_; }
    const std::vector<XShardRecord>& shards()       const { return shards_; }

    // ── Header fields ─────────────────────────────────────────────────────────

    int      n_shards()    const { return n_shards_; }
    int64_t  state_start() const { return state_start_; }
    int64_t  data_start()  const { return data_start_; }
    uint16_t flags()       const { return flags_; }

    bool is_fold_tagged()   const { return (flags_ & XSHARD_FLAG_FOLD_TAGGED)   != 0; }
    bool is_sub_sharded()   const { return (flags_ & XSHARD_FLAG_SUB_SHARDED)   != 0; }
    bool is_state_mutable() const { return (flags_ & XSHARD_FLAG_STATE_MUTABLE) != 0; }

    // ── State block (one byte per shard, mutable in-place) ───────────────────

    uint8_t get_state(int seq) const {
        if (!fp_ || seq < 0 || seq >= n_shards_) return XSHARD_STATE_LOCKED;
        std::fseek(fp_, (long)(state_start_ + seq), SEEK_SET);
        uint8_t b = XSHARD_STATE_LOCKED;
        std::fread(&b, 1, 1, fp_);
        return b;
    }

    bool set_state(int seq, uint8_t status) {
        if (!fp_ || !write_mode_ || seq < 0 || seq >= n_shards_) return false;
        std::fseek(fp_, (long)(state_start_ + seq), SEEK_SET);
        return std::fwrite(&status, 1, 1, fp_) == 1;
    }

    // ── Shard data ────────────────────────────────────────────────────────────

    // Read shard by seq index.  buf must be at least shard.nbytes bytes.
    bool read_shard(int seq, void* buf) const {
        if (!fp_ || seq < 0 || seq >= (int)shards_.size()) return false;
        const auto& s = shards_[seq];
        std::fseek(fp_, (long)(data_start_ + s.offset), SEEK_SET);
        return std::fread(buf, 1, (size_t)s.nbytes, fp_) == (size_t)s.nbytes;
    }

    // Write adapted shard data back in-place.  nbytes must equal shard.nbytes.
    bool write_shard(int seq, const void* buf, int64_t nbytes) {
        if (!fp_ || !write_mode_ || seq < 0 || seq >= (int)shards_.size()) return false;
        const auto& s = shards_[seq];
        if (nbytes != s.nbytes) return false;
        std::fseek(fp_, (long)(data_start_ + s.offset), SEEK_SET);
        bool ok = std::fwrite(buf, 1, (size_t)nbytes, fp_) == (size_t)nbytes;
        if (ok) std::fflush(fp_);
        return ok;
    }

    // Write shard + mark state=trained in one call.
    bool commit_shard(int seq, const void* buf, int64_t nbytes) {
        return write_shard(seq, buf, nbytes) && set_state(seq, XSHARD_STATE_TRAINED);
    }

    // Mark a shard as error without touching data.
    bool mark_error(int seq) { return set_state(seq, XSHARD_STATE_ERROR); }

    // ── Lookup ────────────────────────────────────────────────────────────────

    const XShardRecord* find_by_id(const std::string& id) const {
        for (const auto& s : shards_)
            if (s.id == id) return &s;
        return nullptr;
    }

    const XShardRecord* find_by_tensor(const std::string& name, int shard_index = 0) const {
        for (const auto& s : shards_)
            if (s.tensor_name == name && s.shard_index == shard_index) return &s;
        return nullptr;
    }

    // All shards with a given fold phase.
    std::vector<const XShardRecord*> shards_for_fold(const std::string& fold) const {
        std::vector<const XShardRecord*> out;
        for (const auto& s : shards_)
            if (s.fold == fold) out.push_back(&s);
        return out;
    }

private:
    bool read_header() {
        XShardBinHeader hdr{};
        if (std::fread(&hdr, 1, sizeof(hdr), fp_) != sizeof(hdr)) return false;

        uint32_t magic = 0;
        std::memcpy(&magic, hdr.magic, 4);
        if (magic != XSHARD_MAGIC || hdr.version != XSHARD_VERSION) return false;

        flags_        = hdr.flags;
        manifest_len_ = hdr.manifest_len;

        manifest_.resize(manifest_len_);
        if (std::fread(manifest_.data(), 1, manifest_len_, fp_) != manifest_len_)
            return false;

        using namespace xshard_detail;
        state_start_ = json_int(manifest_, "state_start");
        data_start_  = json_int(manifest_, "data_start");
        n_shards_    = (int)json_int(manifest_, "n_shards");
        shards_      = json_shards(manifest_);
        return true;
    }

    FILE*       fp_           = nullptr;
    std::string path_;
    std::string manifest_;
    std::vector<XShardRecord> shards_;
    int64_t     manifest_len_ = 0;
    int64_t     state_start_  = 0;
    int64_t     data_start_   = 0;
    int         n_shards_     = 0;
    uint16_t    flags_        = 0;
    bool        write_mode_   = false;
};
