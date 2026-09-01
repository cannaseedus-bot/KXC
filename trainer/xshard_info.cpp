// xshard_info.cpp — XSHARD/1 inspector and formatter
// usage: xshard_info <file.xshard> [--manifest] [--fold <fold>] [--state-only]
//
//   (no flags)      print header, fold distribution, state summary, and shard list
//   --manifest      also dump raw manifest JSON at end
//   --fold <phase>  filter shard list to one fold (Pop/Wo/Yax/Sek/Chen/Xul)
//   --state-only    print state block only (useful for polling training progress)

#include "xshard.h"
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

static const char* state_str(uint8_t s) {
    switch (s) {
        case XSHARD_STATE_PENDING: return "pending";
        case XSHARD_STATE_TRAINED: return "trained";
        case XSHARD_STATE_ERROR:   return "error";
        case XSHARD_STATE_LOCKED:  return "locked";
        default: return "unknown";
    }
}

// Return phase angle string matching the fold.
static const char* phase_str(const std::string& fold) {
    if (fold == "Pop")  return "0.0000";
    if (fold == "Wo")   return "1.0472";
    if (fold == "Yax")  return "2.0944";
    if (fold == "Sek")  return "3.1416";
    if (fold == "Chen") return "4.1888";
    if (fold == "Xul")  return "5.2360";
    return "?";
}

static void print_usage() {
    std::fprintf(stderr,
        "usage: xshard_info <file.xshard> [options]\n"
        "  --manifest          dump raw manifest JSON\n"
        "  --fold <phase>      filter shard list (Pop/Wo/Yax/Sek/Chen/Xul)\n"
        "  --state-only        print state block summary only\n"
    );
}

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(); return 1; }

    const char* path         = argv[1];
    bool        dump_manifest = false;
    bool        state_only    = false;
    std::string fold_filter;

    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--manifest") == 0) {
            dump_manifest = true;
        } else if (std::strcmp(argv[i], "--state-only") == 0) {
            state_only = true;
        } else if (std::strcmp(argv[i], "--fold") == 0 && i + 1 < argc) {
            fold_filter = argv[++i];
        }
    }

    XShardFile f;
    if (!f.open(path)) {
        std::fprintf(stderr, "error: cannot open '%s'\n", path);
        return 1;
    }
    if (!f.is_valid()) {
        std::fprintf(stderr, "error: '%s' is not a valid XSHARD/1 file\n", path);
        return 1;
    }

    // ── State-only mode ───────────────────────────────────────────────────────

    if (state_only) {
        int n_trained = 0, n_error = 0, n_pending = 0, n_locked = 0;
        for (int i = 0; i < f.n_shards(); ++i) {
            uint8_t st = f.get_state(i);
            if      (st == XSHARD_STATE_TRAINED) ++n_trained;
            else if (st == XSHARD_STATE_ERROR)   ++n_error;
            else if (st == XSHARD_STATE_LOCKED)  ++n_locked;
            else                                 ++n_pending;
        }
        printf("total=%d  trained=%d  pending=%d  error=%d  locked=%d\n",
               f.n_shards(), n_trained, n_pending, n_error, n_locked);
        printf("progress=%.1f%%\n",
               f.n_shards() > 0 ? 100.0 * n_trained / f.n_shards() : 0.0);
        return 0;
    }

    // ── Header summary ────────────────────────────────────────────────────────

    printf("XSHARD/1\n");
    printf("  file        : %s\n", path);
    printf("  flags       : 0x%04X  (fold_tagged=%s  sub_sharded=%s  state_mutable=%s)\n",
           f.flags(),
           f.is_fold_tagged()   ? "yes" : "no",
           f.is_sub_sharded()   ? "yes" : "no",
           f.is_state_mutable() ? "yes" : "no");
    printf("  n_shards    : %d\n", f.n_shards());
    printf("  state_start : %lld\n", (long long)f.state_start());
    printf("  data_start  : %lld\n", (long long)f.data_start());

    // ── Fold distribution ─────────────────────────────────────────────────────

    const std::string fold_order[] = {"Pop","Wo","Yax","Sek","Chen","Xul"};
    std::map<std::string, int>     fold_count;
    std::map<std::string, int64_t> fold_bytes;

    for (const auto& s : f.shards()) {
        fold_count[s.fold]++;
        fold_bytes[s.fold] += s.nbytes;
    }

    printf("\n  fold distribution:\n");
    printf("  %-5s  %6s  %10s  %8s\n", "fold", "phase", "shards", "data MB");
    printf("  %s\n", std::string(36, '-').c_str());
    for (const auto& fold : fold_order) {
        if (fold_count.find(fold) == fold_count.end()) continue;
        printf("  %-5s  %6s  %10d  %8.1f\n",
               fold.c_str(), phase_str(fold),
               fold_count[fold],
               fold_bytes[fold] / 1024.0 / 1024.0);
    }

    // ── State summary ─────────────────────────────────────────────────────────

    int n_trained = 0, n_error = 0, n_pending = 0, n_locked = 0;
    for (int i = 0; i < f.n_shards(); ++i) {
        uint8_t st = f.get_state(i);
        if      (st == XSHARD_STATE_TRAINED) ++n_trained;
        else if (st == XSHARD_STATE_ERROR)   ++n_error;
        else if (st == XSHARD_STATE_LOCKED)  ++n_locked;
        else                                 ++n_pending;
    }
    printf("\n  state: trained=%d  pending=%d  error=%d  locked=%d  (%.1f%% done)\n",
           n_trained, n_pending, n_error, n_locked,
           f.n_shards() > 0 ? 100.0 * n_trained / f.n_shards() : 0.0);

    // ── Shard list ────────────────────────────────────────────────────────────

    const auto& shards = f.shards();
    const char* filter = fold_filter.empty() ? nullptr : fold_filter.c_str();

    if (filter) {
        printf("\n  shards (fold=%s):\n", filter);
    } else {
        printf("\n  shards:\n");
    }

    printf("  %4s  %-5s  %-6s  %-42s  %10s  %s\n",
           "seq", "fold", "dtype", "id", "nbytes", "state");
    printf("  %s\n", std::string(82, '-').c_str());

    for (const auto& s : shards) {
        if (filter && s.fold != filter) continue;
        uint8_t st = f.get_state(s.seq);

        // truncate id for display
        std::string disp_id = s.id;
        if (disp_id.size() > 42) {
            disp_id = ".." + disp_id.substr(disp_id.size() - 40);
        }

        printf("  %4d  %-5s  %-6s  %-42s  %10lld  %s\n",
               s.seq,
               s.fold.c_str(),
               s.dtype.c_str(),
               disp_id.c_str(),
               (long long)s.nbytes,
               state_str(st));
    }

    // ── Optional manifest dump ────────────────────────────────────────────────

    if (dump_manifest) {
        printf("\n--- manifest JSON (%lld bytes) ---\n%s\n",
               (long long)f.manifest_json().size(),
               f.manifest_json().c_str());
    }

    return 0;
}
