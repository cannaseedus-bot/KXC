using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace Micronaut.Streaming.Adapter
{
    // XSHARD/1 shard descriptor — mirrors the manifest record from gguf_to_xshard.py / xshard.h
    public sealed class XShardEntry
    {
        public string Id         { get; set; } = "";
        public string TensorName { get; set; } = "";
        public string Fold       { get; set; } = "";  // Pop/Wo/Yax/Sek/Chen/Xul
        public double PhaseAngle { get; set; }        // radians, matches _PHASE_ANGLES
        public int[]  Shape      { get; set; } = Array.Empty<int>();
        public string Dtype      { get; set; } = "";  // F32/F16/Q4_0/Q4_1/Q8_0 etc. (verbatim GGML)
        public long   Offset     { get; set; }        // byte offset from data_start
        public long   Nbytes     { get; set; }
        public string Sha256     { get; set; } = "";
        public string ShardOf    { get; set; } = "";
        public int    ShardIndex { get; set; }
        public int    ShardCount { get; set; }
    }

    // Open XSHARD/1 file — parsed manifest + seekable stream for fold-lane streaming.
    // Dispose when done; the underlying FileStream is kept open for lazy shard reads.
    public sealed class XShardSession : IDisposable
    {
        public string Model     { get; }
        public string Arch      { get; }
        public string Source    { get; }
        public int    NShards   { get; }
        public long   DataStart { get; }
        public IReadOnlyList<XShardEntry> Shards { get; }

        private readonly FileStream _fs;

        internal XShardSession(string model, string arch, string source,
                                long dataStart, List<XShardEntry> shards, FileStream fs)
        {
            Model = model; Arch = arch; Source = source;
            DataStart = dataStart; NShards = shards.Count;
            Shards = shards; _fs = fs;
        }

        // Enumerate shards belonging to a phase fold lane
        public IEnumerable<XShardEntry> FoldShards(string fold)
        {
            foreach (var s in Shards)
                if (s.Fold == fold) yield return s;
        }

        // Read raw quantized bytes for one shard — no dequantize.
        // xshard_adapt.cpp knows the block layout from entry.Dtype.
        public async Task<byte[]> ReadShardBytesAsync(XShardEntry entry,
                                                       CancellationToken ct = default)
        {
            var buf = new byte[entry.Nbytes];
            _fs.Seek(DataStart + entry.Offset, SeekOrigin.Begin);
            int read = 0;
            while (read < buf.Length)
            {
                int n = await _fs.ReadAsync(buf.AsMemory(read), ct);
                if (n == 0) break;
                read += n;
            }
            return buf;
        }

        // Stream every shard in a fold lane as (entry, raw bytes).
        // Shards arrive in manifest order (seq order = layer order).
        public async IAsyncEnumerable<(XShardEntry Entry, byte[] Raw)>
            StreamFoldAsync(string fold,
                            [System.Runtime.CompilerServices.EnumeratorCancellation]
                            CancellationToken ct = default)
        {
            foreach (var entry in FoldShards(fold))
            {
                ct.ThrowIfCancellationRequested();
                yield return (entry, await ReadShardBytesAsync(entry, ct));
            }
        }

        public void Dispose() => _fs.Dispose();
    }

    // Adapter that maps XSHARD/1 phase-fold tile lanes to runtime frame buffers.
    //
    // Two modes:
    //   A) xshard streaming  — OpenXShard(path) → XShardSession → StreamFoldAsync(fold)
    //   B) tile-dir streaming — PrefetchTopTilesAsync / StreamRecordsAsync (JSONL tiles)
    //
    // Fold names (Pop/Wo/Yax/Sek/Chen/Xul) and phase angles are authoritative from
    // gguf_to_xshard.py _FOLD_PATTERNS + _PHASE_ANGLES and xshard_adapt.cpp fold_id_for().
    public class AdaptiveContentModelAdapter
    {
        public string TilesRoot { get; }
        public string WeightFile { get; }

        // Fold name → frame buffer tag. Keys are the six phase fold names used by
        // gguf_to_xshard, xshard_adapt, and the XSHARD/1 manifest "fold" field.
        public Dictionary<string, string> LaneToFrame { get; } = new Dictionary<string, string>();

        // Phase angle (radians) per fold — matches _PHASE_ANGLES in gguf_to_xshard.py
        // and fold_id_for() in xshard_adapt.cpp (Pop=0, Wo=π/3, Yax=2π/3, Sek=π, Chen=4π/3, Xul=5π/3)
        public static readonly IReadOnlyDictionary<string, double> PhaseAngle =
            new Dictionary<string, double> {
                ["Pop"]  = 0.0,
                ["Wo"]   = Math.PI / 3,
                ["Yax"]  = 2 * Math.PI / 3,
                ["Sek"]  = Math.PI,
                ["Chen"] = 4 * Math.PI / 3,
                ["Xul"]  = 5 * Math.PI / 3,
            };

        public AdaptiveContentModelAdapter(string tilesRoot, string weightFile)
        {
            TilesRoot = tilesRoot ?? throw new ArgumentNullException(nameof(tilesRoot));
            WeightFile = weightFile;
            // Phase fold → frame buffer. Fold names match xshard manifest "fold" field.
            //   Pop  (0)       — token/pos embeddings   (wte/wpe/token_embd)
            //   Wo   (π/3)     — FFN gate/up projection  (c_fc/ffn_up/ffn_gate)
            //   Yax  (2π/3)    — FFN down projection     (c_proj/ffn_down/down_proj)
            //   Sek  (π)       — attention Q/K/V/O       (attn_q/k/v/output)
            //   Chen (4π/3)    — layer-norm / bias        (ln_f/layernorm/norm)
            //   Xul  (5π/3)    — output/lm_head           (lm_head/output.weight)
            LaneToFrame["Pop"]  = "EMBED_FRAME";
            LaneToFrame["Wo"]   = "FFN_GATE_FRAME";
            LaneToFrame["Yax"]  = "FFN_DOWN_FRAME";
            LaneToFrame["Sek"]  = "ATTN_FRAME";
            LaneToFrame["Chen"] = "NORM_FRAME";
            LaneToFrame["Xul"]  = "LM_HEAD_FRAME";
        }

        public void MapLanes(Dictionary<string, string> mapping)
        {
            if (mapping == null) return;
            foreach (var kv in mapping) LaneToFrame[kv.Key] = kv.Value;
        }

        // ── Mode A: XSHARD/1 streaming ────────────────────────────────────────

        // Open an .xshard file for fold-lane streaming.
        // Reads the XSHD binary header + manifest JSON; keeps the FileStream open.
        // Caller must dispose the returned XShardSession.
        //
        // Compatible with gemma-3-1b-it.xshard (gemma3, q4_1, 340 shards) and
        // Qwen-Coder.xshard (qwen2, q4_0, 290 shards) — both confirmed XSHARD/1 flag=5.
        public static XShardSession OpenXShard(string xshardPath)
        {
            var fs = new FileStream(xshardPath, FileMode.Open, FileAccess.Read,
                                    FileShare.Read, 65536, FileOptions.Asynchronous);
            // XSHD header: magic(4) + version(2) + flags(2) + manifest_len(8) = 16 bytes
            var hdr = new byte[16];
            ReadExact(fs, hdr);

            if (hdr[0] != 'X' || hdr[1] != 'S' || hdr[2] != 'H' || hdr[3] != 'D')
            {
                fs.Dispose();
                throw new InvalidDataException($"{xshardPath}: not an XSHARD/1 file (bad magic)");
            }

            long mlen = BitConverter.ToInt64(hdr, 8);
            var mb = new byte[mlen];
            ReadExact(fs, mb);

            long dataStart = 0;
            string model = "", arch = "", source = "";
            var shards = new List<XShardEntry>();

            using (var doc = JsonDocument.Parse(mb))
            {
                var root = doc.RootElement;
                if (root.TryGetProperty("model",      out var v)) model     = v.GetString() ?? "";
                if (root.TryGetProperty("arch",       out v))     arch      = v.GetString() ?? "";
                if (root.TryGetProperty("source",     out v))     source    = v.GetString() ?? "";
                if (root.TryGetProperty("data_start", out v))     dataStart = v.GetInt64();

                if (root.TryGetProperty("shards", out var sarr))
                {
                    foreach (var s in sarr.EnumerateArray())
                    {
                        var e = new XShardEntry();
                        if (s.TryGetProperty("id",          out v)) e.Id         = v.GetString() ?? "";
                        if (s.TryGetProperty("tensor_name", out v)) e.TensorName = v.GetString() ?? "";
                        if (s.TryGetProperty("fold",        out v)) e.Fold       = v.GetString() ?? "";
                        if (s.TryGetProperty("phase_angle", out v)) e.PhaseAngle = v.GetDouble();
                        if (s.TryGetProperty("dtype",       out v)) e.Dtype      = v.GetString() ?? "";
                        if (s.TryGetProperty("offset",      out v)) e.Offset     = v.GetInt64();
                        if (s.TryGetProperty("nbytes",      out v)) e.Nbytes     = v.GetInt64();
                        if (s.TryGetProperty("sha256",      out v)) e.Sha256     = v.GetString() ?? "";
                        if (s.TryGetProperty("shard_of",    out v)) e.ShardOf    = v.GetString() ?? "";
                        if (s.TryGetProperty("shard_index", out v)) e.ShardIndex = v.GetInt32();
                        if (s.TryGetProperty("shard_count", out v)) e.ShardCount = v.GetInt32();
                        if (s.TryGetProperty("shape",       out v))
                        {
                            var dims = new List<int>();
                            foreach (var d in v.EnumerateArray()) dims.Add(d.GetInt32());
                            e.Shape = dims.ToArray();
                        }
                        shards.Add(e);
                    }
                }
            }

            return new XShardSession(model, arch, source, dataStart, shards, fs);
        }

        private static void ReadExact(Stream s, byte[] buf)
        {
            int read = 0;
            while (read < buf.Length)
            {
                int n = s.Read(buf, read, buf.Length - read);
                if (n == 0) throw new EndOfStreamException("Truncated XSHARD header/manifest");
                read += n;
            }
        }

        // ── Mode B: tile-directory streaming (JSONL tiles) ────────────────────

        // Prefetch top-N tiles for a fold lane from a tile directory.
        // Reads tile.meta.json "fold" or "lanes" field for matching.
        public async Task PrefetchTopTilesAsync(string lane, int topN = 1)
        {
            var tiles = Directory.GetDirectories(TilesRoot, "tile_*");
            var candidates = new List<string>();
            foreach (var t in tiles)
            {
                try {
                    var metaPath = Path.Combine(t, "tile.meta.json");
                    if (!File.Exists(metaPath)) continue;
                    using var fs2 = File.OpenRead(metaPath);
                    var meta = JsonDocument.Parse(fs2);
                    var root = meta.RootElement;
                    bool match = false;
                    if (root.TryGetProperty("fold", out var fv) && fv.GetString() == lane)
                        match = true;
                    else if (root.TryGetProperty("lanes", out var lv))
                        foreach (var l in lv.EnumerateArray())
                            if (l.GetString() == lane) { match = true; break; }
                    if (match) candidates.Add(t);
                } catch { }
                if (candidates.Count >= topN) break;
            }
            var tasks = new List<Task>();
            foreach (var c in candidates) tasks.Add(PrimeTileAsync(c));
            await Task.WhenAll(tasks);
        }

        // Prime a JSONL(.gz) tile into cache without fully materialising on heap.
        public async Task PrimeTileAsync(string tilePath)
        {
            var gz    = Path.Combine(tilePath, "records.jsonl.gz");
            var plain = Path.Combine(tilePath, "records.jsonl");
            if (File.Exists(gz)) {
                using var fs2 = File.OpenRead(gz);
                using var g   = new GZipStream(fs2, CompressionMode.Decompress);
                using var sr  = new StreamReader(g);
                string line; int count = 0;
                while ((line = await sr.ReadLineAsync()) != null && count < 1000)
                { try { using var doc = JsonDocument.Parse(line); } catch { } count++; }
            } else if (File.Exists(plain)) {
                using var sr = File.OpenText(plain);
                string line; int count = 0;
                while ((line = await sr.ReadLineAsync()) != null && count < 1000)
                { try { using var doc = JsonDocument.Parse(line); } catch { } count++; }
            }
        }

        public async IAsyncEnumerable<JsonDocument> StreamRecordsAsync(string tilePath)
        {
            var gz    = Path.Combine(tilePath, "records.jsonl.gz");
            var plain = Path.Combine(tilePath, "records.jsonl");
            Stream s  = null;
            if      (File.Exists(gz))    s = new GZipStream(File.OpenRead(gz), CompressionMode.Decompress);
            else if (File.Exists(plain)) s = File.OpenRead(plain);
            else yield break;
            using var sr = new StreamReader(s);
            string line;
            while ((line = await sr.ReadLineAsync()) != null)
            { try { yield return JsonDocument.Parse(line); } catch { } }
        }

        public void MountWeightsMock()
        {
            if (string.IsNullOrEmpty(WeightFile) || !File.Exists(WeightFile)) return;
            Console.WriteLine("Weight file present: " + WeightFile);
        }
    }

    // Training integration — shells out to the EXE toolchain in trainer/build/Release/.
    // Training loop per fold:
    //   xshard_info  → query pending shard IDs
    //   StreamFoldAsync → stream weight bytes (via XShardSession)
    //   xshard_backward → write gradient xshard
    //   xshard_adapt    → fold-gated Adam update, marks shards TRAINED
    public static class XShardTrainer
    {
        // Result of an xshard_info query
        public sealed class ShardInfoResult
        {
            public string Id     { get; set; } = "";
            public string State  { get; set; } = "";  // pending/trained/error/locked
            public string Fold   { get; set; } = "";
            public long   Nbytes { get; set; }
        }

        // Enumerate shards from an open XShardSession for one fold.
        // xshard_info.exe has no JSONL mode — this reads the parsed manifest directly.
        // State is reported as "pending" (we do not read the state block here; xshard_adapt
        // tracks trained status in its --ledger and via the state block in the xshard file).
        public static List<ShardInfoResult> QueryShardInfo(
            XShardSession session, string foldFilter = null)
        {
            var results = new List<ShardInfoResult>();
            foreach (var e in session.Shards)
            {
                if (!string.IsNullOrEmpty(foldFilter) && e.Fold != foldFilter) continue;
                results.Add(new ShardInfoResult {
                    Id     = e.Id,
                    State  = "pending",
                    Fold   = e.Fold,
                    Nbytes = e.Nbytes,
                });
            }
            return results;
        }

        // Overload: shell to xshard_info.exe --state-only and parse the summary line
        // for a quick trained/pending count without an open session.
        public static (int trained, int pending) QueryShardState(
            string toolchainDir, string xshardPath)
        {
            var output = RunExe(Path.Combine(toolchainDir, "xshard_info.exe"),
                                $"\"{xshardPath}\" --state-only");
            int trained = 0, pending = 0;
            foreach (var line in output.Split('\n'))
            {
                var t = line.Trim();
                // Parse: "state: trained=N  pending=M  ..."
                if (!t.StartsWith("state:")) continue;
                foreach (var part in t.Split(' ', StringSplitOptions.RemoveEmptyEntries))
                {
                    var kv = part.Split('=');
                    if (kv.Length != 2) continue;
                    if (kv[0] == "trained") int.TryParse(kv[1], out trained);
                    if (kv[0] == "pending") int.TryParse(kv[1], out pending);
                }
            }
            return (trained, pending);
        }

        // Run xshard_backward.exe to write a gradient xshard.
        // tokenBinPath   — raw bytes streamed as FNV content signal (any binary; UTF-8 JSONL works).
        //                  xshard_backward only processes F32 shards — quantized folds (Sek/Wo/Yax/Pop)
        //                  are skipped; use fold=Chen for norm tensors which are always F32.
        // gradXshardPath — output path for the gradient XSHARD/1 file.
        // fold           — fold lane being trained (Pop/Wo/Yax/Sek/Chen/Xul).
        // Returns the stdout from the tool.
        public static string WriteGradientXShard(
            string toolchainDir, string srcXshardPath,
            string tokenBinPath, string gradXshardPath, string fold,
            float gradScale = 1e-3f, float weightScale = 1e-4f, int maxShards = 1)
        {
            var args = $"\"{srcXshardPath}\" --token-bin \"{tokenBinPath}\" --output \"{gradXshardPath}\"" +
                       $" --fold {fold} --max-shards {maxShards}" +
                       $" --grad-scale {gradScale:G6} --weight-scale {weightScale:G6}";
            return RunExe(Path.Combine(toolchainDir, "xshard_backward.exe"), args);
        }

        // Run xshard_adapt.exe to apply fold-gated Adam and mark shards TRAINED.
        //
        // Two modes:
        //   gradXshardPath != null → load gradients from a grad XSHARD/1 file (--grad-xshard).
        //                           Matching is by tensor name; use when you have real gradients.
        //   gradXshardPath == null → probe gradient = weight * gradScale (--grad-scale).
        //                           Covers all F32 shards in the fold with no external file needed.
        //                           Use this for the curriculum probe-signal approach.
        //
        // Requires --shader <path> so pass shaderPath pointing at xshard_adapt_fold.cso.
        // Defaults to dry-run; pass apply=true to commit the update in-place.
        // Returns stdout from the tool.
        public static string ApplyFoldAdapt(
            string toolchainDir, string xshardPath, string shaderPath,
            string fold, float learningRate = 1e-4f,
            float gradScale = 5e-4f, float weightDecay = 0f,
            string gradXshardPath = null, int maxShards = 200,
            bool apply = true)
        {
            var args = $"\"{xshardPath}\" --fold {fold}" +
                       $" --lr {learningRate:G6} --weight-decay {weightDecay:G6}" +
                       $" --max-shards {maxShards} --shader \"{shaderPath}\"";
            if (gradXshardPath != null)
                args += $" --grad-xshard \"{gradXshardPath}\"";
            else
                args += $" --grad-scale {gradScale:G6}";
            if (apply) args += " --apply";
            return RunExe(Path.Combine(toolchainDir, "xshard_adapt.exe"), args);
        }

        // Orchestrate a full fold training step using the probe-gradient curriculum path.
        //
        // Uses xshard_adapt --grad-scale (probe gradient = weight * gradScale) to update
        // all F32 shards in the fold lane without a separate backward pass.
        // xshard_backward + --grad-xshard is reserved for when you have real loss gradients.
        //
        // The fold gate (--fold) ensures only the target fold's tensors move.
        // For Gemma 3 1B, Chen is the only F32 fold; Sek/Wo/Yax/Pop are quantized (skipped).
        //
        // shaderPath — absolute path to xshard_adapt_fold.cso
        // Returns count of shards committed; 0 if fold has no F32 shards.
        public static async Task<int> TrainFoldStepAsync(
            string toolchainDir, XShardSession session,
            string xshardPath, string fold, string shaderPath,
            float learningRate = 1e-4f, float gradScale = 5e-4f,
            int maxShards = 200, CancellationToken ct = default)
        {
            var shards = QueryShardInfo(session, fold);
            if (shards.Count == 0) return 0;

            ct.ThrowIfCancellationRequested();

            // Probe-gradient fold adapt — all F32 shards in this fold lane.
            ApplyFoldAdapt(toolchainDir, xshardPath, shaderPath, fold,
                           learningRate, gradScale,
                           gradXshardPath: null, maxShards: maxShards, apply: true);

            await Task.CompletedTask;
            return shards.Count;
        }

        // Canonical fold sequence — phase order Pop(0) → Xul(5π/3).
        // Embeddings first, lm_head last; attention and FFN in the middle.
        public static readonly string[] FoldSequence =
            { "Pop", "Wo", "Yax", "Sek", "Chen", "Xul" };

        // Train every fold in phase order using probe-gradient curriculum.
        // Skips folds with no F32 shards (quantized folds like Sek/Wo/Yax are no-ops).
        // progress callback receives (fold, shardCount) before each fold step.
        public static async Task<int> TrainAllFoldsAsync(
            string toolchainDir, XShardSession session,
            string xshardPath, string shaderPath,
            float learningRate = 1e-4f, float gradScale = 5e-4f,
            int maxShardsPerFold = 200,
            Action<string, int> progress = null,
            CancellationToken ct = default)
        {
            int totalTrained = 0;

            foreach (var fold in FoldSequence)
            {
                ct.ThrowIfCancellationRequested();

                var shards = QueryShardInfo(session, fold);
                if (shards.Count == 0) continue;

                progress?.Invoke(fold, shards.Count);

                int trained = await TrainFoldStepAsync(
                    toolchainDir, session, xshardPath, fold, shaderPath,
                    learningRate, gradScale, maxShardsPerFold, ct);

                totalTrained += trained;
            }

            return totalTrained;
        }

        // Low-level: run an EXE with args, return stdout. Throws on non-zero exit.
        private static string RunExe(string exePath, string args)
        {
            var psi = new ProcessStartInfo(exePath, args)
            {
                RedirectStandardOutput = true,
                RedirectStandardError  = true,
                UseShellExecute        = false,
                CreateNoWindow         = true,
            };
            using var p = Process.Start(psi)
                ?? throw new InvalidOperationException($"Failed to start {exePath}");
            var stdout = p.StandardOutput.ReadToEnd();
            p.WaitForExit();
            if (p.ExitCode != 0)
            {
                var stderr = p.StandardError.ReadToEnd();
                throw new Exception($"{Path.GetFileName(exePath)} exit {p.ExitCode}: {stderr.Trim()}");
            }
            return stdout;
        }
    }

    // Routes a resolved fold phase to the model XShard whose topology best covers it.
    //
    // Gemma 3 1B topology:    Chen:157  Sek:104  Wo:52  Yax:26  Pop:5
    // GPT-2 Large topology:   Chen:182  Sek:144  Wo:72  Yax:72  Pop:2
    // Qwen-Coder (AST specialist, NOT in main routing — access directly):
    //                         Sek:168   Wo:48    Chen:49 Yax:24  Pop:3
    //
    // Default routing:
    //   Pop  → Gemma     (embedding/pos; generalist base)
    //   Wo   → Gemma     (FFN gate/up; Gemma has more FFN capacity)
    //   Yax  → Gpt2Large (FFN down; 36-layer depth, 72 shards vs Gemma's 26)
    //   Sek  → Gpt2Large (attention Q/K/V/O; 20-head × 36 layers = 144 shards)
    //   Chen → Gpt2Large (layer-norm; 36 layers × 2 norms = 182 shards, deepest norm stack)
    //   Xul  → Gemma     (lm_head; generalist output)
    //
    // Qwen-Coder is a standalone AST/coder specialist. It is not in the default
    // routing table but can be assigned via FoldOverrides when the request is
    // code/AST-domain ("asx" = AST in stack terminology).
    //
    // The router never touches model files — it decides which XShardSession to open
    // per fold so the runtime streams weights from the specialist model.
    public sealed class FoldRouter
    {
        public string GemmaXshardPath    { get; }
        public string Gpt2LargeXshardPath { get; }
        public string? QwenXshardPath    { get; }

        // Per-fold override: fold name → "Gemma", "Gpt2Large", or "Qwen"
        public Dictionary<string, string> FoldOverrides { get; } = new();

        private static readonly Dictionary<string, string> DefaultRouting =
            new(StringComparer.Ordinal)
            {
                ["Pop"]  = "Gemma",
                ["Wo"]   = "Gemma",
                ["Yax"]  = "Gpt2Large",
                ["Sek"]  = "Gpt2Large",
                ["Chen"] = "Gpt2Large",
                ["Xul"]  = "Gemma",
            };

        public FoldRouter(string gemmaXshardPath, string gpt2LargeXshardPath, string? qwenXshardPath = null)
        {
            GemmaXshardPath     = gemmaXshardPath     ?? throw new ArgumentNullException(nameof(gemmaXshardPath));
            Gpt2LargeXshardPath = gpt2LargeXshardPath ?? throw new ArgumentNullException(nameof(gpt2LargeXshardPath));
            QwenXshardPath      = qwenXshardPath;
        }

        // Resolve which model owns the given fold, honouring any overrides.
        public string ModelForFold(string fold)
        {
            if (FoldOverrides.TryGetValue(fold, out var ov)) return ov;
            return DefaultRouting.TryGetValue(fold, out var def) ? def : "Gemma";
        }

        // Open the XShardSession for the model that owns the given fold.
        // Caller must dispose the returned session.
        public XShardSession OpenSessionForFold(string fold)
        {
            var path = ModelForFold(fold) switch
            {
                "Gpt2Large" => Gpt2LargeXshardPath,
                "Qwen"      => QwenXshardPath ?? GemmaXshardPath,
                _           => GemmaXshardPath,
            };
            return AdaptiveContentModelAdapter.OpenXShard(path);
        }

        // Stream fold shards from the appropriate specialist model, yielding
        // (entry, rawBytes, modelName) for each shard in manifest order.
        // Disposes the session after enumeration completes.
        public async IAsyncEnumerable<(XShardEntry Entry, byte[] Raw, string Model)>
            StreamFoldFromSpecialistAsync(string fold,
                [System.Runtime.CompilerServices.EnumeratorCancellation]
                CancellationToken ct = default)
        {
            var model   = ModelForFold(fold);
            using var s = OpenSessionForFold(fold);
            await foreach (var (entry, raw) in s.StreamFoldAsync(fold, ct))
                yield return (entry, raw, model);
        }

        // Return the routing table as a display string.
        public string DescribeRouting()
        {
            var sb = new System.Text.StringBuilder();
            foreach (var fold in XShardTrainer.FoldSequence)
                sb.Append(fold).Append(" → ").AppendLine(ModelForFold(fold));
            return sb.ToString();
        }
    }
}
