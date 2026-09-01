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

        // Run xshard_info.exe and parse its JSONL output.
        // Each output line is a JSON object with at minimum {id, state, fold, nbytes}.
        // Pass foldFilter to limit to one fold lane; null = all shards.
        public static List<ShardInfoResult> QueryShardInfo(
            string toolchainDir, string xshardPath, string foldFilter = null)
        {
            var args = $"\"{xshardPath}\"";
            if (!string.IsNullOrEmpty(foldFilter))
                args += $" --fold {foldFilter}";

            var output = RunExe(Path.Combine(toolchainDir, "xshard_info.exe"), args);
            var results = new List<ShardInfoResult>();

            foreach (var line in output.Split('\n', StringSplitOptions.RemoveEmptyEntries))
            {
                try
                {
                    using var doc = JsonDocument.Parse(line.Trim());
                    var r = new ShardInfoResult();
                    var root = doc.RootElement;
                    if (root.TryGetProperty("id",     out var v)) r.Id     = v.GetString() ?? "";
                    if (root.TryGetProperty("state",  out v))     r.State  = v.GetString() ?? "";
                    if (root.TryGetProperty("fold",   out v))     r.Fold   = v.GetString() ?? "";
                    if (root.TryGetProperty("nbytes", out v))     r.Nbytes = v.GetInt64();
                    results.Add(r);
                }
                catch { }
            }
            return results;
        }

        // Run xshard_backward.exe to write a gradient xshard.
        // gradXshardPath — output path for the gradient XSHARD/1 file.
        // fold           — fold lane being trained (Pop/Wo/Yax/Sek/Chen/Xul).
        // Returns the stdout from the tool (may include diagnostic JSON).
        public static string WriteGradientXShard(
            string toolchainDir, string srcXshardPath,
            string gradXshardPath, string fold)
        {
            var args = $"\"{srcXshardPath}\" --grad-out \"{gradXshardPath}\" --fold {fold}";
            return RunExe(Path.Combine(toolchainDir, "xshard_backward.exe"), args);
        }

        // Run xshard_adapt.exe to apply fold-gated Adam and mark shards TRAINED.
        // learningRate — Adam α (default 1e-4).
        // beta1/beta2  — Adam momentum params.
        // Returns stdout from the tool.
        public static string ApplyFoldAdapt(
            string toolchainDir, string xshardPath, string gradXshardPath,
            string fold, float learningRate = 1e-4f,
            float beta1 = 0.9f, float beta2 = 0.999f)
        {
            var args = $"\"{xshardPath}\" --grad \"{gradXshardPath}\" --fold {fold}" +
                       $" --lr {learningRate:G6} --beta1 {beta1:G6} --beta2 {beta2:G6}";
            return RunExe(Path.Combine(toolchainDir, "xshard_adapt.exe"), args);
        }

        // Orchestrate a full fold training step:
        //   1. Query pending shards via xshard_info
        //   2. Stream fold weights via XShardSession (caller supplies the session)
        //   3. Write gradient xshard via xshard_backward
        //   4. Apply fold-gated Adam via xshard_adapt
        // Returns count of pending shards processed; 0 if fold is already fully trained.
        public static async Task<int> TrainFoldStepAsync(
            string toolchainDir, XShardSession session,
            string xshardPath, string fold,
            float learningRate = 1e-4f,
            CancellationToken ct = default)
        {
            var pending = QueryShardInfo(toolchainDir, xshardPath, fold)
                          .FindAll(s => s.State == "pending");

            if (pending.Count == 0) return 0;

            // Stream fold weights through the session so the caller's training loop
            // can access raw quantized bytes fold-by-fold.
            int streamed = 0;
            await foreach (var (entry, _) in session.StreamFoldAsync(fold, ct))
            {
                ct.ThrowIfCancellationRequested();
                streamed++;
            }

            // Gradient write — grad output lives beside the source file.
            var gradPath = xshardPath.Replace(".xshard", $".{fold}.grad.xshard");
            WriteGradientXShard(toolchainDir, xshardPath, gradPath, fold);

            // Fold-gated Adam update + TRAINED marking.
            ApplyFoldAdapt(toolchainDir, xshardPath, gradPath, fold, learningRate);

            return pending.Count;
        }

        // Canonical fold sequence — phase order Pop(0) → Xul(5π/3).
        // Embeddings first, lm_head last; attention and FFN in the middle.
        public static readonly string[] FoldSequence =
            { "Pop", "Wo", "Yax", "Sek", "Chen", "Xul" };

        // Train every fold in phase order, one fold at a time.
        // Skips folds with no pending shards (already trained or no tensors in that lane).
        // progress callback receives (fold, pendingCount) before each fold step.
        public static async Task<int> TrainAllFoldsAsync(
            string toolchainDir, XShardSession session,
            string xshardPath,
            float learningRate = 1e-4f,
            Action<string, int> progress = null,
            CancellationToken ct = default)
        {
            int totalTrained = 0;

            foreach (var fold in FoldSequence)
            {
                ct.ThrowIfCancellationRequested();

                var pending = QueryShardInfo(toolchainDir, xshardPath, fold)
                              .FindAll(s => s.State == "pending");

                if (pending.Count == 0) continue;

                progress?.Invoke(fold, pending.Count);

                int trained = await TrainFoldStepAsync(
                    toolchainDir, session, xshardPath, fold, learningRate, ct);

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
}
