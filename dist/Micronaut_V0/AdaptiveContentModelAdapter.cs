using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Text.Json;
using System.Threading.Tasks;

namespace Micronaut.Streaming.Adapter
{
    // Minimal adapter stub showing how to map tile lanes to runtime frames and stream records without loading entire tiles into memory.
    public class AdaptiveContentModelAdapter
    {
        public string TilesRoot { get; }
        public string WeightFile { get; }
        public Dictionary<string,string> LaneToFrame { get; } = new Dictionary<string,string>();

        public AdaptiveContentModelAdapter(string tilesRoot, string weightFile)
        {
            TilesRoot = tilesRoot ?? throw new ArgumentNullException(nameof(tilesRoot));
            WeightFile = weightFile;
            // default lane->frame map (can be replaced by manifest)
            LaneToFrame["code"] = "COMPUTE_FRAME";
            LaneToFrame["dialog"] = "DIALOG_FRAME";
            LaneToFrame["instruction"] = "INSTRUCTION_FRAME";
            LaneToFrame["metadata"] = "META_FRAME";
        }

        public void MapLanes(Dictionary<string,string> mapping)
        {
            if(mapping==null) return;
            foreach(var kv in mapping) LaneToFrame[kv.Key] = kv.Value;
        }

        // Prefetch top-N tiles for a given lane (non-blocking wrapper)
        public async Task PrefetchTopTilesAsync(string lane, int topN = 1)
        {
            var tiles = Directory.GetDirectories(TilesRoot, "tile_*");
            // simple heuristic: look at tile.meta.json lane counts and pick topN
            var candidates = new List<string>();
            foreach(var t in tiles)
            {
                try{
                    var metaPath = Path.Combine(t, "tile.meta.json");
                    if(!File.Exists(metaPath)) continue;
                    using var fs = File.OpenRead(metaPath);
                    var meta = JsonDocument.Parse(fs);
                    if(meta.RootElement.TryGetProperty("lanes", out var lanes) ){
                        foreach(var l in lanes.EnumerateArray()){
                            if(l.GetString()==lane){ candidates.Add(t); break; }
                        }
                    }
                }catch{}
                if(candidates.Count>=topN) break;
            }

            // start background tasks to stream the records (lightweight)
            var tasks = new List<Task>();
            foreach(var c in candidates) tasks.Add(PrimeTileAsync(c));
            await Task.WhenAll(tasks);
        }

        // Prime (stream) a tile into memory / lane buffer without fully materializing it on the heap.
        public async Task PrimeTileAsync(string tilePath)
        {
            var gz = Path.Combine(tilePath, "records.jsonl.gz");
            var plain = Path.Combine(tilePath, "records.jsonl");
            if(File.Exists(gz)){
                using var fs = File.OpenRead(gz);
                using var g = new GZipStream(fs, CompressionMode.Decompress);
                using var sr = new StreamReader(g);
                string line;
                int count=0;
                while((line = await sr.ReadLineAsync()) != null && count < 1000)
                {
                    // lightweight parse to warm caches; do not keep the object if not needed
                    try{ using var doc = JsonDocument.Parse(line); }catch{} 
                    count++;
                }
            } else if(File.Exists(plain)){
                using var sr = File.OpenText(plain);
                string line; int count=0;
                while((line = await sr.ReadLineAsync()) != null && count < 1000){ try{ using var doc = JsonDocument.Parse(line);}catch{} count++; }
            }
            // Optionally signal runtime to map this tile into the lane-specific frame buffer
        }

        // Example: stream records from a tile and invoke a callback per-record
        public async IAsyncEnumerable<JsonDocument> StreamRecordsAsync(string tilePath)
        {
            var gz = Path.Combine(tilePath, "records.jsonl.gz");
            var plain = Path.Combine(tilePath, "records.jsonl");
            Stream s = null;
            if(File.Exists(gz)) s = new GZipStream(File.OpenRead(gz), CompressionMode.Decompress);
            else if(File.Exists(plain)) s = File.OpenRead(plain);
            else yield break;

            using var sr = new StreamReader(s);
            string line;
            while((line = await sr.ReadLineAsync()) != null)
            {
                try{ var doc = JsonDocument.Parse(line); yield return doc; }
                catch{ continue; }
            }
        }

        // Example mount: memory-map the weights file (placeholder - requires platform-specific API)
        public void MountWeightsMock()
        {
            if(string.IsNullOrEmpty(WeightFile) || !File.Exists(WeightFile)) return;
            // In production, map file into address space or load into VRAM via K'uhul DX12 path.
            Console.WriteLine("Weight file present: " + WeightFile);
        }
    }
}
