// gpt2_train_main.cpp — entry point for D3D11 GPU trainer
//
// Usage (fine-tune from existing weights):
//   gpt2_trainer --model models/from_zero_v0.6.safetensors
//                --data  E:/data/ts_corpus/ts_corpus_packed.bin
//                --out   models/ts_coder_medium.safetensors
//                --steps 1000 --batch 4 --block 128 --lr 3e-5
//
// Usage (train from scratch with preset architecture):
//   gpt2_trainer --model gpt2-medium
//                --data  E:/data/ts_corpus/ts_corpus_packed.bin
//                --out   ts_coder_medium.safetensors
//                --steps 5000 --batch 4 --seq-len 256 --lr 2e-5
//
// Preset names: distilgpt2 | gpt2-small | gpt2-medium | gpt2-large
// --seq-len and --block are aliases for the same field.
// --no-gpu-fwd forces Phase 1 (CPU fwd/bwd, GPU Adam). Auto-set for medium+.
// --completion-only masks loss to <INSTRUCT>...</INSTRUCT> spans (skips <USER> tokens).
// Data format: headed (n_seq uint32, seq_len uint32, data) OR flat int32 stream (auto-detected).

#include "gpt2_trainer.h"
#include "d3d11_engine.h"
#include <iostream>
#include <string>
#include <cstring>
#include <set>

static const std::set<std::string> PRESETS = {
    "distilgpt2", "gpt2-small", "gpt2-medium", "gpt2-large"
};

static std::string argval(int argc, char** argv, const char* key, const char* def = "") {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i+1];
    return def;
}

static bool argflag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return true;
    return false;
}

int main(int argc, char** argv) {
    TrainerConfig cfg;

    // --model: preset name OR safetensors file path
    std::string model_arg = argval(argc, argv, "--model",
        "C:/Users/canna/.micronaut/models/cli_coder_gpt2/model.safetensors");

    if (PRESETS.count(model_arg)) {
        cfg.preset_name = model_arg;
        cfg.model_path  = "";   // random init — no file
        // Phase 1 required for medium+large (VRAM budget)
        if (model_arg == "gpt2-medium" || model_arg == "gpt2-large")
            cfg.use_gpu_fwd = false;
    } else {
        cfg.model_path  = model_arg;
        cfg.preset_name = "";
        cfg.use_gpu_fwd = true;
    }

    cfg.data_path    = argval(argc, argv, "--data",
                              "C:/Users/canna/.micronaut/train/out/tokens.bin");
    cfg.output_path  = argval(argc, argv, "--out",
                              "C:/Users/canna/.micronaut/models/cli_coder_dx11/model.safetensors");
    cfg.max_steps    = std::stoi(argval(argc, argv, "--steps",    "1000"));
    cfg.batch_size   = std::stoi(argval(argc, argv, "--batch",    "4"));
    cfg.lr           = std::stof(argval(argc, argv, "--lr",       "3e-5"));
    cfg.save_every   = std::stoi(argval(argc, argv, "--save-every","200"));

    // --seq-len and --block are the same field; --seq-len takes priority if both given
    {
        std::string seq_len = argval(argc, argv, "--seq-len", "");
        std::string block   = argval(argc, argv, "--block",   "128");
        cfg.block_size = std::stoi(seq_len.empty() ? block : seq_len);
    }

    // Override Phase 1 from command line
    if (argflag(argc, argv, "--no-gpu-fwd"))      cfg.use_gpu_fwd      = false;
    if (argflag(argc, argv, "--gpu-fwd"))          cfg.use_gpu_fwd      = true;
    if (argflag(argc, argv, "--completion-only"))  cfg.completion_only  = true;
    if (argflag(argc, argv, "--no-shuffle"))       cfg.no_shuffle       = true;

    // Print resolved config before init
    std::cerr << "[main] model:   "
              << (cfg.preset_name.empty() ? cfg.model_path : cfg.preset_name + " (scratch)")
              << "\n";
    std::cerr << "[main] data:    " << cfg.data_path   << "\n";
    std::cerr << "[main] out:     " << cfg.output_path << "\n";
    std::cerr << "[main] steps:   " << cfg.max_steps   << "  lr=" << cfg.lr
              << "  batch=" << cfg.batch_size
              << "  seq=" << cfg.block_size
              << "  phase=" << (cfg.use_gpu_fwd ? "3(full-GPU)" : "1(CPU-fwd)") << "\n";

    // VRAM estimate for preset
    if (!cfg.preset_name.empty()) {
        GPT2Config pset = GPT2Config::from_preset(cfg.preset_name);
        std::cerr << "[main] preset:  n_layer=" << pset.n_layer
                  << " n_head=" << pset.n_head
                  << " n_embd=" << pset.n_embd
                  << " → est. " << pset.vram_estimate_mb() << " MB VRAM\n";
    }

    // Init D3D11
    D3D11Engine d11;
    if (!d11.init(/*forceWarp=*/false, /*verbose=*/true)) {
        std::cerr << "[main] D3D11 init failed: " << d11.initReason() << "\n";
        return 1;
    }
    std::cerr << "[main] D3D11 adapter: " << d11.adapterName()
              << " (feature level " << (int)d11.featureLevel() << ")\n";

    GPT2Trainer trainer(&d11);
    if (!trainer.init(cfg)) {
        std::cerr << "[main] trainer init failed\n";
        return 1;
    }

    std::cerr << "[main] starting training: " << cfg.max_steps << " steps\n";
    trainer.train();

    std::cerr << "[main] done\n";
    return 0;
}
