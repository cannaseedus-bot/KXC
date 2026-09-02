#!/usr/bin/env python3
"""
xshard_chat.py -- Chat with fold-trained xshard models.

Three separate forward passes for the three model families:
  GPT-2 Large  : 36L, n_embd=1280, n_head=20, tied lm_head, standard attn
  Gemma 3 1B   : 26L, n_embd=1152, n_q_heads=4, n_kv_heads=1, head_dim=256,
                  RMSNorm, SwiGLU, GQA, RoPE, QK-norm, 4 LNs/layer
  Qwen 2 0.5B  : 24L, n_embd=896,  n_q_heads=14, n_kv_heads=2, head_dim=64,
                  RMSNorm, SwiGLU, GQA, RoPE, biased QKV, 2 LNs/layer

Fold routing (mirrors FoldRouter.DefaultRouting):
  Pop / Wo / Xul   -> Gemma
  Yax / Sek / Chen -> GPT-2 Large
  AST mode: Sek + Wo -> Qwen

Usage:
  python tools/xshard_chat.py                  # GPT-2 Large REPL (default)
  python tools/xshard_chat.py --model gemma    # Gemma REPL
  python tools/xshard_chat.py --model qwen     # Qwen REPL
  python tools/xshard_chat.py --fold-route     # auto-route by fold
  python tools/xshard_chat.py --serve          # OpenAI-compat HTTP :9100
  python tools/xshard_chat.py --serve --fold-route --port 9100
"""

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
#  Xshard paths
# ---------------------------------------------------------------------------
XSHARD_PATHS = {
    "gpt2large": Path("E:/models/GPT2/lg-GPT2/lg-gpt2-f32.xshard"),
    "gemma":     Path("E:/models/GPT2/gemma-3-1b-it/gemma-3-1b-it-f32.xshard"),
    "qwen":      Path("E:/models/GPT2/qwen/qwen-f32.xshard"),
}

XSHARD_MAGIC = b'XSHD'

DEFAULT_ROUTING = {
    "Pop":  "gemma",
    "Wo":   "gemma",
    "Yax":  "gpt2large",
    "Sek":  "gpt2large",
    "Chen": "gpt2large",
    "Xul":  "gemma",
}
AST_OVERRIDES = {"Sek": "qwen", "Wo": "qwen"}

# ---------------------------------------------------------------------------
#  Xshard loader
# ---------------------------------------------------------------------------

def load_xshard(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    print(f"[xshard] loading {path.name}  ({path.stat().st_size/1024/1024:.0f} MB)", flush=True)
    with open(path, "rb") as f:
        if f.read(4) != XSHARD_MAGIC:
            raise ValueError(f"Bad magic in {path.name}")
        f.read(4)
        mlen = struct.unpack("<Q", f.read(8))[0]
        manifest = json.loads(f.read(mlen))
        data_start = manifest["data_start"]

        from collections import defaultdict
        groups: dict[str, list] = defaultdict(list)
        for s in manifest["shards"]:
            groups[s["tensor_name"]].append(s)

        weights: dict[str, np.ndarray] = {}
        for tname, shards in groups.items():
            shards = sorted(shards, key=lambda s: s.get("shard_index", 0))
            chunks = []
            for s in shards:
                f.seek(data_start + s["offset"])
                raw = f.read(s["nbytes"])
                chunks.append(_decode_raw(raw, s.get("dtype", "F32")))
            weights[tname] = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)

    print(f"  {len(weights)} tensors", flush=True)
    return weights


def _decode_raw(raw: bytes, dtype_str: str) -> np.ndarray:
    if dtype_str == "F32":
        return np.frombuffer(raw, dtype=np.float32).copy()
    if dtype_str in ("F16", "BF16"):
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if dtype_str == "Q8_0":
        n_blocks = len(raw) // 34
        out = np.empty(n_blocks * 32, dtype=np.float32)
        for i in range(n_blocks):
            b = raw[i*34:(i+1)*34]
            scale = np.frombuffer(b[:2], dtype=np.float16)[0].astype(np.float32)
            out[i*32:(i+1)*32] = np.frombuffer(b[2:], dtype=np.int8).astype(np.float32) * scale
        return out
    n = len(raw) // 4
    return np.frombuffer(raw[:n*4], dtype=np.float32).copy()

# ---------------------------------------------------------------------------
#  Shared math ops
# ---------------------------------------------------------------------------

def rms_norm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return w * x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def causal_attn(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    # Q/K/V: [n_heads, seq, head_dim]
    seq, d_k = Q.shape[1], Q.shape[2]
    scores = (Q @ K.transpose(0, 2, 1)) / math.sqrt(d_k)
    mask = np.triu(np.full((seq, seq), -1e10), k=1)
    return softmax(scores + mask) @ V


def rope_cos_sin(seq_len: int, head_dim: int, base: float = 10000.0):
    half = head_dim // 2
    theta = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    pos = np.arange(seq_len, dtype=np.float32)
    angles = np.outer(pos, theta)                   # [seq, half]
    return np.cos(angles), np.sin(angles)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    # x: [n_heads, seq, head_dim]
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

# ---------------------------------------------------------------------------
#  GPT-2 Large forward
# ---------------------------------------------------------------------------

def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x**3)))


def layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(-1, keepdims=True)
    var  = ((x - mean)**2).mean(-1, keepdims=True)
    return w * (x - mean) / np.sqrt(var + eps) + b


def gpt2_forward(tokens: list[int], w: dict, n_head: int, n_layer: int) -> np.ndarray:
    seq    = len(tokens)
    n_embd = w["transformer.wte.weight"].shape[1]
    d_head = n_embd // n_head
    x = w["transformer.wte.weight"][tokens] + w["transformer.wpe.weight"][:seq]

    for i in range(n_layer):
        p = f"transformer.h.{i}."
        a = layernorm(x, w[p+"ln_1.weight"], w[p+"ln_1.bias"])
        qkv = a @ w[p+"attn.c_attn.weight"] + w[p+"attn.c_attn.bias"]
        Q, K, V = np.split(qkv, 3, axis=-1)
        Q = Q.reshape(seq, n_head, d_head).transpose(1, 0, 2)
        K = K.reshape(seq, n_head, d_head).transpose(1, 0, 2)
        V = V.reshape(seq, n_head, d_head).transpose(1, 0, 2)
        attn = causal_attn(Q, K, V).transpose(1, 0, 2).reshape(seq, n_embd)
        x = x + attn @ w[p+"attn.c_proj.weight"] + w[p+"attn.c_proj.bias"]
        b2 = layernorm(x, w[p+"ln_2.weight"], w[p+"ln_2.bias"])
        h = b2 @ w[p+"mlp.c_fc.weight"] + w[p+"mlp.c_fc.bias"]
        x = x + gelu(h) @ w[p+"mlp.c_proj.weight"] + w[p+"mlp.c_proj.bias"]

    x = layernorm(x, w["transformer.ln_f.weight"], w["transformer.ln_f.bias"])
    return x[-1] @ w["transformer.wte.weight"].T     # tied lm_head

# ---------------------------------------------------------------------------
#  Gemma 3 1B forward
#  26L, n_embd=1152, n_q_heads=4, n_kv_heads=1, head_dim=256
#  4 norms/layer: input_layernorm, post_attention_layernorm,
#                 pre_feedforward_layernorm, post_feedforward_layernorm
#  QK-norm per head (Gemma 3 specific), GQA 4:1, RoPE
# ---------------------------------------------------------------------------

GEMMA_N_LAYER    = 26
GEMMA_N_Q_HEADS  = 4
GEMMA_N_KV_HEADS = 1
GEMMA_HEAD_DIM   = 256
GEMMA_ROPE_BASE  = 10000.0


def gemma3_forward(tokens: list[int], w: dict) -> np.ndarray:
    seq      = len(tokens)
    n_embd   = w["model.embed_tokens.weight"].shape[1]
    n_groups = GEMMA_N_Q_HEADS // GEMMA_N_KV_HEADS

    # Gemma scale factor on embeddings
    x = w["model.embed_tokens.weight"][tokens] * math.sqrt(n_embd)

    cos, sin = rope_cos_sin(seq, GEMMA_HEAD_DIM, GEMMA_ROPE_BASE)

    for i in range(GEMMA_N_LAYER):
        p = f"model.layers.{i}."

        # ── Attention ──────────────────────────────────────────────────────
        r = rms_norm(x, w[p+"input_layernorm.weight"])

        Q = r @ w[p+"self_attn.q_proj.weight"].T   # [seq, n_q*hd]
        K = r @ w[p+"self_attn.k_proj.weight"].T   # [seq, n_kv*hd]
        V = r @ w[p+"self_attn.v_proj.weight"].T   # [seq, n_kv*hd]

        Q = Q.reshape(seq, GEMMA_N_Q_HEADS,  GEMMA_HEAD_DIM).transpose(1, 0, 2)
        K = K.reshape(seq, GEMMA_N_KV_HEADS, GEMMA_HEAD_DIM).transpose(1, 0, 2)
        V = V.reshape(seq, GEMMA_N_KV_HEADS, GEMMA_HEAD_DIM).transpose(1, 0, 2)

        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        # Per-head QK-norm (Gemma 3 specific)
        Q = rms_norm(Q, w[p+"self_attn.q_norm.weight"])
        K = rms_norm(K, w[p+"self_attn.k_norm.weight"])

        # GQA: expand KV to match Q head count
        K = np.repeat(K, n_groups, axis=0)
        V = np.repeat(V, n_groups, axis=0)

        attn = causal_attn(Q, K, V).transpose(1, 0, 2).reshape(seq, GEMMA_N_Q_HEADS * GEMMA_HEAD_DIM)
        attn_out = attn @ w[p+"self_attn.o_proj.weight"].T

        x = x + rms_norm(attn_out, w[p+"post_attention_layernorm.weight"])

        # ── MLP ───────────────────────────────────────────────────────────
        r2       = rms_norm(x, w[p+"pre_feedforward_layernorm.weight"])
        gate     = r2 @ w[p+"mlp.gate_proj.weight"].T
        up       = r2 @ w[p+"mlp.up_proj.weight"].T
        mlp_out  = (silu(gate) * up) @ w[p+"mlp.down_proj.weight"].T

        x = x + rms_norm(mlp_out, w[p+"post_feedforward_layernorm.weight"])

    x = rms_norm(x, w["model.norm.weight"])
    return x[-1] @ w["model.embed_tokens.weight"].T  # tied lm_head

# ---------------------------------------------------------------------------
#  Qwen 2 0.5B forward
#  24L, n_embd=896, n_q_heads=14, n_kv_heads=2, head_dim=64
#  2 norms/layer: input_layernorm, post_attention_layernorm
#  biased QKV, GQA 7:1, RoPE base=1e6
# ---------------------------------------------------------------------------

QWEN_N_LAYER    = 24
QWEN_N_Q_HEADS  = 14
QWEN_N_KV_HEADS = 2
QWEN_HEAD_DIM   = 64
QWEN_ROPE_BASE  = 1_000_000.0


def qwen2_forward(tokens: list[int], w: dict) -> np.ndarray:
    seq      = len(tokens)
    n_groups = QWEN_N_Q_HEADS // QWEN_N_KV_HEADS

    x = w["model.embed_tokens.weight"][tokens]

    cos, sin = rope_cos_sin(seq, QWEN_HEAD_DIM, QWEN_ROPE_BASE)

    for i in range(QWEN_N_LAYER):
        p = f"model.layers.{i}."

        # ── Attention ──────────────────────────────────────────────────────
        r = rms_norm(x, w[p+"input_layernorm.weight"])

        Q = r @ w[p+"self_attn.q_proj.weight"].T + w[p+"self_attn.q_proj.bias"]
        K = r @ w[p+"self_attn.k_proj.weight"].T + w[p+"self_attn.k_proj.bias"]
        V = r @ w[p+"self_attn.v_proj.weight"].T + w[p+"self_attn.v_proj.bias"]

        Q = Q.reshape(seq, QWEN_N_Q_HEADS,  QWEN_HEAD_DIM).transpose(1, 0, 2)
        K = K.reshape(seq, QWEN_N_KV_HEADS, QWEN_HEAD_DIM).transpose(1, 0, 2)
        V = V.reshape(seq, QWEN_N_KV_HEADS, QWEN_HEAD_DIM).transpose(1, 0, 2)

        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        K = np.repeat(K, n_groups, axis=0)
        V = np.repeat(V, n_groups, axis=0)

        attn = causal_attn(Q, K, V).transpose(1, 0, 2).reshape(seq, QWEN_N_Q_HEADS * QWEN_HEAD_DIM)
        x = x + attn @ w[p+"self_attn.o_proj.weight"].T

        # ── MLP ───────────────────────────────────────────────────────────
        r2      = rms_norm(x, w[p+"post_attention_layernorm.weight"])
        gate    = r2 @ w[p+"mlp.gate_proj.weight"].T
        up      = r2 @ w[p+"mlp.up_proj.weight"].T
        x       = x + (silu(gate) * up) @ w[p+"mlp.down_proj.weight"].T

    x = rms_norm(x, w["model.norm.weight"])
    return x[-1] @ w["model.embed_tokens.weight"].T  # tied lm_head

# ---------------------------------------------------------------------------
#  Dispatch table
# ---------------------------------------------------------------------------

def _prefix_weights(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add transformer. prefix for GPT-2 xshard keys that lack it."""
    out = {}
    for k, v in raw.items():
        nk = k if k.startswith("transformer.") else "transformer." + k
        out[nk] = v
    return out


def forward(model_name: str, tokens: list[int], w: dict) -> np.ndarray:
    if model_name == "gpt2large":
        n_layer, n_embd, n_head = _detect_gpt2_arch(w)
        return gpt2_forward(tokens, w, n_head, n_layer)
    if model_name == "gemma":
        return gemma3_forward(tokens, w)
    if model_name == "qwen":
        return qwen2_forward(tokens, w)
    raise ValueError(f"Unknown model: {model_name}")


def _detect_gpt2_arch(w: dict) -> tuple[int, int, int]:
    n_embd  = w["transformer.wte.weight"].shape[1]
    n_layer = sum(1 for k in w if k.startswith("transformer.h.") and k.endswith("ln_1.weight"))
    n_head  = n_embd // 64
    return n_layer, n_embd, n_head

# ---------------------------------------------------------------------------
#  Tokenizers
# ---------------------------------------------------------------------------

_tok_cache: dict[str, object] = {}


def get_tokenizer(model_name: str):
    if model_name in _tok_cache:
        return _tok_cache[model_name]

    if model_name == "gpt2large":
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            _tok_cache[model_name] = ("tiktoken", enc, 50256)
            return _tok_cache[model_name]
        except ImportError:
            print("[warn] tiktoken not installed: pip install tiktoken")

    if model_name == "qwen":
        tok_path = XSHARD_PATHS["qwen"].parent / "tokenizer.json"
        if tok_path.exists():
            try:
                from tokenizers import Tokenizer
                enc = Tokenizer.from_file(str(tok_path))
                eos = enc.token_to_id("<|im_end|>") or enc.token_to_id("</s>") or 0
                _tok_cache[model_name] = ("hf", enc, eos)
                return _tok_cache[model_name]
            except ImportError:
                print("[warn] tokenizers not installed: pip install tokenizers")

    if model_name == "gemma":
        gguf_path = XSHARD_PATHS["gemma"].parent / "gemma-3-1b-it-q8_0.gguf"
        if gguf_path.exists():
            enc = _load_gemma_tokenizer(gguf_path)
            if enc:
                _tok_cache[model_name] = enc
                return enc

    # Fallback: GPT-2 tokenizer (output will be approximate)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        print(f"[warn] using GPT-2 tokenizer as fallback for {model_name}")
        _tok_cache[model_name] = ("tiktoken", enc, 50256)
        return _tok_cache[model_name]
    except ImportError:
        print("[warn] no tokenizer available — output will be token IDs")
        _tok_cache[model_name] = None
        return None


def _load_gemma_tokenizer(gguf_path: Path):
    """Extract SentencePiece tokenizer from GGUF metadata."""
    try:
        import tempfile, os
        gguf_py = Path(r"C:\Users\canna\.ASX.cpp\llama-b9968-bin-win-cpu-x64\llama.cpp\gguf-py")
        if not gguf_py.exists():
            raise FileNotFoundError("gguf-py not found")
        sys.path.insert(0, str(gguf_py))
        from gguf import GGUFReader
        reader = GGUFReader(str(gguf_path))
        sp_bytes = None
        for field in reader.fields.values():
            if "tokenizer" in field.name and "model" in field.name:
                sp_bytes = bytes(field.parts[-1])
                break
        if not sp_bytes:
            return None
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".model")
        tmp.write(sp_bytes); tmp.close()
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(tmp.name)
        os.unlink(tmp.name)
        eos = sp.PieceToId("</s>") or sp.PieceToId("<eos>") or 1
        return ("spm", sp, eos)
    except Exception as e:
        print(f"[warn] could not load Gemma tokenizer: {e}")
        return None


def enc_encode(tok, text: str) -> list[int]:
    if tok is None:
        return [ord(c) % 50257 for c in text[:512]]
    kind, enc, _ = tok
    if kind == "tiktoken":
        return enc.encode(text)
    if kind == "hf":
        return enc.encode(text).ids
    if kind == "spm":
        return enc.Encode(text)
    return []


def enc_decode(tok, ids: list[int]) -> str:
    if tok is None:
        return " ".join(str(t) for t in ids)
    kind, enc, _ = tok
    try:
        if kind == "tiktoken":
            return enc.decode(ids)
        if kind == "hf":
            return enc.decode(ids)
        if kind == "spm":
            return enc.Decode(ids)
    except Exception:
        return " ".join(str(i) for i in ids)
    return ""


def eos_id(tok) -> int:
    if tok is None:
        return 50256
    return tok[2]

# ---------------------------------------------------------------------------
#  Sampling + generation
# ---------------------------------------------------------------------------

def sample(logits: np.ndarray, temperature: float, top_k: int) -> int:
    logits = logits / max(temperature, 1e-6)
    if top_k > 1:
        kth = np.sort(logits)[-top_k]
        logits = np.where(logits < kth, -1e10, logits)
    p = softmax(logits)
    return int(np.random.choice(len(p), p=p))


def generate_tokens(prompt_ids: list[int], model_name: str, w: dict,
                    tok, max_new: int, temperature: float, top_k: int,
                    n_ctx: int = 1024) -> list[int]:
    tokens = list(prompt_ids)
    eos = eos_id(tok)
    for _ in range(max_new):
        logits = forward(model_name, tokens[-n_ctx:], w)
        nxt = sample(logits, temperature, top_k)
        tokens.append(nxt)
        print(".", end="", flush=True)
        if nxt == eos:
            break
    print()
    return tokens

# ---------------------------------------------------------------------------
#  Fold classifier
# ---------------------------------------------------------------------------

def get_fold(msg: str) -> str:
    l = msg.lower()
    if re.search(r'\b(ast|asx|parse|syntax|token|grammar|compile|ir)\b', l):               return "Sek"
    if re.search(r'\b(load|search|find|read|fetch|look|what|who|when|where|list|show|get)\b', l): return "Pop"
    if re.search(r'\b(build|create|write|code|implement|define|construct|make|generate|scaffold)\b', l): return "Wo"
    if re.search(r'\b(plan|predict|analyze|compare|evaluate|why|explain|design|think|reason)\b', l):    return "Yax"
    if re.search(r'\b(execute|run|transform|convert|apply|calculate|compute|dispatch|call)\b', l):       return "Sek"
    if re.search(r'\b(review|reflect|improve|optimize|refactor|summarize|check|validate|test)\b', l):   return "Chen"
    if re.search(r'\b(save|export|done|finish|complete|store|archive|commit|push)\b', l):                return "Xul"
    if re.match(r'^\s*(hello|hi|hey|yo|sup)\b', l):                                                     return "Pop"
    return "Sek"

# ---------------------------------------------------------------------------
#  Model cache
# ---------------------------------------------------------------------------

_model_cache: dict[str, tuple[dict, object]] = {}


def get_model(name: str) -> tuple[dict, object]:
    if name not in _model_cache:
        path = XSHARD_PATHS.get(name)
        if not path or not path.exists():
            raise FileNotFoundError(f"xshard not found for {name}: {path}")
        raw = load_xshard(path)
        if name == "gpt2large":
            w = _prefix_weights(raw)
            n_layer, n_embd, n_head = _detect_gpt2_arch(w)
            print(f"  GPT-2 Large: {n_layer}L  embd={n_embd}  heads={n_head}", flush=True)
        else:
            w = raw
        tok = get_tokenizer(name)
        _model_cache[name] = (w, tok)
    return _model_cache[name]

# ---------------------------------------------------------------------------
#  Prompt building
# ---------------------------------------------------------------------------

def build_prompt(model_name: str, history: list[tuple[str,str]], user: str) -> str:
    if model_name == "qwen":
        # Qwen2 ChatML format
        parts = ["<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"]
        for u, a in history[-3:]:
            parts.append(f"<|im_start|>user\n{u}<|im_end|>\n")
            parts.append(f"<|im_start|>assistant\n{a}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
        return "".join(parts)
    if model_name == "gemma":
        # Gemma instruct format
        parts = []
        for u, a in history[-3:]:
            parts.append(f"<start_of_turn>user\n{u}<end_of_turn>\n")
            parts.append(f"<start_of_turn>model\n{a}<end_of_turn>\n")
        parts.append(f"<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n")
        return "".join(parts)
    # GPT-2: plain turn format
    parts = []
    for u, a in history[-3:]:
        parts.append(f"User: {u}\nAssistant: {a}\n")
    parts.append(f"User: {user}\nAssistant:")
    return "".join(parts)


def trim_response(model_name: str, text: str) -> str:
    # Strip stop sequences
    stops = {
        "qwen":      ["<|im_end|>", "<|im_start|>", "\nUser:"],
        "gemma":     ["<end_of_turn>", "<start_of_turn>", "\nUser:"],
        "gpt2large": ["\nUser:", "\nAssistant:"],
    }
    for stop in stops.get(model_name, []):
        if stop in text:
            text = text[:text.index(stop)]
    return text.strip()

# ---------------------------------------------------------------------------
#  REPL
# ---------------------------------------------------------------------------

def chat_loop(args):
    routing = _build_routing(args)

    # Pre-load model(s)
    if args.fold_route:
        for m in set(routing.values()):
            get_model(m)
    else:
        get_model(args.model)

    print(f"\n[xshard-chat] fold-route={args.fold_route}  ast={args.ast}")
    print("Commands: /fold  /quit  ctrl-c\n")

    history: list[tuple[str,str]] = []
    last_fold = "Sek"

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]"); break
        if not user: continue
        if user == "/quit": break
        if user == "/fold": print(f"  last fold: {last_fold}"); continue

        fold = get_fold(user)
        last_fold = fold
        active = routing.get(fold, args.model) if args.fold_route else args.model

        w, tok = get_model(active)
        prompt = build_prompt(active, history, user)
        ids = enc_encode(tok, prompt)
        if len(ids) > 900:
            ids = ids[-900:]

        print(f"[{fold} -> {active}] generating...", end="", flush=True)
        out = generate_tokens(ids, active, w, tok,
                              max_new=args.max_new,
                              temperature=args.temperature,
                              top_k=args.top_k)
        new_ids = out[len(ids):]
        response = trim_response(active, enc_decode(tok, new_ids))

        print(f"Assistant [{fold}/{active}]: {response}\n")
        history.append((user, response))

# ---------------------------------------------------------------------------
#  HTTP server (OpenAI-compatible /v1/chat/completions)
# ---------------------------------------------------------------------------

def serve_loop(args):
    import http.server

    routing = _build_routing(args)
    if args.fold_route:
        for m in set(routing.values()):
            get_model(m)
    else:
        get_model(args.model)

    cfg = args

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *a): pass

        def do_GET(self):
            if self.path == "/v1/models":
                body = json.dumps({"object":"list","data":[{"id":"xshard-fold","object":"model"}]}).encode()
                self._ok(body)
            else:
                self.send_response(404); self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(length))
            except Exception:
                self.send_response(400); self.end_headers(); return

            messages    = req.get("messages", [])
            max_tokens  = min(req.get("max_tokens", cfg.max_new), 256)
            temperature = req.get("temperature", cfg.temperature)

            user_text = next((m["content"] for m in reversed(messages) if m.get("role")=="user"), "")
            fold      = get_fold(user_text)
            active    = routing.get(fold, cfg.model) if cfg.fold_route else cfg.model

            w, tok = get_model(active)

            # Rebuild history from messages for prompt
            history: list[tuple[str,str]] = []
            i = 0
            msgs = [m for m in messages if m.get("role") in ("user","assistant")]
            while i < len(msgs) - 1:
                if msgs[i].get("role")=="user" and msgs[i+1].get("role")=="assistant":
                    history.append((msgs[i]["content"], msgs[i+1]["content"]))
                    i += 2
                else:
                    i += 1

            prompt = build_prompt(active, history, user_text)
            ids = enc_encode(tok, prompt)
            if len(ids) > 900:
                ids = ids[-900:]

            print(f"  [{fold} -> {active}] {user_text[:60]!r}", flush=True)
            print(f"  generating (max={max_tokens})...", end="", flush=True)
            out = generate_tokens(ids, active, w, tok,
                                  max_new=max_tokens,
                                  temperature=temperature,
                                  top_k=40)
            new_ids = out[len(ids):]
            response = trim_response(active, enc_decode(tok, new_ids))
            print(f" {len(new_ids)}t", flush=True)

            body = json.dumps({
                "id": "xshard-1",
                "object": "chat.completion",
                "model": f"xshard-{active}-{fold}",
                "choices": [{"index":0, "message":{"role":"assistant","content":response}, "finish_reason":"stop"}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(new_ids), "total_tokens": len(ids)+len(new_ids)}
            }).encode()
            self._ok(body)

        def _ok(self, body: bytes):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n[xshard-serve] http://127.0.0.1:{args.port}")
    print(f"  fold-route={args.fold_route}  ast={args.ast}")
    print("  Ctrl-C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[stopped]")

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _build_routing(args) -> dict:
    if not args.fold_route:
        return {f: args.model for f in DEFAULT_ROUTING}
    r = dict(DEFAULT_ROUTING)
    if args.ast:
        r.update(AST_OVERRIDES)
    return r

# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Chat with fold-trained xshard models")
    ap.add_argument("--model",  default="gpt2large",
                    choices=["gpt2large", "gemma", "qwen"])
    ap.add_argument("--xshard", default=None,
                    help="Override xshard path for --model")
    ap.add_argument("--fold-route", action="store_true",
                    help="Route by fold: Pop/Wo/Xul=Gemma, Yax/Sek/Chen=GPT-2L, AST=Qwen")
    ap.add_argument("--ast", action="store_true",
                    help="AST override: Sek+Wo -> Qwen (requires --fold-route)")
    ap.add_argument("--serve",  action="store_true",
                    help="Run as OpenAI-compatible HTTP server")
    ap.add_argument("--port",   type=int, default=9100)
    ap.add_argument("--max-new",     type=int,   default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k",       type=int,   default=40)
    ap.add_argument("--seed",        type=int,   default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    if args.xshard:
        XSHARD_PATHS[args.model] = Path(args.xshard)

    if args.serve:
        serve_loop(args)
    else:
        chat_loop(args)


if __name__ == "__main__":
    main()
