#!/usr/bin/env python3
"""
xshard_chat.py -- Chat with xshard-resident fold models.

Loads a fold-tagged xshard file, reassembles tensors from shards,
runs a GPT-2 forward pass with tiktoken BPE, and provides a chat REPL
or an OpenAI-compatible HTTP server (--serve).

Fold routing (mirrors FoldRouter.DefaultRouting):
  Pop / Wo / Xul  -> Gemma xshard  (default to GPT-2 Large if Gemma not loaded)
  Yax / Sek / Chen -> GPT-2 Large xshard
  AST mode (--ast): Sek + Wo -> Qwen xshard

Usage:
  python tools/xshard_chat.py                            # GPT-2 Large REPL
  python tools/xshard_chat.py --model gemma              # Gemma REPL
  python tools/xshard_chat.py --fold-route               # auto-route by fold
  python tools/xshard_chat.py --serve                    # OpenAI-compat HTTP server :9100
  python tools/xshard_chat.py --serve --port 9200        # custom port
  python tools/xshard_chat.py --serve --fold-route       # server with fold routing
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
FOLD_SEQ = ["Pop", "Wo", "Yax", "Sek", "Chen", "Xul"]

DEFAULT_ROUTING = {
    "Pop":  "gpt2large",
    "Wo":   "gpt2large",
    "Yax":  "gpt2large",
    "Sek":  "gpt2large",
    "Chen": "gpt2large",
    "Xul":  "gpt2large",
}

AST_OVERRIDES = {"Sek": "qwen", "Wo": "qwen"}

# ---------------------------------------------------------------------------
#  Xshard loader
# ---------------------------------------------------------------------------

def load_xshard(path: Path) -> dict[str, np.ndarray]:
    """Read all shards from an xshard file and reassemble tensors."""
    path = Path(path)
    print(f"[xshard] loading {path.name}  ({path.stat().st_size / 1024 / 1024:.0f} MB)", flush=True)

    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != XSHARD_MAGIC:
            raise ValueError(f"Bad magic: {magic!r}")
        f.read(4)  # version + flags
        mlen = struct.unpack("<Q", f.read(8))[0]
        manifest = json.loads(f.read(mlen))
        data_start = manifest["data_start"]

        # Group sub-shards by tensor_name
        from collections import defaultdict
        groups: dict[str, list] = defaultdict(list)
        for s in manifest["shards"]:
            groups[s["tensor_name"]].append(s)

        weights: dict[str, np.ndarray] = {}
        for tensor_name, shards in groups.items():
            shards_sorted = sorted(shards, key=lambda s: s.get("shard_index", 0))
            chunks = []
            for s in shards_sorted:
                f.seek(data_start + s["offset"])
                raw = f.read(s["nbytes"])
                dtype_str = s.get("dtype", "F32")
                arr = _decode_raw(raw, dtype_str)
                chunks.append(arr)

            if len(chunks) == 1:
                weights[tensor_name] = chunks[0]
            else:
                # Concatenate sub-shards along axis 0
                weights[tensor_name] = np.concatenate(chunks, axis=0)

    print(f"  {len(weights)} tensors loaded", flush=True)
    return weights


def _decode_raw(raw: bytes, dtype_str: str) -> np.ndarray:
    """Decode raw bytes to float32 array (dequantize Q8_0 if needed)."""
    if dtype_str == "F32":
        return np.frombuffer(raw, dtype=np.float32).copy()
    if dtype_str in ("F16", "BF16"):
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        return arr
    if dtype_str == "Q8_0":
        # Q8_0 block: 2 bytes (f16 scale) + 32 bytes (int8 quants) = 34 bytes per block
        BLOCK = 34
        n_blocks = len(raw) // BLOCK
        out = np.empty(n_blocks * 32, dtype=np.float32)
        for i in range(n_blocks):
            b = raw[i * BLOCK: (i + 1) * BLOCK]
            scale = np.frombuffer(b[:2], dtype=np.float16)[0].astype(np.float32)
            quants = np.frombuffer(b[2:], dtype=np.int8).astype(np.float32)
            out[i * 32: (i + 1) * 32] = quants * scale
        return out
    # Fallback: treat as F32
    n = len(raw) // 4
    return np.frombuffer(raw[:n * 4], dtype=np.float32).copy()


# ---------------------------------------------------------------------------
#  Tensor name normalisation (strip "transformer." if present, add it back
#  for forward pass compatibility)
# ---------------------------------------------------------------------------

def normalise_weights(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Ensure all keys have the transformer. prefix and correct shapes."""
    out: dict[str, np.ndarray] = {}
    for k, v in raw.items():
        key = k if k.startswith("transformer.") else "transformer." + k
        out[key] = v
    return out


# ---------------------------------------------------------------------------
#  GPT-2 forward pass (numpy, CPU)
# ---------------------------------------------------------------------------

def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(-1, keepdims=True)
    var  = ((x - mean) ** 2).mean(-1, keepdims=True)
    return w * (x - mean) / np.sqrt(var + eps) + b


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def causal_attn(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    seq, d_k = q.shape[1], q.shape[2]
    scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(d_k)
    mask = np.triu(np.full((seq, seq), -1e10), k=1)
    return softmax(scores + mask) @ v


def _detect_arch(w: dict[str, np.ndarray]) -> tuple[int, int, int]:
    """Returns (n_layer, n_embd, n_head) from weight keys."""
    n_embd  = w["transformer.wte.weight"].shape[1]
    n_layer = sum(1 for k in w
                  if k.startswith("transformer.h.") and k.endswith("ln_1.weight"))
    # Guess n_head from QKV weight shape: c_attn.weight is [n_embd, 3*n_embd]
    # n_head is typically n_embd // 64 for GPT-2 variants
    n_head = n_embd // 64
    return n_layer, n_embd, n_head


def gpt2_forward(tokens: list[int], w: dict[str, np.ndarray],
                 n_head: int, n_layer: int) -> np.ndarray:
    seq = len(tokens)
    n_embd = w["transformer.wte.weight"].shape[1]
    d_head = n_embd // n_head

    x = w["transformer.wte.weight"][tokens] + w["transformer.wpe.weight"][:seq]

    for i in range(n_layer):
        p = f"transformer.h.{i}."
        a = layernorm(x, w[p + "ln_1.weight"], w[p + "ln_1.bias"])
        qkv = a @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(seq, n_head, d_head).transpose(1, 0, 2)
        k = k.reshape(seq, n_head, d_head).transpose(1, 0, 2)
        v = v.reshape(seq, n_head, d_head).transpose(1, 0, 2)
        attn = causal_attn(q, k, v).transpose(1, 0, 2).reshape(seq, n_embd)
        x = x + attn @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"]
        b = layernorm(x, w[p + "ln_2.weight"], w[p + "ln_2.bias"])
        h = b @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"]
        h = gelu(h)
        x = x + h @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"]

    x = layernorm(x, w["transformer.ln_f.weight"], w["transformer.ln_f.bias"])
    # lm_head tied to wte
    return x[-1] @ w["transformer.wte.weight"].T   # [vocab]


# ---------------------------------------------------------------------------
#  Sampling
# ---------------------------------------------------------------------------

def sample(logits: np.ndarray, temperature: float, top_k: int) -> int:
    logits = logits / max(temperature, 1e-6)
    if top_k > 1:
        kth = np.sort(logits)[-top_k]
        logits = np.where(logits < kth, -1e10, logits)
    probs = softmax(logits)
    return int(np.random.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
#  Fold classifier (mirrors Get-FoldPhase in fold_router_chat.ps1)
# ---------------------------------------------------------------------------

def get_fold(msg: str) -> str:
    l = msg.lower()
    if re.search(r'\b(ast|asx|parse|syntax|token|grammar|compile|ir)\b', l):     return "Sek"
    if re.search(r'\b(load|search|find|read|fetch|look|what|who|when|where|list|show|get)\b', l): return "Pop"
    if re.search(r'\b(build|create|write|code|implement|define|construct|make|generate|scaffold)\b', l): return "Wo"
    if re.search(r'\b(plan|predict|analyze|compare|evaluate|why|explain|design|think|reason)\b', l): return "Yax"
    if re.search(r'\b(execute|run|transform|convert|apply|calculate|compute|dispatch|call)\b', l): return "Sek"
    if re.search(r'\b(review|reflect|improve|optimize|refactor|summarize|check|validate|test)\b', l): return "Chen"
    if re.search(r'\b(save|export|done|finish|complete|store|archive|commit|push)\b', l):           return "Xul"
    if re.match(r'^\s*(hello|hi|hey|yo|sup|howdy|greetings)\b', l):                                return "Pop"
    return "Sek"


# ---------------------------------------------------------------------------
#  Tokenization (tiktoken GPT-2 BPE)
# ---------------------------------------------------------------------------

def get_tokenizer():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return enc
    except ImportError:
        print("[warn] tiktoken not installed. Install with: pip install tiktoken")
        print("       Falling back to character-level tokenization.")
        return None


def encode(enc, text: str) -> list[int]:
    if enc is None:
        return [ord(c) % 50257 for c in text[:512]]
    return enc.encode(text)


def decode(enc, tokens: list[int]) -> str:
    if enc is None:
        return "".join(chr(t % 128) for t in tokens)
    try:
        return enc.decode(tokens)
    except Exception:
        return " ".join(str(t) for t in tokens)


# ---------------------------------------------------------------------------
#  Model cache (for fold-route mode)
# ---------------------------------------------------------------------------

_model_cache: dict[str, tuple[dict, int, int, int]] = {}


def get_model(name: str) -> tuple[dict, int, int, int]:
    """Returns (weights, n_layer, n_embd, n_head). Cached."""
    if name not in _model_cache:
        path = XSHARD_PATHS.get(name)
        if path is None or not path.exists():
            # Fallback to gpt2large
            name = "gpt2large"
            path = XSHARD_PATHS["gpt2large"]
        raw = load_xshard(path)
        w = normalise_weights(raw)
        n_layer, n_embd, n_head = _detect_arch(w)
        print(f"  arch: {n_layer}L  embd={n_embd}  heads={n_head}", flush=True)
        _model_cache[name] = (w, n_layer, n_embd, n_head)
    return _model_cache[name]


# ---------------------------------------------------------------------------
#  Generation
# ---------------------------------------------------------------------------

def generate_tokens(prompt_ids: list[int], w: dict, n_layer: int, n_head: int,
                    max_new: int, temperature: float, top_k: int,
                    n_ctx: int = 1024) -> list[int]:
    tokens = list(prompt_ids)
    eos = 50256  # GPT-2 EOS / <|endoftext|>
    for _ in range(max_new):
        logits = gpt2_forward(tokens[-n_ctx:], w, n_head, n_layer)
        nxt = sample(logits, temperature, top_k)
        tokens.append(nxt)
        print(".", end="", flush=True)
        if nxt == eos:
            break
    print()
    return tokens


# ---------------------------------------------------------------------------
#  Chat REPL
# ---------------------------------------------------------------------------

def chat_loop(args):
    enc = get_tokenizer()

    # Determine which model(s) to load
    if args.fold_route:
        routing = dict(DEFAULT_ROUTING)
        if args.ast:
            routing.update(AST_OVERRIDES)
        # Load all needed models eagerly
        needed = set(routing.values())
        for model_name in needed:
            get_model(model_name)
        print(f"\n[fold-route] routing: {routing}")
    else:
        model_name = args.model
        get_model(model_name)
        routing = None

    print(f"\n[xshard-chat] max_new={args.max_new}  temp={args.temperature}  top_k={args.top_k}")
    print("Type a message. /fold to see last fold, /quit or ctrl-c to exit.\n")

    history: list[tuple[str, str]] = []
    last_fold = "Sek"

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]")
            break

        if not user:
            continue
        if user == "/quit":
            break
        if user == "/fold":
            print(f"  last fold: {last_fold}")
            continue

        # Fold detection
        fold = get_fold(user)
        last_fold = fold

        # Route to model
        if routing:
            active_model = routing.get(fold, "gpt2large")
        else:
            active_model = args.model

        w, n_layer, n_embd, n_head = get_model(active_model)

        # Build prompt: simple turn-based format
        prompt_parts = []
        for u, a in history[-3:]:  # last 3 turns context
            prompt_parts.append(f"User: {u}\nAssistant: {a}\n")
        prompt_parts.append(f"User: {user}\nAssistant:")
        prompt_text = "".join(prompt_parts)

        prompt_ids = encode(enc, prompt_text)
        if len(prompt_ids) > 900:
            prompt_ids = prompt_ids[-900:]

        print(f"[{fold} -> {active_model}] generating...", end="", flush=True)
        out_ids = generate_tokens(
            prompt_ids, w, n_layer, n_head,
            max_new=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        new_ids = out_ids[len(prompt_ids):]
        response = decode(enc, new_ids).strip()

        # Trim at "User:" if the model starts a new turn
        if "\nUser:" in response:
            response = response[:response.index("\nUser:")].strip()

        print(f"Assistant [{fold}/{active_model}]: {response}\n")
        history.append((user, response))


# ---------------------------------------------------------------------------
#  HTTP server (OpenAI-compatible /v1/chat/completions)
# ---------------------------------------------------------------------------

def serve_loop(args):
    """Minimal OpenAI-compatible server for fold_router_chat.ps1."""
    import http.server
    import urllib.parse

    # Determine routing
    if args.fold_route:
        routing = dict(DEFAULT_ROUTING)
        if args.ast:
            routing.update(AST_OVERRIDES)
        needed = set(routing.values())
        for m in needed:
            get_model(m)
    else:
        routing = None
        get_model(args.model)

    enc = get_tokenizer()

    cfg_args = args  # capture for handler

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass  # suppress default access log

        def do_GET(self):
            if self.path == "/v1/models":
                body = json.dumps({
                    "object": "list",
                    "data": [{"id": "xshard-fold", "object": "model"}]
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            try:
                req = json.loads(raw)
            except Exception:
                self.send_response(400)
                self.end_headers()
                return

            messages = req.get("messages", [])
            max_tokens = req.get("max_tokens", cfg_args.max_new)
            temperature = req.get("temperature", cfg_args.temperature)
            top_k = 40

            # Extract the last user message for fold detection
            user_text = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    user_text = m.get("content", "")
                    break

            fold = get_fold(user_text)
            if routing:
                active_model = routing.get(fold, "gpt2large")
            else:
                active_model = cfg_args.model

            w, n_layer, n_embd, n_head = get_model(active_model)
            print(f"  [{fold} -> {active_model}] {user_text[:60]!r}", flush=True)

            # Build prompt from message history
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    parts.append(content + "\n")
                elif role == "user":
                    parts.append(f"User: {content}\n")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}\n")
            parts.append("Assistant:")
            prompt_text = "".join(parts)

            prompt_ids = encode(enc, prompt_text)
            if len(prompt_ids) > 900:
                prompt_ids = prompt_ids[-900:]

            print(f"  generating (max_tokens={max_tokens})...", end="", flush=True)
            out_ids = generate_tokens(
                prompt_ids, w, n_layer, n_head,
                max_new=min(max_tokens, 256),
                temperature=temperature,
                top_k=top_k,
            )
            new_ids = out_ids[len(prompt_ids):]
            response = decode(enc, new_ids).strip()
            if "\nUser:" in response:
                response = response[:response.index("\nUser:")].strip()
            print(f" done ({len(new_ids)} tokens)", flush=True)

            body = json.dumps({
                "id": "xshard-1",
                "object": "chat.completion",
                "model": f"xshard-{active_model}-{fold}",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(new_ids),
                    "total_tokens": len(prompt_ids) + len(new_ids)
                }
            }).encode()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n[xshard-serve] listening on http://127.0.0.1:{args.port}")
    print(f"  model: {args.model if not routing else 'fold-routed'}  "
          f"fold-route: {args.fold_route}  ast: {args.ast}")
    print(f"  Connect fold_router_chat.ps1 -> Reconnect -> should pick up :9100")
    print("  Ctrl-C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[xshard-serve] stopped")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Chat with xshard fold models (numpy GPT-2 forward pass)")
    ap.add_argument("--model",  default="gpt2large",
                    choices=["gpt2large", "gemma", "qwen"],
                    help="Which xshard model to load (default: gpt2large)")
    ap.add_argument("--xshard", default=None,
                    help="Direct path to an xshard file (overrides --model)")
    ap.add_argument("--fold-route", action="store_true",
                    help="Enable fold-based routing across all three xshard models")
    ap.add_argument("--ast", action="store_true",
                    help="AST mode: Sek+Wo -> Qwen (requires --fold-route)")
    ap.add_argument("--max-new",     type=int,   default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k",       type=int,   default=40)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--serve",       action="store_true",
                    help="Run as OpenAI-compatible HTTP server (POST /v1/chat/completions)")
    ap.add_argument("--port",        type=int,   default=9100,
                    help="Port for --serve mode (default: 9100)")
    args = ap.parse_args()

    np.random.seed(args.seed)

    if args.xshard:
        XSHARD_PATHS["gpt2large"] = Path(args.xshard)
        args.model = "gpt2large"

    # Verify paths
    if args.fold_route:
        for name, path in XSHARD_PATHS.items():
            if not path.exists():
                print(f"[warn] {name} xshard not found: {path}")
    else:
        path = XSHARD_PATHS.get(args.model)
        if path and not path.exists():
            print(f"ERROR: xshard not found: {path}")
            sys.exit(1)

    if args.serve:
        serve_loop(args)
    else:
        chat_loop(args)


if __name__ == "__main__":
    main()
