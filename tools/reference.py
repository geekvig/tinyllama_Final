#!/usr/bin/env python3
"""
reference.py -- the whole Llama forward pass in ~60 lines of NumPy.

This is the "answer key" for the C++ engine. It is deliberately written
the slow, obvious way (no KV cache, full matrices) so that it is easy to
convince yourself it is right. The C++ code must produce the same logits
(up to float rounding, ~1e-4).

Also contains the SentencePiece encoder in Python, used to produce the
expected token ids.

Use it on any GGUF (needs `pip install numpy gguf`):
    python tools/reference.py model.gguf "Once upon a time"
(only practical for small models -- it dequantizes everything to float32)
"""
import sys
import numpy as np


def rmsnorm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope(x, n_heads, head_dim, base):
    # x: [T, n_heads*head_dim]; rotate pairs (2i, 2i+1) inside each head ("NORM" rope)
    T = x.shape[0]
    x = x.reshape(T, n_heads, head_dim // 2, 2)
    pos = np.arange(T, dtype=np.float64)[:, None]
    inv = base ** (-2.0 * np.arange(head_dim // 2) / head_dim)
    ang = pos * inv[None, :]                       # [T, head_dim/2]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x0, x1 = x[..., 0], x[..., 1]
    out = np.stack([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1)
    return out.reshape(T, n_heads * head_dim)


def forward_all(w, ids, cfg):
    """Return logits for EVERY position: [T, vocab]."""
    dim, nh, nkv = cfg["dim"], cfg["n_head"], cfg["n_kv_head"]
    hd = dim // nh
    T = len(ids)
    x = w["token_embd.weight"][ids].astype(np.float64)
    mask = np.triu(np.full((T, T), -np.inf), k=1)   # causal: no looking ahead
    for l in range(cfg["n_layer"]):
        p = f"blk.{l}."
        h = rmsnorm(x, w[p + "attn_norm.weight"], cfg["eps"])
        q = h @ w[p + "attn_q.weight"].T            # weights are [out, in]
        k = h @ w[p + "attn_k.weight"].T
        v = h @ w[p + "attn_v.weight"].T
        q = rope(q, nh, hd, cfg["rope_base"])
        k = rope(k, nkv, hd, cfg["rope_base"])
        out = np.zeros_like(q)
        for head in range(nh):
            kvh = head // (nh // nkv)                # grouped-query attention
            qh = q[:, head * hd:(head + 1) * hd]
            kh = k[:, kvh * hd:(kvh + 1) * hd]
            vh = v[:, kvh * hd:(kvh + 1) * hd]
            s = qh @ kh.T / np.sqrt(hd) + mask
            s = np.exp(s - s.max(axis=1, keepdims=True))
            s /= s.sum(axis=1, keepdims=True)
            out[:, head * hd:(head + 1) * hd] = s @ vh
        x = x + out @ w[p + "attn_output.weight"].T
        h = rmsnorm(x, w[p + "ffn_norm.weight"], cfg["eps"])
        g = h @ w[p + "ffn_gate.weight"].T
        u = h @ w[p + "ffn_up.weight"].T
        x = x + ((g / (1 + np.exp(-g))) * u) @ w[p + "ffn_down.weight"].T
    x = rmsnorm(x, w["output_norm.weight"], cfg["eps"])
    return x @ w["output.weight"].T


def spm_encode(text, tokens, scores, add_bos=True):
    """SentencePiece-style encode: identical algorithm to include/tokenizer.hpp."""
    index = {}
    for i, t in enumerate(tokens):
        index.setdefault(t, i)
    s = (" " + text).replace(" ", "▁")
    sym = list(s)
    while len(sym) > 1:
        best, best_i = -1e30, -1
        for i in range(len(sym) - 1):
            j = index.get(sym[i] + sym[i + 1])
            if j is not None and scores[j] > best:
                best, best_i = scores[j], i
        if best_i < 0:
            break
        sym[best_i:best_i + 2] = [sym[best_i] + sym[best_i + 1]]
    ids = [1] if add_bos else []
    for p in sym:
        if p in index:
            ids.append(index[p])
        else:
            ids.extend(index.get("<0x%02X>" % b, 0) for b in p.encode("utf-8"))
    return ids


def load_gguf_as_float(path):
    import gguf
    r = gguf.GGUFReader(path)
    w = {}
    for t in r.tensors:
        shape = [int(d) for d in reversed(t.shape)]      # GGUF ne -> numpy [out, in]
        w[t.name] = gguf.quants.dequantize(t.data, t.tensor_type).reshape(shape).astype(np.float32)
    def kv(key, default=None):
        f = r.fields.get(key)
        if f is None:
            return default
        return f.contents()
    arch = kv("general.architecture")
    cfg = dict(dim=kv(f"{arch}.embedding_length"), n_layer=kv(f"{arch}.block_count"),
               n_head=kv(f"{arch}.attention.head_count"),
               n_kv_head=kv(f"{arch}.attention.head_count_kv", kv(f"{arch}.attention.head_count")),
               rope_base=kv(f"{arch}.rope.freq_base", 10000.0),
               eps=kv(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5))
    if "output.weight" not in w:
        w["output.weight"] = w["token_embd.weight"]
    return w, cfg, kv("tokenizer.ggml.tokens"), kv("tokenizer.ggml.scores")


if __name__ == "__main__":
    w, cfg, tokens, scores = load_gguf_as_float(sys.argv[1])
    ids = spm_encode(sys.argv[2] if len(sys.argv) > 2 else "Once upon a time", tokens, scores)
    logits = forward_all(w, ids, cfg)
    print("ids:", ids)
    print("argmax of last position:", int(np.argmax(logits[-1])), repr(tokens[int(np.argmax(logits[-1]))]))
    print("first 8 logits:", np.round(logits[-1][:8], 5))
