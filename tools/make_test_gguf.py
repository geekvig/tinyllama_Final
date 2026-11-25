#!/usr/bin/env python3
"""
make_test_gguf.py -- build tiny, random-weight Llama models as REAL .gguf files.

Why: a real model is 600 MB - 4 GB. To test the C++ engine quickly (and to
practise debugging) we want a model that loads in milliseconds but uses
every feature the engine supports:

  * architecture "llama", SentencePiece tokenizer stored in GGUF metadata
  * GQA (4 query heads, 2 kv heads)
  * tensor types F32, F16, Q8_0, Q4_0 (in tiny_q.gguf)
  * K-quants Q4_K, Q5_K, Q6_K (in tiny_k.gguf, dim 256 because a K-quant
    row must be a multiple of 256)
  * tied embeddings (tiny_f32.gguf has NO output.weight)

It also writes the reference answers produced by tools/reference.py
(pure NumPy) so tests/run_tests.cpp can compare logits:

  tests/data/tiny_f32.gguf      tests/data/tiny_f32.ref.bin
  tests/data/tiny_q.gguf        tests/data/tiny_q.ref.bin
  tests/data/tiny_k.gguf        tests/data/tiny_k.ref.bin
  tests/data/prompt_ids.txt     token ids of the test prompt

Usage:   pip install numpy gguf
         python tools/make_test_gguf.py            (writes into tests/data)
"""
import os
import sys
import numpy as np
import gguf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "data")

DIM, N_LAYER, N_HEAD, N_KV_HEAD, HIDDEN, CTX = 64, 2, 4, 2, 128, 128
PROMPT = "once upon a time there was a tiny llama"


def build_vocab():
    """A miniature SentencePiece vocabulary: specials, 256 byte tokens, chars, merges."""
    tokens, scores, types = [], [], []

    def add(t, s, ty):
        tokens.append(t); scores.append(s); types.append(ty)

    add("<unk>", 0.0, 2)           # UNKNOWN
    add("<s>", 0.0, 3)             # CONTROL (BOS)
    add("</s>", 0.0, 3)            # CONTROL (EOS)
    for b in range(256):           # BYTE fallback tokens
        add("<0x%02X>" % b, 0.0, 6)
    sp = "▁"                  # the SentencePiece space marker "▁"
    singles = [sp] + list("abcdefghijklmnopqrstuvwxyz")
    for ch in singles:
        add(ch, -10.0, 1)
    # merges: higher score = merged earlier
    merges = ["th", "he", sp + "t", sp + "a", "on", "ce", "up", "ti", "me", "er", "re",
              "as", sp + "th", sp + "the", sp + "on", sp + "once", sp + "up", sp + "upon",
              sp + "ti", sp + "time", sp + "w", sp + "was", "ll", "am", sp + "l", sp + "ll",
              sp + "lla", sp + "llama", "in", sp + "tin", sp + "tiny", "ere", sp + "there"]
    for i, m in enumerate(merges):
        add(m, -1.0 - 0.1 * i, 1)
    while len(tokens) < 352:       # pad to a round size
        add("<unused%d>" % len(tokens), -100.0, 5)
    return tokens, scores, types


def random_weights(n_vocab, seed, DIM=DIM, HIDDEN=HIDDEN):
    rng = np.random.default_rng(seed)
    r = lambda *shape: (rng.standard_normal(shape) * 0.08).astype(np.float32)
    w = {"token_embd.weight": r(n_vocab, DIM),
         "output_norm.weight": (1.0 + 0.1 * rng.standard_normal(DIM)).astype(np.float32),
         "output.weight": r(n_vocab, DIM)}
    kv = DIM // N_HEAD * N_KV_HEAD
    for l in range(N_LAYER):
        p = f"blk.{l}."
        w[p + "attn_norm.weight"] = (1.0 + 0.1 * rng.standard_normal(DIM)).astype(np.float32)
        w[p + "ffn_norm.weight"] = (1.0 + 0.1 * rng.standard_normal(DIM)).astype(np.float32)
        w[p + "attn_q.weight"] = r(DIM, DIM)        # numpy shape [out, in] -> GGUF ne [in, out]
        w[p + "attn_k.weight"] = r(kv, DIM)
        w[p + "attn_v.weight"] = r(kv, DIM)
        w[p + "attn_output.weight"] = r(DIM, DIM)
        w[p + "ffn_gate.weight"] = r(HIDDEN, DIM)
        w[p + "ffn_up.weight"] = r(HIDDEN, DIM)
        w[p + "ffn_down.weight"] = r(DIM, HIDDEN)
    return w


# which GGML type each tensor gets in the "quantized" test models
def qtype_for(name, kquants=False):
    Q = gguf.GGMLQuantizationType
    if name.endswith("norm.weight"):
        return Q.F32
    if kquants:
        if "attn_q" in name or "ffn_gate" in name or name == "token_embd.weight":
            return Q.Q4_K
        if "attn_k" in name or "attn_v" in name or "ffn_up" in name:
            return Q.Q5_K
        return Q.Q6_K                      # attn_output, ffn_down, output
    if name.endswith("norm.weight"):
        return Q.F32                       # 1-D vectors always stay F32
    if "attn_q" in name or "attn_output" in name or name == "token_embd.weight":
        return Q.Q8_0
    if "attn_k" in name or "attn_v" in name or name == "output.weight":
        return Q.Q4_0
    return Q.F16                           # ffn_* -> F16


# gguf-py cannot QUANTIZE K-quants (only dequantize), so for those we write
# random blocks with sensible scale fields and let gguf-py tell us what
# float values they decode to. That is all a test needs.
KQ_LAYOUT = {  # type: (bytes per 256-element block, [(byte offset of fp16 scale, value)])
    "Q4_K": (144, [(0, 0.004), (2, 0.004)]),
    "Q5_K": (176, [(0, 0.002), (2, 0.002)]),
    "Q6_K": (210, [(208, 0.0006)]),
}


def random_kquant(shape, t, rng):
    nbytes, scales = KQ_LAYOUT[t.name]
    rows, cols = shape
    nb = cols // 256
    raw = rng.integers(0, 256, size=(rows, nb, nbytes), dtype=np.uint8)
    for off, val in scales:
        h = np.array([val], dtype=np.float16).view(np.uint8)
        raw[:, :, off] = h[0]
        raw[:, :, off + 1] = h[1]
    raw = raw.reshape(rows, nb * nbytes)
    deq = gguf.quants.dequantize(raw, t).reshape(shape).astype(np.float32)
    return raw, deq


def write_gguf(path, weights, tokens, scores, types, quantized, tied, kquants=False,
               DIM=DIM, HIDDEN=HIDDEN):
    Q = gguf.GGMLQuantizationType
    rng = np.random.default_rng(99)
    wr = gguf.GGUFWriter(path, "llama")
    wr.add_name("tinyllama-test")
    wr.add_context_length(CTX)
    wr.add_embedding_length(DIM)
    wr.add_block_count(N_LAYER)
    wr.add_feed_forward_length(HIDDEN)
    wr.add_head_count(N_HEAD)
    wr.add_head_count_kv(N_KV_HEAD)
    wr.add_rope_dimension_count(DIM // N_HEAD)
    wr.add_rope_freq_base(10000.0)
    wr.add_layer_norm_rms_eps(1e-5)
    wr.add_file_type(7 if quantized else 0)
    wr.add_tokenizer_model("llama")
    wr.add_token_list(tokens)
    wr.add_token_scores(scores)
    wr.add_token_types(types)
    wr.add_bos_token_id(1)
    wr.add_eos_token_id(2)
    wr.add_unk_token_id(0)
    wr.add_add_bos_token(True)
    wr.add_add_space_prefix(True)

    effective = {}  # what the reader will actually see (after quantization)
    for name, arr in weights.items():
        if tied and name == "output.weight":
            continue
        t = qtype_for(name, kquants) if quantized else Q.F32
        if t in (Q.Q4_K, Q.Q5_K, Q.Q6_K):
            raw, deq = random_kquant(arr.shape, t, rng)
            wr.add_tensor(name, raw, raw_dtype=t)
            effective[name] = deq
        elif t == Q.F32:
            wr.add_tensor(name, arr)
            effective[name] = arr
        elif t == Q.F16:
            a16 = arr.astype(np.float16)
            wr.add_tensor(name, a16)
            effective[name] = a16.astype(np.float32)
        else:
            qd = gguf.quants.quantize(arr, t)
            wr.add_tensor(name, qd, raw_dtype=t)
            effective[name] = gguf.quants.dequantize(qd, t).reshape(arr.shape).astype(np.float32)
    if tied:
        effective["output.weight"] = effective["token_embd.weight"]
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_tensors_to_file()
    wr.close()
    return effective


def main():
    os.makedirs(OUT, exist_ok=True)
    tokens, scores, types = build_vocab()
    w = random_weights(len(tokens), seed=1234)
    cfg = dict(dim=DIM, n_layer=N_LAYER, n_head=N_HEAD, n_kv_head=N_KV_HEAD,
               rope_base=10000.0, eps=1e-5)

    ids = reference.spm_encode(PROMPT, tokens, scores)
    with open(os.path.join(OUT, "prompt_ids.txt"), "w") as f:
        f.write(" ".join(map(str, ids)) + "\n")
    print("prompt ids:", ids)

    for fname, quantized, tied, kq in [("tiny_f32", False, True, False),
                                       ("tiny_q", True, False, False),
                                       ("tiny_k", True, False, True)]:
        path = os.path.join(OUT, fname + ".gguf")
        dim, hidden = (256, 512) if kq else (DIM, HIDDEN)
        ww = random_weights(len(tokens), 1234, dim, hidden) if kq else w
        eff = write_gguf(path, ww, tokens, scores, types, quantized, tied, kq, dim, hidden)
        c = dict(cfg, dim=dim)
        logits = reference.forward_all(eff, ids, c)         # [T, vocab]
        logits.astype(np.float32).tofile(os.path.join(OUT, fname + ".ref.bin"))
        print(f"wrote {path} ({os.path.getsize(path)} bytes), ref logits shape {logits.shape}, "
              f"argmax(last) = {int(np.argmax(logits[-1]))}")


if __name__ == "__main__":
    main()
