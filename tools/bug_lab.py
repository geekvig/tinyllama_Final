#!/usr/bin/env python3
"""
bug_lab.py -- inject a deliberate bug into a COPY of the project, so you
can practise debugging without ever breaking the working code.

    python tools/bug_lab.py --list          show the labs
    python tools/bug_lab.py 4               create labs/lab04_kv_overflow/ with bug 4
    python tools/bug_lab.py all             create every lab

Then open the lab folder in VS Code (File > Open Folder) or build it from a
terminal exactly like the main project. The design doc (Chapter 10) walks
through every lab: symptom, gdb commands, VS Code GUI steps, the fix.
"""
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Each lab: (folder name, title, [(file, old_text, new_text)], how to trigger it)
LABS = {
    1: ("lab01_typo_exception", "A typo throws an exception deep inside the loader",
        [("include/model.hpp",
          'L.wq     = Weight::from_gguf(g.tensor(p + "attn_q.weight"));',
          'L.wq     = Weight::from_gguf(g.tensor(p + "attn_q.weigth"));')],
        "tinyllama -m tests/data/tiny_q.gguf -p hello"),
    2: ("lab02_gguf_strlen", "GGUF string length read as 32-bit instead of 64-bit",
        [("include/gguf.hpp",
          "uint64_t len = rd<uint64_t>();       // GGUF v2+: 64-bit length (Bug Lab 2!)",
          "uint64_t len = rd<uint32_t>();       // GGUF v2+: 64-bit length (Bug Lab 2!)")],
        "tinyllama -m tests/data/tiny_q.gguf --info"),
    3: ("lab03_fp16_scale", "Q8_0 scale used as raw bits instead of converting fp16",
        [("include/quant.hpp",
          "inline void dequantize_row_q8_0(const BlockQ8_0* x, float* y, int n) {\n"
          "    for (int b = 0; b < n / QK; ++b) {\n"
          "        const float d = fp16_to_fp32(x[b].d);",
          "inline void dequantize_row_q8_0(const BlockQ8_0* x, float* y, int n) {\n"
          "    for (int b = 0; b < n / QK; ++b) {\n"
          "        const float d = (float)x[b].d;")],
        "run_tests        (then: run_tests model_)"),
    4: ("lab04_kv_overflow", "Heap buffer overflow: KV-cache guards removed",
        [("include/kv_cache.hpp",
          "        if (len_ + T > max_len_)   // Bug Lab 4 deletes this line\n"
          "            throw std::runtime_error(\"KVCache: context full (\" + std::to_string(max_len_) +\n"
          "                                     \" tokens). Use a larger -c / --ctx.\");\n",
          ""),
         ("include/model.hpp",
          "            if ((int)ids.size() >= n_ctx) break;          // context full\n", "")],
        "tinyllama -m tests/data/tiny_q.gguf -p \"once upon a time\" -c 8 -n 40"),
    5: ("lab05_null_deref", "Segmentation fault: dereferencing a null tensor pointer",
        [("include/model.hpp",
          'm.output_ = Weight::from_gguf(g.tensor("token_embd.weight")); // tied embeddings',
          'm.output_ = Weight::from_gguf(*g.find_tensor("output.weight")); // "simplified"')],
        "tinyllama -m tests/data/tiny_f32.gguf -p hello"),
    6: ("lab06_use_after_free", "Use-after-free: a shallow copy of a self-pointing object",
        [("include/weights.hpp",
          "    Weight(const Weight&) = delete;            // see comment above\n"
          "    Weight& operator=(const Weight&) = delete;",
          "    Weight(const Weight&) = default;           // \"I needed copies\"\n"
          "    Weight& operator=(const Weight&) = default;"),
         ("include/model.hpp", "            LayerWeights& L = m.layers_[l];",
          "            LayerWeights L;                       // build locally, copy in at the end"),
         ("include/model.hpp",
          '            L.w_down = Weight::from_gguf(g.tensor(p + "ffn_down.weight"));',
          '            L.w_down = Weight::from_gguf(g.tensor(p + "ffn_down.weight"));\n'
          '            m.layers_[l] = L;                     // COPY')],
        "tinyllama -m tests/data/tiny_q.gguf -p \"once upon a time\" -n 10 --temp 0   (note: run_tests may still PASS!)"),
    7: ("lab07_rope_freq", "RoPE frequencies computed over the whole row, not per head",
        [("include/ops.hpp",
          "inv_freq[i] = std::pow(freq_base, -2.0f * i / (float)n_rot);",
          "inv_freq[i] = std::pow(freq_base, -2.0f * i / (float)(n_heads * head_dim));")],
        "run_tests        (note which tests still PASS!)"),
    8: ("lab08_causal_mask", "Off-by-one in the causal mask: a token cannot see itself",
        [("include/ops.hpp",
          "                for (int j = 0; j <= p; ++j) {\n                    float s",
          "                for (int j = 0; j < p; ++j) {\n                    float s"),
         ("include/ops.hpp",
          "for (int j = 0; j <= p; ++j) { scores[j] = std::exp",
          "for (int j = 0; j < p; ++j) { scores[j] = std::exp"),
         ("include/ops.hpp",
          "                for (int j = 0; j <= p; ++j) {\n                    const float w",
          "                for (int j = 0; j < p; ++j) {\n                    const float w")],
        "run_tests attention"),
    9: ("lab09_data_race", "Data race: all threads share one scratch buffer",
        [("include/ops.hpp",
          "    parallel_for(n_head, [&](int64_t h0, int64_t h1) {\n"
          "        std::vector<float> scores(n_ctx_used);               // per-chunk scratch\n",
          "    std::vector<float> scores(n_ctx_used);                   // \"allocate once, it's faster\"\n"
          "    parallel_for(n_head, [&](int64_t h0, int64_t h1) {\n")],
        "run_tests generation   (try -t 1 vs default; run several times)"),
    10: ("lab10_deadlock", "Deadlock: a worker never signals that the work is done",
         [("include/thread_pool.hpp",
           "if (++done_ == n_tasks_) cv_done_.notify_all();",
           "++done_;   // \"the caller will notice\"")],
         "run_tests  (hangs sometimes -- run it a few times)"),
}

COPY = ["include", "src", "tests", "tools", "CMakeLists.txt", ".vscode", "build.sh", "build.bat"]


def make_lab(n):
    folder, title, patches, how = LABS[n]
    dst = os.path.join(ROOT, "labs", folder)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for item in COPY:
        src = os.path.join(ROOT, item)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(dst, item))
        elif os.path.exists(src):
            shutil.copy2(src, dst)
    for rel, old, new in patches:
        p = os.path.join(dst, rel)
        with open(p, encoding="utf-8") as f:
            text = f.read()
        if text.count(old) != 1:
            sys.exit(f"lab {n}: patch target not found exactly once in {rel} -- has the code changed?")
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(text.replace(old, new))
    print(f"Lab {n}: {title}\n  created {dst}\n  trigger: {how}\n")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "--list"):
        print(__doc__)
        for n, (folder, title, _, how) in LABS.items():
            print(f"  {n:2d}  {title}\n      trigger: {how}")
        return
    which = list(LABS) if sys.argv[1] == "all" else [int(a) for a in sys.argv[1:]]
    for n in which:
        make_lab(n)


if __name__ == "__main__":
    main()
