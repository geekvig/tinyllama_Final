# tinyllama — my own llama.cpp, from scratch

A small, readable, **multithreaded C++17** inference engine for Llama-architecture
models stored in **GGUF** files (the llama.cpp format). Header-only, no dependencies.

The companion PDF (*TinyLlama Design Doc & Tutorial*) explains every line, the GGUF
format, the math, the threading, and has 10 deliberate-bug debugging labs.

## Quick start (Windows, MSYS2 UCRT64 g++)

```bat
:: one-time, in the "MSYS2 UCRT64" shell:
::   pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-gdb
build.bat
build\run_tests.exe
build\tinyllama.exe -m tests\data\tiny_q.gguf -p "once upon a time" -n 30
```

Linux / WSL / MSYS2 shell: `./build.sh` then `./build/run_tests.exe`.

## Run a real model

Download a Llama-architecture GGUF with the SentencePiece ("llama") tokenizer, e.g.
TinyLlama-1.1B-Chat-v1.0 (`Q8_0`, `Q4_0` or `Q4_K_M`) from Hugging Face, then:

```bat
build\tinyllama.exe -m C:\models\tinyllama-1.1b-chat-v1.0.Q8_0.gguf --chat -p "Why is the sky blue?" -n 200
build\tinyllama.exe -m C:\models\tinyllama-1.1b-chat-v1.0.Q8_0.gguf --info
```

Supported tensor types: F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K.
Supported: `general.architecture = llama`, `tokenizer.ggml.model = llama`
(Llama 1/2, TinyLlama, many fine-tunes). Not yet: Llama 3 (`gpt2` BPE tokenizer), Q2_K/Q3_K.

## Layout

| path | what |
|---|---|
| `include/gguf.hpp` | GGUF parser (header, metadata, tensor table, alignment) |
| `include/quant.hpp`, `fp16.hpp` | ggml-compatible Q8_0/Q4_0/K-quant blocks, fp16, SIMD dot products |
| `include/weights.hpp` | `Weight` (F32/quantized matrix) + multithreaded `linear` |
| `include/ops.hpp` | RMSNorm, RoPE, causal GQA attention, SwiGLU, embed |
| `include/kv_cache.hpp` | per-layer K/V memory |
| `include/model.hpp` | config, layer weights, forward pass, generation loop |
| `include/tokenizer.hpp` | SentencePiece tokenizer read from GGUF |
| `include/sampling.hpp` | temperature / top-k / top-p / repetition penalty |
| `include/thread_pool.hpp` | persistent thread pool + `parallel_for` |
| `src/main.cpp` | the `tinyllama` CLI |
| `tests/run_tests.cpp` | 19 tests (no framework needed) |
| `tests/data/` | tiny GGUF models + reference logits (NumPy and llama.cpp) |
| `tools/make_test_gguf.py` | regenerates `tests/data` (`pip install numpy gguf`) |
| `tools/reference.py` | the whole forward pass in NumPy (answer key) |
| `tools/compare_llamacpp.py` | compare against real llama.cpp (`pip install llama-cpp-python`) |
| `tools/bug_lab.py` | creates `labs/labNN_*` copies with deliberate bugs |
| `.vscode/` | build tasks (Ctrl+Shift+B) and gdb debug configs (F5) |
