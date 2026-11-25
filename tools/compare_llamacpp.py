#!/usr/bin/env python3
"""
compare_llamacpp.py -- check tinyllama against the REAL llama.cpp.

Needs:  pip install llama-cpp-python numpy
(on Windows this compiles llama.cpp; it can take a few minutes)

    python tools/compare_llamacpp.py MODEL.gguf "Once upon a time" build/tinyllama.exe

It (1) tokenizes the prompt with llama.cpp and with tinyllama (--tokens),
(2) asks both for the logits of the last prompt position, and
(3) prints the largest difference and both top-5 predictions.
Expect: identical tokens; top-1 usually identical; differences of a few
percent of the logit range for quantized models (llama.cpp and tinyllama
round activations slightly differently).
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from llama_cpp import Llama


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    model, prompt, exe = sys.argv[1], sys.argv[2], sys.argv[3]

    llm = Llama(model_path=model, n_ctx=512, logits_all=True, verbose=False)
    toks = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
    llm.eval(toks)
    ref = np.array(llm.scores[len(toks) - 1], dtype=np.float32)

    out = os.path.join(tempfile.gettempdir(), "tinyllama_logits.bin")
    r = subprocess.run([exe, "-m", model, "-p", prompt, "--tokens", "--dump-logits", out],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stderr)
    ours = np.fromfile(out, dtype=np.float32)

    print("llama.cpp tokens:", toks)
    d = np.abs(ours - ref)
    print(f"max |tinyllama - llama.cpp| = {d.max():.4f}   (max |logit| = {np.abs(ref).max():.2f})")
    for name, v in (("llama.cpp", ref), ("tinyllama", ours)):
        top = np.argsort(-v)[:5]
        print(f"{name:10s} top-5:", [(int(i), llm.detokenize([int(i)]).decode('utf-8', 'replace')) for i in top])


if __name__ == "__main__":
    main()
