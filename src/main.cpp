// =====================================================================
//  main.cpp  --  the `tinyllama` command line program
// =====================================================================
//  Examples:
//    tinyllama -m model.gguf --info
//    tinyllama -m model.gguf -p "Once upon a time" -n 64
//    tinyllama -m model.gguf -p "Hello" -n 32 --temp 0 -t 4
//    tinyllama -m model.gguf -p "Hello" --dump-logits out.bin
//
//  Design doc: Chapter 2 (running it) and Chapter 5.8.
// =====================================================================

#include "model.hpp"
#include "tokenizer.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

using namespace tinyml;

struct Args {
    std::string model, prompt = "Once upon a time", dump_logits;
    int n_predict = 128, n_ctx = 0, threads = 0;
    unsigned seed = 42;
    bool info = false, show_tokens = false, chat = false;
    std::string system = "You are a helpful assistant.";
    SamplingConfig sc;
};

static void usage() {
    std::printf(
        "usage: tinyllama -m MODEL.gguf [options]\n"
        "  -m, --model FILE        GGUF model file (llama architecture)\n"
        "  -p, --prompt TEXT       prompt (default: \"Once upon a time\")\n"
        "  -n, --n-predict N       tokens to generate (default 128)\n"
        "  -c, --ctx N             context size / KV-cache length (default: min(2048, trained))\n"
        "  -t, --threads N         worker threads (default: all cores)\n"
        "  -s, --seed N            RNG seed (default 42)\n"
        "      --temp F            temperature, 0 = greedy (default 0.8)\n"
        "      --top-k N           (default 40, 0 = off)\n"
        "      --top-p F           (default 0.95, 1 = off)\n"
        "      --repeat-penalty F  (default 1.1, 1 = off)\n"
        "      --chat              wrap the prompt in TinyLlama's chat template (Zephyr style)\n"
        "      --system TEXT       system message for --chat\n"
        "      --f32-act           keep activations in float32 (slower, exact; default quantizes to Q8_0)\n"
        "      --info              print GGUF metadata + tensor table and exit\n"
        "      --tokens            print the prompt's token ids\n"
        "      --dump-logits FILE  run the prompt, write last-position logits (float32) and exit\n");
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + k);
            return argv[++i];
        };
        if (k == "-m" || k == "--model") a.model = next();
        else if (k == "-p" || k == "--prompt") a.prompt = next();
        else if (k == "-n" || k == "--n-predict") a.n_predict = std::stoi(next());
        else if (k == "-c" || k == "--ctx") a.n_ctx = std::stoi(next());
        else if (k == "-t" || k == "--threads") a.threads = std::stoi(next());
        else if (k == "-s" || k == "--seed") a.seed = (unsigned)std::stoul(next());
        else if (k == "--temp") a.sc.temperature = std::stof(next());
        else if (k == "--top-k") a.sc.top_k = std::stoi(next());
        else if (k == "--top-p") a.sc.top_p = std::stof(next());
        else if (k == "--repeat-penalty") a.sc.repeat_penalty = std::stof(next());
        else if (k == "--info") a.info = true;
        else if (k == "--chat") a.chat = true;
        else if (k == "--f32-act") quantize_activations() = false;
        else if (k == "--system") a.system = next();
        else if (k == "--tokens") a.show_tokens = true;
        else if (k == "--dump-logits") a.dump_logits = next();
        else if (k == "-h" || k == "--help") { usage(); std::exit(0); }
        else throw std::runtime_error("unknown argument: " + k);
    }
    if (a.model.empty()) { usage(); throw std::runtime_error("no model given (-m)"); }
    return a;
}

static std::string value_to_string(const GGUFValue& v) {
    switch (v.type) {
        case GGUFType::STRING: return "\"" + (v.str.size() > 60 ? v.str.substr(0, 60) + "..." : v.str) + "\"";
        case GGUFType::FLOAT32: case GGUFType::FLOAT64: return std::to_string(v.f);
        case GGUFType::INT8: case GGUFType::INT16: case GGUFType::INT32: case GGUFType::INT64:
            return std::to_string(v.i);
        case GGUFType::BOOL: return v.u ? "true" : "false";
        case GGUFType::ARRAY: {
            std::string s = "[" + std::to_string(v.arr.size()) + " items: ";
            for (size_t i = 0; i < v.arr.size() && i < 4; ++i) s += value_to_string(v.arr[i]) + ", ";
            return s + "...]";
        }
        default: return std::to_string(v.u);
    }
}

static void print_info(const GGUFFile& g) {
    std::printf("GGUF version %u, alignment %llu, %zu metadata keys, %zu tensors, %.1f MB\n\n",
                g.version, (unsigned long long)g.alignment, g.kv.size(), g.tensors.size(),
                g.file_size() / 1e6);
    std::printf("---- metadata ----\n");
    for (const auto& [k, v] : g.kv) std::printf("  %-45s = %s\n", k.c_str(), value_to_string(v).c_str());
    std::printf("\n---- tensors (name, type, ne = [inner, outer]) ----\n");
    for (const auto& t : g.tensors) {
        std::string ne;
        for (size_t d = 0; d < t.ne.size(); ++d) ne += (d ? ", " : "") + std::to_string(t.ne[d]);
        std::printf("  %-32s %-5s [%s]  offset=%llu\n", t.name.c_str(), ggml_type_name(t.type),
                    ne.c_str(), (unsigned long long)t.offset);
    }
}

static int run(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    if (a.threads > 0) set_num_threads(a.threads);

    auto t0 = std::chrono::steady_clock::now();
    LlamaModel model = LlamaModel::load(a.model);
    Tokenizer tok = Tokenizer::from_gguf(model.gguf());
    auto t1 = std::chrono::steady_clock::now();

    if (a.info) { print_info(model.gguf()); return 0; }

    const LlamaConfig& c = model.cfg;
    std::fprintf(stderr,
        "model: dim=%d layers=%d heads=%d kv_heads=%d ffn=%d vocab=%d ctx_train=%d rope_base=%.0f\n"
        "loaded in %.2f s, threads=%d\n",
        c.dim, c.n_layer, c.n_head, c.n_kv_head, c.hidden_dim, c.n_vocab, c.n_ctx_train, c.rope_base,
        std::chrono::duration<double>(t1 - t0).count(), get_num_threads());

    // Chat models were fine-tuned on a fixed conversation format. TinyLlama
    // 1.1B Chat uses the "Zephyr" template; "</s>" becomes the EOS token id.
    std::string text = a.prompt;
    if (a.chat)
        text = "<|system|>\n" + a.system + "</s>\n<|user|>\n" + a.prompt + "</s>\n<|assistant|>\n";
    std::vector<int> prompt = tok.encode(text);
    if (a.show_tokens) {
        std::fprintf(stderr, "prompt tokens (%zu):", prompt.size());
        for (int id : prompt) std::fprintf(stderr, " %d('%s')", id, tok.piece(id).c_str());
        std::fprintf(stderr, "\n");
    }

    int n_ctx = a.n_ctx > 0 ? a.n_ctx : std::min(2048, c.n_ctx_train);
    if ((int)prompt.size() >= n_ctx) throw std::runtime_error("prompt longer than context (-c)");

    if (!a.dump_logits.empty()) {
        auto caches = model.make_caches(n_ctx);
        Tensor logits = model.forward(prompt, caches, 0);
        std::ofstream f(a.dump_logits, std::ios::binary);
        f.write(reinterpret_cast<const char*>(logits.data()), (std::streamsize)logits.numel() * 4);
        std::fprintf(stderr, "wrote %zu logits to %s (argmax = %d)\n", logits.numel(),
                     a.dump_logits.c_str(), argmax(logits.data(), (int)logits.numel()));
        return 0;
    }

    if (!a.chat) std::printf("%s", a.prompt.c_str());
    std::fflush(stdout);
    std::mt19937 rng(a.seed);
    int n_gen = 0;
    auto t2 = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point t_first;
    model.generate(prompt, a.n_predict, a.sc, rng, n_ctx, tok.eos_id, [&](int id) {
        if (n_gen++ == 0) t_first = std::chrono::steady_clock::now();
        std::printf("%s", tok.decode_piece(id).c_str());
        std::fflush(stdout);
        return true;
    });
    auto t3 = std::chrono::steady_clock::now();
    double prefill_s = std::chrono::duration<double>(t_first - t2).count();
    double decode_s = std::chrono::duration<double>(t3 - t_first).count();
    std::fprintf(stderr, "\n\nprompt: %zu tokens in %.2f s (%.1f tok/s) | generated: %d tokens in %.2f s (%.1f tok/s)\n",
                 prompt.size(), prefill_s, prompt.size() / prefill_s, n_gen, decode_s,
                 n_gen > 1 ? (n_gen - 1) / decode_s : 0.0);
    return 0;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);   // so "▁" and emoji print correctly in cmd/PowerShell
#endif
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        // Tip: in gdb, `catch throw` stops at the ORIGINAL throw site,
        // long before we land here. See design doc Chapter 9.
        std::fprintf(stderr, "\nerror: %s\n", e.what());
        return 1;
    }
}
