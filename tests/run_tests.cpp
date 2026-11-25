// =====================================================================
//  run_tests.cpp  --  a dependency-free test runner for tinyllama
// =====================================================================
//  No GoogleTest needed (nothing to install on Windows): a TEST() macro
//  registers a function, CHECK()/CHECK_NEAR() record failures and keep
//  going, main() runs everything and returns non-zero on any failure.
//
//    run_tests                    run all tests (data in tests/data)
//    run_tests rope               run tests whose name contains "rope"
//    run_tests --data DIR ...     use another data directory
//
//  Design doc: Chapter 8 ("Testing: how do we know it is right?").
// =====================================================================

#include "model.hpp"
#include "tokenizer.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace tinyml;

// ---------------- the tiny framework ----------------
struct TestCase { const char* name; std::function<void()> fn; };
static std::vector<TestCase>& registry() { static std::vector<TestCase> r; return r; }
struct Registrar { Registrar(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); } };
#define TEST(name) static void name(); static Registrar reg_##name(#name, name); static void name()

static int g_failures = 0;
#define CHECK(cond) do { if (!(cond)) { ++g_failures; \
    std::printf("    CHECK failed: %s  (%s:%d)\n", #cond, __FILE__, __LINE__); } } while (0)
#define CHECK_NEAR(a, b, tol) do { double _a = (a), _b = (b); if (!(std::fabs(_a - _b) <= (tol))) { ++g_failures; \
    std::printf("    CHECK_NEAR failed: %s = %.6g vs %s = %.6g (tol %g)  (%s:%d)\n", #a, _a, #b, _b, (double)(tol), __FILE__, __LINE__); } } while (0)
#define CHECK_THROWS(stmt) do { bool _t = false; try { stmt; } catch (const std::exception&) { _t = true; } \
    if (!_t) { ++g_failures; std::printf("    expected exception: %s  (%s:%d)\n", #stmt, __FILE__, __LINE__); } } while (0)

static std::string g_data = "tests/data";
static std::string data(const std::string& f) { return g_data + "/" + f; }

static std::vector<float> read_floats(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path + " (run from the project root, or pass --data DIR)");
    std::vector<float> v((size_t)f.tellg() / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)v.size() * 4);
    return v;
}
static std::vector<int> prompt_ids() {
    std::ifstream f(data("prompt_ids.txt"));
    if (!f) throw std::runtime_error("cannot open prompt_ids.txt");
    std::vector<int> ids; int x;
    while (f >> x) ids.push_back(x);
    return ids;
}
static Tensor random_tensor(std::vector<int> shape, unsigned seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, scale);
    Tensor t(std::move(shape));
    for (size_t i = 0; i < t.numel(); ++i) t.at_flat(i) = d(rng);
    return t;
}
static float max_abs_diff(const float* a, const float* b, size_t n) {
    float m = 0; for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i])); return m;
}

// =====================================================================
//                             THE TESTS
// =====================================================================

TEST(fp16_roundtrip) {
    for (float f : {0.0f, 1.0f, -2.5f, 0.333251953125f, 65504.0f, 6.1035156e-05f, 5.9604645e-08f})
        CHECK(fp16_to_fp32(fp32_to_fp16(f)) == f);       // exactly representable values
    CHECK(fp16_to_fp32(0x3C00) == 1.0f);                  // known bit patterns
    CHECK(fp16_to_fp32(0xC000) == -2.0f);
    CHECK(std::isinf(fp16_to_fp32(0x7C00)));
    // the fast path (hardware or table) must agree with the slow path on ALL 65536 patterns
    int mismatches = 0;
    for (uint32_t i = 0; i < 65536; ++i) {
        float a = fp16_to_fp32((uint16_t)i), b = fp16_to_fp32_slow((uint16_t)i);
        if (!(a == b || (std::isnan(a) && std::isnan(b)))) ++mismatches;
    }
    CHECK(mismatches == 0);
}

TEST(quant_q8_0_and_q4_0_roundtrip) {
    Tensor x = random_tensor({1, 256}, 1);
    std::vector<BlockQ8_0> q8(256 / QK);
    std::vector<BlockQ4_0> q4(256 / QK);
    quantize_row_q8_0(x.data(), q8.data(), 256);
    quantize_row_q4_0(x.data(), q4.data(), 256);
    std::vector<float> y8(256), y4(256);
    dequantize_row_q8_0(q8.data(), y8.data(), 256);
    dequantize_row_q4_0(q4.data(), y4.data(), 256);
    float e8 = max_abs_diff(x.data(), y8.data(), 256), e4 = max_abs_diff(x.data(), y4.data(), 256);
    std::printf("    max error: Q8_0 %.4f   Q4_0 %.4f\n", e8, e4);
    CHECK(e8 < 0.03f);
    CHECK(e4 < 0.5f);
    CHECK(e4 > e8);
    // fused dot == dot with dequantized weights
    Tensor a = random_tensor({1, 256}, 2);
    CHECK_NEAR(dot_q8_0(a.data(), q8.data(), 256), dot_f32(a.data(), y8.data(), 256), 1e-3);
    CHECK_NEAR(dot_q4_0(a.data(), q4.data(), 256), dot_f32(a.data(), y4.data(), 256), 1e-3);
}

TEST(parallel_for_covers_every_index_once) {
    for (int threads : {1, 2, 3, 8, 64}) {
        ScopedThreads st(threads);
        std::vector<std::atomic<int>> hits(1000);
        parallel_for(1000, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; ++i) hits[i]++; });
        bool ok = true;
        for (auto& h : hits) ok &= (h.load() == 1);
        CHECK(ok);
    }
}

TEST(parallel_for_propagates_exceptions) {
    ScopedThreads st(4);
    CHECK_THROWS(parallel_for(100, [&](int64_t lo, int64_t) { if (lo > 0) throw std::runtime_error("boom"); }));
    // and the pool still works afterwards
    std::atomic<int> sum{0};
    parallel_for(100, [&](int64_t lo, int64_t hi) { sum += (int)(hi - lo); });
    CHECK(sum.load() == 100);
}

TEST(linear_threads_bit_identical) {
    // Build a quantized weight by hand and compare 1 thread vs many.
    Tensor w = random_tensor({96, 64}, 3);
    Weight W; W.type = WType::F32; W.rows = 96; W.cols = 64; W.name = "w";
    W.owned.assign(w.data(), w.data() + w.numel());
    W.data = reinterpret_cast<const uint8_t*>(W.owned.data());
    Tensor x = random_tensor({5, 64}, 4);
    Tensor y1, yN;
    { ScopedThreads st(1); y1 = ops::linear(x, W); }
    { ScopedThreads st(7); yN = ops::linear(x, W); }
    CHECK(max_abs_diff(y1.data(), yN.data(), y1.numel()) == 0.0f);
    // and against a naive triple loop
    float worst = 0;
    for (int t = 0; t < 5; ++t) for (int o = 0; o < 96; ++o) {
        double s = 0; for (int i = 0; i < 64; ++i) s += x.at(t, i) * w.at(o, i);
        worst = std::max(worst, (float)std::fabs(s - y1.at(t, o)));
    }
    CHECK(worst < 1e-4f);
}

TEST(rope_preserves_length_and_pos0_is_identity) {
    Tensor x = random_tensor({3, 32}, 5);
    Tensor r = x;
    ops::rope_inplace(r, 2, 16, 16, 0, 10000.0f);
    for (int c = 0; c < 32; ++c) CHECK_NEAR(r.at(0, c), x.at(0, c), 1e-6);  // angle 0 at position 0
    for (int t = 0; t < 3; ++t) for (int h = 0; h < 2; ++h) {
        double a = 0, b = 0;
        for (int d = 0; d < 16; ++d) { a += x.at(t, h * 16 + d) * x.at(t, h * 16 + d); b += r.at(t, h * 16 + d) * r.at(t, h * 16 + d); }
        CHECK_NEAR(a, b, 1e-4);   // a rotation never changes the length of a vector
    }
}

TEST(attention_first_token_sees_only_itself) {
    // With one key, softmax over one score = 1.0, so output == V[0] (per kv head).
    KVCache cache(8, 4);
    Tensor q = random_tensor({1, 16}, 6), k = random_tensor({1, 8}, 7), v = random_tensor({1, 8}, 8);
    cache.append(k, v);
    Tensor o = ops::attention(q, cache, 0, /*n_head=*/4, /*n_kv_head=*/2, /*head_dim=*/4);
    for (int h = 0; h < 4; ++h) for (int d = 0; d < 4; ++d)
        CHECK_NEAR(o.at(0, h * 4 + d), v.at(0, (h / 2) * 4 + d), 1e-6);
}

TEST(kv_cache_overflow_throws) {
    KVCache cache(4, 2);
    Tensor k = random_tensor({2, 4}, 9);
    cache.append(k, k);
    CHECK_THROWS(cache.append(k, k));
}

TEST(gguf_parse_metadata_and_tensors) {
    auto g = GGUFFile::open(data("tiny_q.gguf"));
    CHECK(g->version == 3);
    CHECK(g->get_str("general.architecture") == "llama");
    CHECK(g->get_int("llama.embedding_length") == 64);
    CHECK(g->get_int("llama.attention.head_count_kv") == 2);
    CHECK(g->tensors.size() == 3 + 2 * 9);
    const auto& q = g->tensor("blk.0.attn_q.weight");
    CHECK(q.type == GGMLType::Q8_0);
    CHECK(q.ne.size() == 2 && q.ne[0] == 64 && q.ne[1] == 64);
    CHECK(g->tensor("blk.1.attn_k.weight").type == GGMLType::Q4_0);
    CHECK(g->tensor("blk.0.ffn_up.weight").type == GGMLType::F16);
    CHECK(((g->tensor("output.weight").data - g->base()) % g->alignment) == 0);  // data is aligned
    CHECK_THROWS(g->tensor("no.such.tensor"));
    CHECK_THROWS(GGUFFile::open(data("prompt_ids.txt")));   // not a GGUF file
}

TEST(tokenizer_matches_reference_ids) {
    auto g = GGUFFile::open(data("tiny_f32.gguf"));
    Tokenizer tok = Tokenizer::from_gguf(*g);
    std::vector<int> ids = tok.encode("once upon a time there was a tiny llama");
    std::vector<int> expect = prompt_ids();
    CHECK(ids == expect);
    CHECK(tok.decode(ids) == "once upon a time there was a tiny llama");
    // characters not in the vocab fall back to byte tokens and still round-trip
    std::string weird = "zoo 42!";
    CHECK(tok.decode(tok.encode(weird)) == weird);
    // special tokens: expected ids were produced by real llama.cpp on this vocab
    CHECK((tok.encode("hi</s>there") == std::vector<int>{1, 259, 267, 268, 2, 318}));
    CHECK((tok.encode("a\n</s>\nb") == std::vector<int>{1, 289, 13, 2, 259, 13, 261}));
    CHECK((tok.encode("  two  spaces") == std::vector<int>{1, 259, 259, 288, 282, 274, 259, 259, 278, 275, 260, 291, 278}));
    CHECK((tok.encode("hi</s>there", true, false) == std::vector<int>{1, 259, 267, 268, 63, 50, 278, 65, 286, 317}));
}

// RAII switch for the activation-quantization mode.
struct ScopedActQuant {
    bool prev;
    explicit ScopedActQuant(bool on) : prev(quantize_activations()) { quantize_activations() = on; }
    ~ScopedActQuant() { quantize_activations() = prev; }
};

static void check_model_against_reference(const char* gguf, const char* ref, float tol) {
    ScopedActQuant exact(false);   // the NumPy reference uses float activations
    LlamaModel m = LlamaModel::load(data(gguf));
    std::vector<int> ids = prompt_ids();
    std::vector<float> R = read_floats(data(ref));            // [T, vocab]
    const int V = m.cfg.n_vocab, T = (int)ids.size();
    CHECK((int)R.size() == T * V);

    // (a) whole prompt at once ("prefill") -> last position
    auto caches = m.make_caches(64);
    Tensor last = m.forward(ids, caches, 0);
    float d_prefill = max_abs_diff(last.data(), R.data() + (size_t)(T - 1) * V, V);

    // (b) one token at a time ("decode") -> EVERY position, exercises KV cache + causal mask
    auto caches2 = m.make_caches(64);
    float d_step = 0;
    for (int t = 0; t < T; ++t) {
        Tensor lg = m.forward({ids[t]}, caches2, t);
        d_step = std::max(d_step, max_abs_diff(lg.data(), R.data() + (size_t)t * V, V));
    }
    std::printf("    %s: max |C++ - reference|  prefill %.2e   token-by-token %.2e\n", gguf, d_prefill, d_step);
    CHECK(d_prefill < tol);
    CHECK(d_step < tol);
}

TEST(model_f32_matches_reference) { check_model_against_reference("tiny_f32.gguf", "tiny_f32.ref.bin", 1e-3f); }
TEST(model_quantized_matches_reference) { check_model_against_reference("tiny_q.gguf", "tiny_q.ref.bin", 1e-3f); }
TEST(model_kquants_matches_reference) { check_model_against_reference("tiny_k.gguf", "tiny_k.ref.bin", 1e-3f); }

// The *.llamacpp.bin files hold the last-position logits that the REAL
// llama.cpp (via llama-cpp-python) produced for the same prompt. In its
// default (fast) mode our engine does the same activation quantization, so
// we should agree to within a couple of percent of the logit range.
static void check_against_llamacpp(const char* gguf, const char* ref) {
    ScopedActQuant fast(true);
    LlamaModel m = LlamaModel::load(data(gguf));
    auto caches = m.make_caches(64);
    Tensor lg = m.forward(prompt_ids(), caches, 0);
    std::vector<float> R = read_floats(data(ref));
    float range = 0;
    for (float v : R) range = std::max(range, std::fabs(v));
    float d = max_abs_diff(lg.data(), R.data(), R.size());
    std::printf("    %s: max |C++ - llama.cpp| = %.3g  (%.2f%% of max |logit| %.3g)\n", gguf, d, 100 * d / range, range);
    CHECK(d / range < 0.03f);
    CHECK(argmax(lg.data(), (int)lg.numel()) == argmax(R.data(), (int)R.size()));
}
TEST(agrees_with_llama_cpp_f32)    { check_against_llamacpp("tiny_f32.gguf", "tiny_f32.llamacpp.bin"); }
TEST(agrees_with_llama_cpp_q8_q4)  { check_against_llamacpp("tiny_q.gguf", "tiny_q.llamacpp.bin"); }
TEST(agrees_with_llama_cpp_kquant) { check_against_llamacpp("tiny_k.gguf", "tiny_k.llamacpp.bin"); }

TEST(quantized_activation_kernels_match_float_math) {
    // integer kernels vs "dequantize both sides, multiply in float"
    Tensor w = random_tensor({1, 128}, 11), x = random_tensor({1, 128}, 12);
    std::vector<BlockQ8_0> w8(4), x8(4); std::vector<BlockQ4_0> w4(4);
    quantize_row_q8_0(w.data(), w8.data(), 128);
    quantize_row_q4_0(w.data(), w4.data(), 128);
    quantize_row_q8_0(x.data(), x8.data(), 128);
    std::vector<float> wd8(128), wd4(128), xd(128);
    dequantize_row_q8_0(w8.data(), wd8.data(), 128);
    dequantize_row_q4_0(w4.data(), wd4.data(), 128);
    dequantize_row_q8_0(x8.data(), xd.data(), 128);
    CHECK_NEAR(dot_q8_0_q8_0(w8.data(), x8.data(), 4), dot_f32(wd8.data(), xd.data(), 128), 1e-3);
    CHECK_NEAR(dot_q4_0_q8_0(w4.data(), x8.data(), 4), dot_f32(wd4.data(), xd.data(), 128), 1e-3);
}

TEST(generation_is_identical_for_any_thread_count) {
    LlamaModel m = LlamaModel::load(data("tiny_q.gguf"));
    SamplingConfig sc; sc.temperature = 0.9f;   // real sampling, fixed seed
    std::vector<int> base;
    for (int threads : {1, 2, 4, 8}) {
        ScopedThreads st(threads);
        std::mt19937 rng(123);
        auto out = m.generate(prompt_ids(), 40, sc, rng, 64);
        if (base.empty()) base = out; else CHECK(out == base);
    }
}

TEST(missing_tensor_gives_clear_error) {
    // Loading a GGUF that is not a model must throw, not crash.
    CHECK_THROWS(LlamaModel::load(data("does_not_exist.gguf")));
}

// =====================================================================
int main(int argc, char** argv) {
    std::string filter;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--data" && i + 1 < argc) g_data = argv[++i];
        else filter = a;
    }
    int ran = 0, failed_tests = 0;
    for (auto& tc : registry()) {
        if (!filter.empty() && std::string(tc.name).find(filter) == std::string::npos) continue;
        ++ran;
        int before = g_failures;
        std::printf("[ RUN  ] %s\n", tc.name);
        try { tc.fn(); }
        catch (const std::exception& e) { ++g_failures; std::printf("    unexpected exception: %s\n", e.what()); }
        bool ok = g_failures == before;
        failed_tests += !ok;
        std::printf("[ %s ] %s\n", ok ? " OK " : "FAIL", tc.name);
    }
    std::printf("\n%d tests, %d failed  (threads available: %d)\n", ran, failed_tests, get_num_threads());
    return failed_tests ? 1 : 0;
}
