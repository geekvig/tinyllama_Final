#pragma once
// =====================================================================
//  model.hpp  --  Llama model: config, layer weights, forward pass,
//                 and the generation loop
// =====================================================================
//
//   token ids
//      |
//      v
//   [embedding lookup]  token_embd.weight          -> x : [T, dim]
//      |
//      v   repeat for every layer l = 0 .. n_layer-1  (block_forward):
//   +-----------------------------------------------------------+
//   |  h  = rmsnorm(x) * attn_norm                              |
//   |  q,k,v = h Wq^T, h Wk^T, h Wv^T                            |
//   |  rope(q), rope(k);  cache.append(k, v)                    |
//   |  a  = attention(q, cache)          (causal, GQA)          |
//   |  x  = x + a Wo^T                    <- residual #1        |
//   |  h  = rmsnorm(x) * ffn_norm                               |
//   |  x  = x + ( silu(h Wg^T) * (h Wu^T) ) Wd^T  <- residual #2 |
//   +-----------------------------------------------------------+
//      |
//      v
//   [final rmsnorm] output_norm.weight
//      |
//      v
//   [lm head]  output.weight (or token_embd.weight if tied) -> logits
//
//  Design doc: Chapter 4 (math) and Chapter 5.6 (this file).
// =====================================================================

#include "gguf.hpp"
#include "weights.hpp"
#include "ops.hpp"
#include "kv_cache.hpp"
#include "sampling.hpp"
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace tinyml {

// Hyper-parameters, read from GGUF metadata keys "<arch>.*".
struct LlamaConfig {
    std::string arch;
    int n_vocab = 0;
    int dim = 0;          // llama.embedding_length
    int n_layer = 0;      // llama.block_count
    int n_head = 0;       // llama.attention.head_count
    int n_kv_head = 0;    // llama.attention.head_count_kv (GQA)
    int head_dim = 0;     // dim / n_head
    int hidden_dim = 0;   // llama.feed_forward_length
    int n_ctx_train = 0;  // llama.context_length
    int n_rot = 0;        // llama.rope.dimension_count
    float rope_base = 10000.0f;  // llama.rope.freq_base
    float rms_eps = 1e-5f;       // llama.attention.layer_norm_rms_epsilon

    static LlamaConfig from_gguf(const GGUFFile& g) {
        LlamaConfig c;
        c.arch = g.get_str("general.architecture");
        if (c.arch != "llama")
            throw std::runtime_error("unsupported architecture '" + c.arch +
                                     "' (tinyllama implements 'llama': Llama 1/2, TinyLlama, "
                                     "Mistral-style GGUFs exported as llama)");
        const std::string a = c.arch + ".";
        c.dim         = (int)g.get_int(a + "embedding_length");
        c.n_layer     = (int)g.get_int(a + "block_count");
        c.n_head      = (int)g.get_int(a + "attention.head_count");
        c.n_kv_head   = (int)g.get_int(a + "attention.head_count_kv", c.n_head);
        c.hidden_dim  = (int)g.get_int(a + "feed_forward_length");
        c.n_ctx_train = (int)g.get_int(a + "context_length", 2048);
        c.head_dim    = c.dim / c.n_head;
        c.n_rot       = (int)g.get_int(a + "rope.dimension_count", c.head_dim);
        c.rope_base   = (float)g.get_float(a + "rope.freq_base", 10000.0);
        c.rms_eps     = (float)g.get_float(a + "attention.layer_norm_rms_epsilon", 1e-5);
        c.n_vocab     = (int)g.tensor("token_embd.weight").ne[1];
        if (c.dim % c.n_head || c.n_head % c.n_kv_head)
            throw std::runtime_error("config: dim/n_head/n_kv_head are not divisible");
        return c;
    }
};

struct LayerWeights {
    Tensor attn_norm, ffn_norm;        // [dim]
    Weight wq, wk, wv, wo;             // [dim,dim], [kv_dim,dim], [kv_dim,dim], [dim,dim]
    Weight w_gate, w_up, w_down;       // [hidden,dim], [hidden,dim], [dim,hidden]
};

// One decoder block. x: [T, dim] is updated in place.
inline void block_forward(Tensor& x, const LayerWeights& L, const LlamaConfig& c,
                          KVCache& cache, int pos0) {
    // ---- attention sub-layer ----
    Tensor h = ops::rmsnorm(x, L.attn_norm, c.rms_eps);
    Tensor q = ops::linear(h, L.wq);
    Tensor k = ops::linear(h, L.wk);
    Tensor v = ops::linear(h, L.wv);
    ops::rope_inplace(q, c.n_head,    c.head_dim, c.n_rot, pos0, c.rope_base);
    ops::rope_inplace(k, c.n_kv_head, c.head_dim, c.n_rot, pos0, c.rope_base);
    cache.append(k, v);
    Tensor a = ops::attention(q, cache, pos0, c.n_head, c.n_kv_head, c.head_dim);
    ops::add_inplace(x, ops::linear(a, L.wo));            // residual #1

    // ---- SwiGLU feed-forward sub-layer ----
    h = ops::rmsnorm(x, L.ffn_norm, c.rms_eps);
    Tensor gate = ops::linear(h, L.w_gate);
    Tensor up   = ops::linear(h, L.w_up);
    ops::silu_mul_inplace(gate, up);
    ops::add_inplace(x, ops::linear(gate, L.w_down));     // residual #2
}

class LlamaModel {
public:
    LlamaConfig cfg;

    // Load everything from a GGUF file. Throws std::runtime_error with a
    // readable message on any problem.
    static LlamaModel load(const std::string& path) {
        LlamaModel m;
        m.file_ = GGUFFile::open(path);       // keeps the bytes alive (zero-copy weights)
        const GGUFFile& g = *m.file_;
        m.cfg = LlamaConfig::from_gguf(g);

        m.tok_embd_ = Weight::from_gguf(g.tensor("token_embd.weight"));
        m.output_norm_ = vector_from_gguf(g.tensor("output_norm.weight"));
        if (const GGUFTensorInfo* out = g.find_tensor("output.weight")) {
            m.output_ = Weight::from_gguf(*out);
        } else {
            m.output_ = Weight::from_gguf(g.tensor("token_embd.weight")); // tied embeddings
        }

        m.layers_.resize(m.cfg.n_layer);
        for (int l = 0; l < m.cfg.n_layer; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            LayerWeights& L = m.layers_[l];
            L.attn_norm = vector_from_gguf(g.tensor(p + "attn_norm.weight"));
            L.wq     = Weight::from_gguf(g.tensor(p + "attn_q.weight"));
            L.wk     = Weight::from_gguf(g.tensor(p + "attn_k.weight"));
            L.wv     = Weight::from_gguf(g.tensor(p + "attn_v.weight"));
            L.wo     = Weight::from_gguf(g.tensor(p + "attn_output.weight"));
            L.ffn_norm = vector_from_gguf(g.tensor(p + "ffn_norm.weight"));
            L.w_gate = Weight::from_gguf(g.tensor(p + "ffn_gate.weight"));
            L.w_up   = Weight::from_gguf(g.tensor(p + "ffn_up.weight"));
            L.w_down = Weight::from_gguf(g.tensor(p + "ffn_down.weight"));
        }
        return m;
    }

    const GGUFFile& gguf() const { return *file_; }

    std::vector<KVCache> make_caches(int n_ctx) const {
        std::vector<KVCache> caches;
        caches.reserve(cfg.n_layer);
        for (int l = 0; l < cfg.n_layer; ++l) caches.emplace_back(cfg.n_kv_head * cfg.head_dim, n_ctx);
        return caches;
    }

    // Run `ids` (absolute positions pos0 .. pos0+T-1) through the network
    // and return the logits of the LAST position: [1, n_vocab].
    Tensor forward(const std::vector<int>& ids, std::vector<KVCache>& caches, int pos0) const {
        Tensor x = ops::embed(tok_embd_, ids);
        for (int l = 0; l < cfg.n_layer; ++l) block_forward(x, layers_[l], cfg, caches[l], pos0);

        Tensor last({1, cfg.dim});           // only the last row predicts the next token
        const float* src = x.row(x.dim(0) - 1);
        std::copy(src, src + cfg.dim, last.row(0));
        Tensor normed = ops::rmsnorm(last, output_norm_, cfg.rms_eps);
        return ops::linear(normed, output_);
    }

    // Prefill the prompt once, then decode one token at a time.
    // on_token(id) is called for each new token; return false to stop early.
    std::vector<int> generate(const std::vector<int>& prompt, int n_new, const SamplingConfig& sc,
                              std::mt19937& rng, int n_ctx, int eos_id = -1,
                              const std::function<bool(int)>& on_token = nullptr) const {
        if (prompt.empty()) throw std::runtime_error("generate: empty prompt");
        auto caches = make_caches(n_ctx);
        std::vector<int> ids = prompt;
        Tensor logits = forward(prompt, caches, 0);
        for (int step = 0; step < n_new; ++step) {
            int next = sample_token(logits, sc, rng, ids);
            ids.push_back(next);
            if (on_token && !on_token(next)) break;
            if (next == eos_id) break;
            if ((int)ids.size() >= n_ctx) break;          // context full
            logits = forward({next}, caches, (int)ids.size() - 1);
        }
        return ids;
    }

private:
    std::shared_ptr<GGUFFile> file_;   // MUST outlive every Weight below
    Weight tok_embd_, output_;
    Tensor output_norm_;
    std::vector<LayerWeights> layers_;
};

} // namespace tinyml
