#pragma once
// =====================================================================
//  ops.hpp  --  the math building blocks of a Llama layer
// =====================================================================
//  Every function here is a pure function of its inputs (no hidden
//  state) and splits its work with parallel_for when it is big enough.
//
//    rmsnorm      -- normalise each row, scale by a learned weight
//    rope         -- rotate Q/K pairs by a position-dependent angle
//    attention    -- causal multi-head attention with GQA + KV cache
//    silu_mul     -- SwiGLU's   silu(gate) * up
//    add_inplace  -- residual connection  x += y
//    softmax      -- turn scores into probabilities
//    embed        -- token ids -> rows of the embedding matrix
//
//  Design doc: Chapter 4 (the math) and Chapter 5.4 (this file).
// =====================================================================

#include "tensor.hpp"
#include "thread_pool.hpp"
#include "weights.hpp"
#include "kv_cache.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tinyml {
namespace ops {

// Don't wake the pool for less work than this (wake-up costs ~microseconds).
constexpr int64_t kParallelWork = 1 << 15;

inline int64_t rows_per_task(int cols) {
    return std::max<int64_t>(1, kParallelWork / std::max(1, cols));
}

// ---------------------------------------------------------------------
// RMSNorm:  y = x / sqrt(mean(x^2) + eps) * weight
// (LayerNorm without the mean subtraction and bias -- what Llama uses.)
// ---------------------------------------------------------------------
inline Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps) {
    const int rows = x.dim(0), dim = x.dim(1);
    if ((int)weight.numel() != dim) throw std::runtime_error("rmsnorm: weight size mismatch");
    Tensor out(x.shape());
    parallel_for(rows, [&](int64_t r0, int64_t r1) {
        for (int r = (int)r0; r < (int)r1; ++r) {
            const float* xr = x.row(r);
            float ss = 0.0f;
            for (int c = 0; c < dim; ++c) ss += xr[c] * xr[c];
            const float scale = 1.0f / std::sqrt(ss / dim + eps);
            float* o = out.row(r);
            for (int c = 0; c < dim; ++c) o[c] = xr[c] * scale * weight.at_flat(c);
        }
    }, rows_per_task(dim));
    return out;
}

// ---------------------------------------------------------------------
// RoPE (rotary position embedding), "NORM" style as llama.cpp uses for
// the llama architecture:
//   inside EACH HEAD, element pairs (2i, 2i+1) are rotated by angle
//        theta = pos * base^(-2i / n_rot)
//   where n_rot is normally head_dim. Lower pairs spin fast (they
//   encode fine position), higher pairs spin slowly (coarse position).
// x: [T, n_heads * head_dim], modified in place.
// The angle depends on the index INSIDE the head -- using the index in
// the whole row is a classic bug (Bug Lab 7; the original tinyml had it).
// ---------------------------------------------------------------------
inline void rope_inplace(Tensor& x, int n_heads, int head_dim, int n_rot,
                         int position_offset, float freq_base) {
    const int T = x.dim(0);
    if (x.dim(1) != n_heads * head_dim) throw std::runtime_error("rope: shape mismatch");
    // Pre-compute the inverse frequencies once (same for every head/token).
    std::vector<float> inv_freq(n_rot / 2);
    for (int i = 0; i < n_rot / 2; ++i)
        inv_freq[i] = std::pow(freq_base, -2.0f * i / (float)n_rot);

    parallel_for(T, [&](int64_t t0, int64_t t1) {
        for (int t = (int)t0; t < (int)t1; ++t) {
            const float pos = (float)(position_offset + t);
            float* row = x.row(t);
            for (int h = 0; h < n_heads; ++h) {
                float* v = row + h * head_dim;
                for (int i = 0; i < n_rot / 2; ++i) {
                    const float a = pos * inv_freq[i];
                    const float c = std::cos(a), s = std::sin(a);
                    const float x0 = v[2 * i], x1 = v[2 * i + 1];
                    v[2 * i]     = x0 * c - x1 * s;
                    v[2 * i + 1] = x0 * s + x1 * c;
                }
            }
        }
    }, rows_per_task(x.dim(1)));
}

// ---------------------------------------------------------------------
// Causal multi-head attention with Grouped-Query Attention (GQA).
//
//   q:      [T, n_head * head_dim]   queries for the T NEW tokens
//   cache:  holds K and V for positions 0 .. pos0+T-1 (already appended)
//   pos0:   absolute position of q's first row
//
// GQA: n_head query heads share n_kv_head K/V heads. Query head h uses
// kv head  h / (n_head / n_kv_head).  (n_kv_head == n_head is ordinary
// multi-head attention; TinyLlama has 32 query heads and 4 kv heads.)
//
// Causal mask: the query at absolute position p may look at keys
// 0..p (inclusive) -- never the future.
//
// Threading: one chunk = a range of query heads. Every chunk has its OWN
// scratch buffer for the scores (Bug Lab 10 shares one -> data race).
// ---------------------------------------------------------------------
inline Tensor attention(const Tensor& q, const KVCache& cache, int pos0,
                        int n_head, int n_kv_head, int head_dim) {
    const int T = q.dim(0);
    const int n_ctx_used = pos0 + T;
    if (n_ctx_used > cache.length()) throw std::runtime_error("attention: cache shorter than positions");
    const int group = n_head / n_kv_head;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    Tensor out({T, n_head * head_dim});

    parallel_for(n_head, [&](int64_t h0, int64_t h1) {
        std::vector<float> scores(n_ctx_used);               // per-chunk scratch
        for (int h = (int)h0; h < (int)h1; ++h) {
            const int kvh = h / group;                        // GQA mapping
            for (int t = 0; t < T; ++t) {
                const int p = pos0 + t;                       // absolute position of this query
                const float* qv = q.row(t) + h * head_dim;
                // 1) scores against keys 0..p
                float maxs = -INFINITY;
                for (int j = 0; j <= p; ++j) {
                    float s = dot_f32(qv, cache.key(j) + kvh * head_dim, head_dim) * scale;
                    scores[j] = s;
                    maxs = std::max(maxs, s);
                }
                // 2) softmax (subtract max for numerical stability)
                float sum = 0.0f;
                for (int j = 0; j <= p; ++j) { scores[j] = std::exp(scores[j] - maxs); sum += scores[j]; }
                const float inv = 1.0f / sum;
                // 3) weighted sum of values
                float* o = out.row(t) + h * head_dim;
                for (int d = 0; d < head_dim; ++d) o[d] = 0.0f;
                for (int j = 0; j <= p; ++j) {
                    const float w = scores[j] * inv;
                    const float* vv = cache.value(j) + kvh * head_dim;
                    for (int d = 0; d < head_dim; ++d) o[d] += w * vv[d];
                }
            }
        }
    });
    return out;
}

// SwiGLU's gating:  out = silu(gate) * up,   silu(z) = z * sigmoid(z)
inline void silu_mul_inplace(Tensor& gate, const Tensor& up) {
    if (gate.numel() != up.numel()) throw std::runtime_error("silu_mul: size mismatch");
    parallel_for((int64_t)gate.numel(), [&](int64_t lo, int64_t hi) {
        for (int64_t i = lo; i < hi; ++i) {
            const float g = gate.at_flat(i);
            gate.at_flat(i) = g / (1.0f + std::exp(-g)) * up.at_flat(i);
        }
    }, kParallelWork);
}

// Residual connection: x += y
inline void add_inplace(Tensor& x, const Tensor& y) {
    if (x.numel() != y.numel()) throw std::runtime_error("add: size mismatch");
    parallel_for((int64_t)x.numel(), [&](int64_t lo, int64_t hi) {
        for (int64_t i = lo; i < hi; ++i) x.at_flat(i) += y.at_flat(i);
    }, kParallelWork);
}

// Softmax of a single vector (used by sampling).
inline std::vector<float> softmax(const std::vector<float>& z) {
    std::vector<float> p(z.size());
    if (z.empty()) return p;
    float m = *std::max_element(z.begin(), z.end()), sum = 0.0f;
    for (size_t i = 0; i < z.size(); ++i) { p[i] = std::exp(z[i] - m); sum += p[i]; }
    for (float& v : p) v /= sum;
    return p;
}

// Token ids -> [T, dim] by copying (and dequantizing) embedding rows.
inline Tensor embed(const Weight& table, const std::vector<int>& ids) {
    Tensor out({(int)ids.size(), table.cols});
    for (int id : ids)   // validate on the calling thread so errors are clear
        if (id < 0 || id >= table.rows)
            throw std::runtime_error("embed: token id " + std::to_string(id) + " out of range");
    parallel_for((int64_t)ids.size(), [&](int64_t lo, int64_t hi) {
        for (int64_t i = lo; i < hi; ++i) table.dequantize_row(ids[(size_t)i], out.row((int)i));
    });
    return out;
}

} // namespace ops
} // namespace tinyml
