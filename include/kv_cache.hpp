#pragma once
// =====================================================================
//  kv_cache.hpp  --  per-layer memory of past Keys and Values
// =====================================================================
//  When generating token number 100, attention needs the K and V vectors
//  of tokens 0..99. They never change once computed (causal attention:
//  old tokens never look at new ones), so we compute each one ONCE and
//  keep it here. Without the cache every new token would recompute the
//  whole prompt: O(n^2) total work instead of O(n).
//
//  Layout: two flat buffers of [max_len, kv_dim] floats, row-major.
//  kv_dim = n_kv_head * head_dim  (smaller than dim when using GQA --
//  that is the whole point of GQA: a smaller cache).
//
//  Attention reads rows straight out of these buffers through key(j)
//  and value(j): no copying per step (the original code copied the
//  whole cache into a new Tensor on every call).
//
//  Design doc: Chapter 4.6 and 5.5.
// =====================================================================

#include "tensor.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace tinyml {

class KVCache {
public:
    KVCache(int kv_dim, int max_len) : kv_dim_(kv_dim), max_len_(max_len) {
        k_.assign((size_t)max_len * kv_dim, 0.0f);
        v_.assign((size_t)max_len * kv_dim, 0.0f);
    }

    // Append T new rows. k, v: [T, kv_dim]
    void append(const Tensor& k, const Tensor& v) {
        const int T = k.dim(0);
        if (k.dim(1) != kv_dim_ || v.dim(1) != kv_dim_ || v.dim(0) != T)
            throw std::runtime_error("KVCache::append: shape mismatch");
        if (len_ + T > max_len_)   // Bug Lab 4 deletes this line
            throw std::runtime_error("KVCache: context full (" + std::to_string(max_len_) +
                                     " tokens). Use a larger -c / --ctx.");
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < kv_dim_; ++c) {
                k_[(size_t)(len_ + t) * kv_dim_ + c] = k.at(t, c);
                v_[(size_t)(len_ + t) * kv_dim_ + c] = v.at(t, c);
            }
        len_ += T;
    }

    const float* key(int pos) const   { return k_.data() + (size_t)pos * kv_dim_; }
    const float* value(int pos) const { return v_.data() + (size_t)pos * kv_dim_; }

    int length() const { return len_; }
    int capacity() const { return max_len_; }
    void clear() { len_ = 0; }

private:
    int kv_dim_, max_len_, len_ = 0;
    std::vector<float> k_, v_;
};

} // namespace tinyml
