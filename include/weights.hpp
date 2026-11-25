#pragma once
// =====================================================================
//  weights.hpp  --  a weight matrix that may be F32, Q8_0 or Q4_0,
//                   and the multithreaded `linear` op that uses it
// =====================================================================
//  CONVENTION (same as PyTorch nn.Linear and ggml):
//      W has shape [out_features, in_features]   ("rows" x "cols")
//      y = linear(x, W)   means   y[t][o] = sum_i x[t][i] * W[o][i]
//
//  Why rows = OUTPUT features? Because then computing one output
//  number is a dot product of two CONTIGUOUS arrays (a row of x and a
//  row of W) -- perfect for SIMD and for the CPU cache. It is also
//  exactly how a GGUF file lays out a matrix: GGML's ne = [in, out]
//  means "in" is the fast-moving dimension, i.e. each row has `in`
//  elements.  (The original tinyml code used x @ W with W = [in, out];
//  converting to this layout was the biggest change for GGUF.)
//
//  F16 / BF16 weights are converted to F32 once at load time (simple,
//  costs 2x memory for those tensors). Q8_0 / Q4_0 stay quantized and
//  are multiplied directly ("fused dequantize + dot"). K-quants (Q4_K,
//  Q5_K, Q6_K) are dequantized one row at a time into scratch memory.
//
//  Design doc: Chapter 5.3 and Chapter 6.
// =====================================================================

#include "tensor.hpp"
#include "quant.hpp"
#include "gguf.hpp"
#include "thread_pool.hpp"
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace tinyml {

enum class WType { F32, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K };

struct Weight {
    WType type = WType::F32;
    int rows = 0;                   // out_features
    int cols = 0;                   // in_features
    const uint8_t* data = nullptr;  // points into the GGUF buffer OR into `owned`
    std::vector<float> owned;       // used when we had to convert (F16 -> F32)
    std::string name;

    bool empty() const { return data == nullptr; }

    size_t row_bytes() const {
        switch (type) {
            case WType::F32:  return (size_t)cols * 4;
            case WType::Q8_0: return (size_t)cols / QK * sizeof(BlockQ8_0);
            case WType::Q4_0: return (size_t)cols / QK * sizeof(BlockQ4_0);
            case WType::Q4_K: return (size_t)cols / QK_K * sizeof(BlockQ4_K);
            case WType::Q5_K: return (size_t)cols / QK_K * sizeof(BlockQ5_K);
            case WType::Q6_K: return (size_t)cols / QK_K * sizeof(BlockQ6_K);
        }
        return 0;
    }
    const uint8_t* row_ptr(int r) const { return data + (size_t)r * row_bytes(); }

    // dot( x[0..cols), row r )
    float dot_row(const float* x, int r) const {
        const uint8_t* p = row_ptr(r);
        switch (type) {
            case WType::F32:  return dot_f32(x, reinterpret_cast<const float*>(p), cols);
            case WType::Q8_0: return dot_q8_0(x, reinterpret_cast<const BlockQ8_0*>(p), cols);
            case WType::Q4_0: return dot_q4_0(x, reinterpret_cast<const BlockQ4_0*>(p), cols);
            default: {
                // K-quants: dequantize the row into a per-thread scratch buffer.
                // thread_local = every thread gets its own copy -> no data race.
                thread_local std::vector<float> scratch;
                scratch.resize(cols);
                dequantize_row(r, scratch.data());
                return dot_f32(x, scratch.data(), cols);
            }
        }
    }

    // Copy row r out as floats (used by the embedding lookup).
    void dequantize_row(int r, float* out) const {
        const uint8_t* p = row_ptr(r);
        switch (type) {
            case WType::F32:  std::memcpy(out, p, (size_t)cols * 4); break;
            case WType::Q8_0: dequantize_row_q8_0(reinterpret_cast<const BlockQ8_0*>(p), out, cols); break;
            case WType::Q4_0: dequantize_row_q4_0(reinterpret_cast<const BlockQ4_0*>(p), out, cols); break;
            case WType::Q4_K: dequantize_row_q4_K(reinterpret_cast<const BlockQ4_K*>(p), out, cols); break;
            case WType::Q5_K: dequantize_row_q5_K(reinterpret_cast<const BlockQ5_K*>(p), out, cols); break;
            case WType::Q6_K: dequantize_row_q6_K(reinterpret_cast<const BlockQ6_K*>(p), out, cols); break;
        }
    }

    // Build a Weight from a GGUF tensor. 1-D tensors (norm weights) come
    // out as rows=1. Throws a clear error for types we do not support.
    static Weight from_gguf(const GGUFTensorInfo& t) {
        Weight w;
        w.name = t.name;
        w.cols = (int)t.ne[0];
        uint64_t rows = 1;
        for (size_t d = 1; d < t.ne.size(); ++d) rows *= t.ne[d];
        w.rows = (int)rows;
        const uint64_t n = t.n_elements();
        switch (t.type) {
            case GGMLType::F32:
                w.type = WType::F32; w.data = t.data; break;
            case GGMLType::F16: {
                w.type = WType::F32;
                w.owned.resize(n);
                const uint16_t* src = reinterpret_cast<const uint16_t*>(t.data);
                for (uint64_t i = 0; i < n; ++i) {
                    uint16_t h; std::memcpy(&h, src + i, 2);
                    w.owned[i] = fp16_to_fp32(h);
                }
                w.data = reinterpret_cast<const uint8_t*>(w.owned.data());
                break;
            }
            case GGMLType::BF16: {
                w.type = WType::F32;
                w.owned.resize(n);
                for (uint64_t i = 0; i < n; ++i) {
                    uint16_t h; std::memcpy(&h, t.data + 2 * i, 2);
                    uint32_t bits = (uint32_t)h << 16; float f; std::memcpy(&f, &bits, 4);
                    w.owned[i] = f;
                }
                w.data = reinterpret_cast<const uint8_t*>(w.owned.data());
                break;
            }
            case GGMLType::Q8_0: w.type = WType::Q8_0; w.data = t.data; break;
            case GGMLType::Q4_0: w.type = WType::Q4_0; w.data = t.data; break;
            case GGMLType::Q4_K: w.type = WType::Q4_K; w.data = t.data; break;
            case GGMLType::Q5_K: w.type = WType::Q5_K; w.data = t.data; break;
            case GGMLType::Q6_K: w.type = WType::Q6_K; w.data = t.data; break;
            default:
                throw std::runtime_error(std::string("tensor '") + t.name + "' has type " +
                    ggml_type_name(t.type) + ", which tinyllama does not implement yet "
                    "(supported: F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K).");
        }
        // vector::data() of a moved vector keeps pointing at the same heap
        // block, but a COPY would not -- so Weight must only ever be moved.
        return w;
    }

    Weight() = default;
    Weight(Weight&&) = default;
    Weight& operator=(Weight&&) = default;
    Weight(const Weight&) = delete;            // see comment above
    Weight& operator=(const Weight&) = delete;
};

// Small F32 vector parameter (RMSNorm weights) as a plain Tensor.
inline Tensor vector_from_gguf(const GGUFTensorInfo& t) {
    Weight w = Weight::from_gguf(t);
    if (w.type != WType::F32) throw std::runtime_error("expected float vector for " + t.name);
    std::vector<float> v((size_t)w.rows * w.cols);
    std::memcpy(v.data(), w.data, v.size() * 4);
    // Read the size BEFORE std::move(v): C++ does not promise which
    // function argument is evaluated first (a real bug I hit -- Chapter 8).
    const int n = (int)v.size();
    return Tensor({n}, std::move(v));
}

// Global switch: quantize activations to Q8_0 before multiplying by
// Q8_0/Q4_0 weights (llama.cpp does this; ~2-3x faster). Turn it off
// (--f32-act) to get exact float activations -- the unit tests do that to
// compare against the float reference bit-for-bit-ish.
inline bool& quantize_activations() { static bool on = true; return on; }

namespace ops {

// y = x W^T.   x: [T, in]  ->  y: [T, out]
//
// Threading: we split the OUTPUT features (rows of W) across threads.
// Each thread reads its slice of W exactly once and reuses each row for
// all T tokens -- good for both prefill (T large) and decode (T = 1).
// Each y[t][o] is written by exactly one thread -> no locks needed and
// results are identical for any thread count.
inline Tensor linear(const Tensor& x, const Weight& W) {
    if (W.empty()) throw std::runtime_error("linear: weight '" + W.name + "' is empty");
    if (x.ndim() != 2 || x.dim(1) != W.cols)
        throw std::runtime_error("linear: shape mismatch " + x.to_string() + " vs weight '" +
                                 W.name + "' [" + std::to_string(W.rows) + "," + std::to_string(W.cols) + "]");
    const int T = x.dim(0), OUT = W.rows;
    Tensor y({T, OUT});

    if (quantize_activations() && (W.type == WType::Q8_0 || W.type == WType::Q4_0)) {
        // 1) quantize every activation row once (shared, read-only afterwards)
        const int nb = W.cols / QK;
        std::vector<BlockQ8_0> xq((size_t)T * nb);
        parallel_for(T, [&](int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t) quantize_row_q8_0(x.row((int)t), &xq[(size_t)t * nb], W.cols);
        });
        // 2) integer dot products, split over output rows as below
        parallel_for(OUT, [&](int64_t o0, int64_t o1) {
            for (int64_t o = o0; o < o1; ++o) {
                const uint8_t* wr = W.row_ptr((int)o);
                for (int t = 0; t < T; ++t) {
                    const BlockQ8_0* xr = &xq[(size_t)t * nb];
                    y.at(t, (int)o) = (W.type == WType::Q8_0)
                        ? dot_q8_0_q8_0(reinterpret_cast<const BlockQ8_0*>(wr), xr, nb)
                        : dot_q4_0_q8_0(reinterpret_cast<const BlockQ4_0*>(wr), xr, nb);
                }
            }
        }, 16);
        return y;
    }

    parallel_for(OUT, [&](int64_t o0, int64_t o1) {
        for (int64_t o = o0; o < o1; ++o)
            for (int t = 0; t < T; ++t)
                y.at(t, (int)o) = W.dot_row(x.row(t), (int)o);
    }, /*min_per_task=*/16);
    return y;
}

} // namespace ops
} // namespace tinyml
