#pragma once
// =====================================================================
//  quant.hpp  --  ggml-compatible Q8_0 and Q4_0 blocks
// =====================================================================
//  Quantization stores weights as small integers plus a scale.
//  Weights are cut into blocks of 32 numbers (QK = 32). Each block
//  stores ONE fp16 scale `d` and 32 small integers q[i]:
//
//        weight[i]  ~=  d * q[i]
//
//  These structs are BYTE-FOR-BYTE identical to ggml's block_q8_0 and
//  block_q4_0 (ggml/src/ggml-common.h), so we can point them straight
//  at the bytes inside a .gguf file.
//
//    Q8_0:  2 bytes (d) + 32 bytes (int8 q)          = 34 bytes / 32 w
//    Q4_0:  2 bytes (d) + 16 bytes (two 4-bit q/byte) = 18 bytes / 32 w
//
//  Q4_0 nibble order (easy to get wrong -- see Bug Lab 3):
//    byte j (0..15) holds  element j      in its LOW  nibble
//                    and   element j+16   in its HIGH nibble.
//    Each nibble is stored with +8 added, so the value is (nibble - 8).
//
//  K-quants (Q4_K/Q5_K/Q6_K) are at the bottom of this file.
//
//  Design doc: Chapter 3.6 and Chapter 6.
// =====================================================================

#include "fp16.hpp"
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define TINYML_HAVE_AVX2 1
#endif

namespace tinyml {

constexpr int QK = 32;

#pragma pack(push, 1)
struct BlockQ8_0 {
    uint16_t d;        // fp16 scale
    int8_t   qs[QK];   // quantized values
};
struct BlockQ4_0 {
    uint16_t d;           // fp16 scale
    uint8_t  qs[QK / 2];  // nibbles (see header comment for order)
};
#pragma pack(pop)
static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 must match ggml (34 bytes)");
static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 must match ggml (18 bytes)");

// ------------------------------------------------------------------
// Quantize (float -> blocks). Same rounding as ggml's reference code.
// We only need these for tests / for writing our own files.
// ------------------------------------------------------------------
inline void quantize_row_q8_0(const float* x, BlockQ8_0* y, int n) {
    for (int b = 0; b < n / QK; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < QK; ++i) amax = std::max(amax, std::fabs(x[b * QK + i]));
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        y[b].d = fp32_to_fp16(d);
        for (int i = 0; i < QK; ++i) y[b].qs[i] = (int8_t)std::lround(x[b * QK + i] * id);
    }
}

inline void quantize_row_q4_0(const float* x, BlockQ4_0* y, int n) {
    for (int b = 0; b < n / QK; ++b) {
        float amax = 0.0f, maxv = 0.0f;   // keep the SIGN of the largest value
        for (int i = 0; i < QK; ++i) {
            float v = x[b * QK + i];
            if (std::fabs(v) > amax) { amax = std::fabs(v); maxv = v; }
        }
        const float d = maxv / -8.0f;
        const float id = d ? 1.0f / d : 0.0f;
        y[b].d = fp32_to_fp16(d);
        for (int j = 0; j < QK / 2; ++j) {
            int q0 = std::min(15, (int)(x[b * QK + j] * id + 8.5f));
            int q1 = std::min(15, (int)(x[b * QK + j + QK / 2] * id + 8.5f));
            y[b].qs[j] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

// ------------------------------------------------------------------
// Dequantize (blocks -> float). Used for embeddings lookup and tests.
// ------------------------------------------------------------------
inline void dequantize_row_q8_0(const BlockQ8_0* x, float* y, int n) {
    for (int b = 0; b < n / QK; ++b) {
        const float d = fp16_to_fp32(x[b].d);
        for (int i = 0; i < QK; ++i) y[b * QK + i] = x[b].qs[i] * d;
    }
}

inline void dequantize_row_q4_0(const BlockQ4_0* x, float* y, int n) {
    for (int b = 0; b < n / QK; ++b) {
        const float d = fp16_to_fp32(x[b].d);
        for (int j = 0; j < QK / 2; ++j) {
            y[b * QK + j]          = ((x[b].qs[j] & 0x0F) - 8) * d;
            y[b * QK + j + QK / 2] = ((x[b].qs[j] >> 4)   - 8) * d;
        }
    }
}

// ------------------------------------------------------------------
// Dot products: float activations . weights row.  These are the hot
// loops of the whole program (90%+ of runtime).
// ------------------------------------------------------------------
#ifdef TINYML_HAVE_AVX2
inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}
#endif

inline float dot_f32(const float* a, const float* b, int n) {
    int i = 0;
    float sum = 0.0f;
#ifdef TINYML_HAVE_AVX2
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    sum = hsum256(acc);
#endif
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

inline float dot_q8_0(const float* a, const BlockQ8_0* w, int n) {
    float total = 0.0f;
    for (int b = 0; b < n / QK; ++b) {
        const float* ab = a + b * QK;
        float s;
#ifdef TINYML_HAVE_AVX2
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < QK; i += 8) {
            __m128i q8  = _mm_loadl_epi64((const __m128i*)(w[b].qs + i));   // 8 x int8
            __m256  qf  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));      // -> 8 x float
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(ab + i), qf, acc);
        }
        s = hsum256(acc);
#else
        s = 0.0f;
        for (int i = 0; i < QK; ++i) s += ab[i] * (float)w[b].qs[i];
#endif
        total += s * fp16_to_fp32(w[b].d);   // apply the scale ONCE per block
    }
    return total;
}

inline float dot_q4_0(const float* a, const BlockQ4_0* w, int n) {
    float total = 0.0f;
    for (int b = 0; b < n / QK; ++b) {
        const float* ab = a + b * QK;
        float s = 0.0f;
        for (int j = 0; j < QK / 2; ++j) {
            s += ab[j]          * (float)((w[b].qs[j] & 0x0F) - 8);
            s += ab[j + QK / 2] * (float)((w[b].qs[j] >> 4)   - 8);
        }
        total += s * fp16_to_fp32(w[b].d);
    }
    return total;
}

// ------------------------------------------------------------------
// INTEGER dot products:  quantized weights . quantized activations.
//
// This is llama.cpp's main CPU speed trick (ggml_vec_dot_q8_0_q8_0).
// Before a matmul, the activation row x is itself quantized to Q8_0.
// Then each block's dot product is pure int8 x int8 -> int32 math, and
// only ONE float multiply per 32 elements remains:
//
//     sum_i w[i]*x[i]  ~=  (d_w * d_x) * sum_i qw[i]*qx[i]
//
// With AVX2, 32 int8 products are done by 2 instructions.
// ------------------------------------------------------------------
#ifdef TINYML_HAVE_AVX2
// 32 x (int8 * int8), summed in pairs -> 8 x float
inline __m256 mul_sum_i8_pairs_float(__m256i x, __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);          // |x| (now unsigned-safe)
    const __m256i sy = _mm256_sign_epi8(y, x);          // y * sign(x)
    const __m256i dot16 = _mm256_maddubs_epi16(ax, sy); // u8*s8 pairs -> 16 x int16
    const __m256i dot32 = _mm256_madd_epi16(dot16, _mm256_set1_epi16(1)); // -> 8 x int32
    return _mm256_cvtepi32_ps(dot32);
}
#endif

inline float dot_q8_0_q8_0(const BlockQ8_0* w, const BlockQ8_0* x, int nb) {
#ifdef TINYML_HAVE_AVX2
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const __m256 d = _mm256_set1_ps(fp16_to_fp32(w[b].d) * fp16_to_fp32(x[b].d));
        const __m256i qw = _mm256_loadu_si256((const __m256i*)w[b].qs);
        const __m256i qx = _mm256_loadu_si256((const __m256i*)x[b].qs);
        acc = _mm256_fmadd_ps(d, mul_sum_i8_pairs_float(qw, qx), acc);
    }
    return hsum256(acc);
#else
    float total = 0.0f;
    for (int b = 0; b < nb; ++b) {
        int s = 0;
        for (int i = 0; i < QK; ++i) s += (int)w[b].qs[i] * (int)x[b].qs[i];
        total += (float)s * fp16_to_fp32(w[b].d) * fp16_to_fp32(x[b].d);
    }
    return total;
#endif
}

inline float dot_q4_0_q8_0(const BlockQ4_0* w, const BlockQ8_0* x, int nb) {
#ifdef TINYML_HAVE_AVX2
    __m256 acc = _mm256_setzero_ps();
    const __m256i low4 = _mm256_set1_epi8(0x0F), eight = _mm256_set1_epi8(8);
    for (int b = 0; b < nb; ++b) {
        const __m256 d = _mm256_set1_ps(fp16_to_fp32(w[b].d) * fp16_to_fp32(x[b].d));
        // unpack 16 bytes -> 32 nibbles: low nibbles = elements 0..15, high = 16..31
        const __m128i raw = _mm_loadu_si128((const __m128i*)w[b].qs);
        __m256i qw = _mm256_insertf128_si256(_mm256_castsi128_si256(raw), _mm_srli_epi16(raw, 4), 1);
        qw = _mm256_sub_epi8(_mm256_and_si256(qw, low4), eight);   // nibble - 8  -> [-8, 7]
        const __m256i qx = _mm256_loadu_si256((const __m256i*)x[b].qs);
        acc = _mm256_fmadd_ps(d, mul_sum_i8_pairs_float(qw, qx), acc);
    }
    return hsum256(acc);
#else
    float total = 0.0f;
    for (int b = 0; b < nb; ++b) {
        int s = 0;
        for (int j = 0; j < QK / 2; ++j) {
            s += ((w[b].qs[j] & 0x0F) - 8) * (int)x[b].qs[j];
            s += ((w[b].qs[j] >> 4)   - 8) * (int)x[b].qs[j + QK / 2];
        }
        total += (float)s * fp16_to_fp32(w[b].d) * fp16_to_fp32(x[b].d);
    }
    return total;
#endif
}

// ------------------------------------------------------------------
// K-quants (Q4_K, Q5_K, Q6_K): "super-blocks" of 256 weights.
// Popular downloads (Q4_K_M, Q5_K_M, and even "Q4_0" files whose
// output.weight is stored as Q6_K) use these. Their layout is more
// intricate (sub-block scales are themselves quantized), so we take
// the simple route: dequantize one row into a scratch buffer, then do
// a float dot product. Byte layouts copied from ggml-common.h.
// ------------------------------------------------------------------
constexpr int QK_K = 256;

#pragma pack(push, 1)
struct BlockQ4_K {
    uint16_t d, dmin;        // super-block scale for scales / for mins
    uint8_t  scales[12];     // 8 sub-blocks: 6-bit scale + 6-bit min, packed
    uint8_t  qs[QK_K / 2];   // 4-bit quants
};
struct BlockQ5_K {
    uint16_t d, dmin;
    uint8_t  scales[12];
    uint8_t  qh[QK_K / 8];   // 5th bit of each quant
    uint8_t  qs[QK_K / 2];   // low 4 bits
};
struct BlockQ6_K {
    uint8_t  ql[QK_K / 2];   // low 4 bits
    uint8_t  qh[QK_K / 4];   // high 2 bits
    int8_t   scales[QK_K / 16]; // 16 sub-blocks, 8-bit scales
    uint16_t d;              // super-block scale
};
#pragma pack(pop)
static_assert(sizeof(BlockQ4_K) == 144, "BlockQ4_K must match ggml");
static_assert(sizeof(BlockQ5_K) == 176, "BlockQ5_K must match ggml");
static_assert(sizeof(BlockQ6_K) == 210, "BlockQ6_K must match ggml");

// Unpack the j-th 6-bit (scale, min) pair from the 12-byte `scales` field.
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}

inline void dequantize_row_q4_K(const BlockQ4_K* x, float* y, int n) {
    for (int b = 0; b < n / QK_K; ++b) {
        const float d = fp16_to_fp32(x[b].d), min = fp16_to_fp32(x[b].dmin);
        const uint8_t* q = x[b].qs;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is, x[b].scales, sc, m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[b].scales, sc, m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

inline void dequantize_row_q5_K(const BlockQ5_K* x, float* y, int n) {
    for (int b = 0; b < n / QK_K; ++b) {
        const float d = fp16_to_fp32(x[b].d), min = fp16_to_fp32(x[b].dmin);
        const uint8_t* ql = x[b].qs;
        const uint8_t* qh = x[b].qh;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is, x[b].scales, sc, m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[b].scales, sc, m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4)  + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    }
}

inline void dequantize_row_q6_K(const BlockQ6_K* x, float* y, int n) {
    for (int b = 0; b < n / QK_K; ++b) {
        const float d = fp16_to_fp32(x[b].d);
        const uint8_t* ql = x[b].ql;
        const uint8_t* qh = x[b].qh;
        const int8_t*  sc = x[b].scales;
        for (int half = 0; half < QK_K; half += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l]      = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

} // namespace tinyml
