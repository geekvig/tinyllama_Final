#pragma once
// =====================================================================
//  fp16.hpp  --  IEEE-754 half precision (16-bit float) <-> float
// =====================================================================
//  GGUF files store many things as fp16: whole F16 tensors, and the
//  per-block scale "d" inside Q8_0 / Q4_0 blocks. C++17 has no half
//  type, so we decode the bits by hand.
//
//   bit 15      bits 14..10      bits 9..0
//   [sign]      [exponent (5)]   [mantissa (10)]
//
//   value = (-1)^sign * 2^(exponent-15) * 1.mantissa     (normal numbers)
//
//  Design doc: Chapter 3.6 ("How numbers are stored").
// =====================================================================

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#if defined(__F16C__)
#include <immintrin.h>
#endif

namespace tinyml {

// The careful, portable bit-twiddling version (the "teaching" version).
inline float fp16_to_fp32_slow(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                                   // +/- 0
        } else {                                           // subnormal: normalise it
            int e = -1;
            do { mant <<= 1; ++e; } while ((mant & 0x400) == 0);
            mant &= 0x3FF;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);          // inf / NaN
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));                     // safe type-pun
    return f;
}

// The FAST version used everywhere. It is called twice per 32 weights in
// the hot loop, so the branches above would cost ~20% of total runtime.
//  * CPUs with F16C (almost every x86 CPU since 2012) convert in hardware.
//  * Otherwise: a 65536-entry lookup table (256 KB), built once. ggml does
//    the same (ggml_table_f32_f16).
inline float fp16_to_fp32(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    static const std::vector<float> table = [] {
        std::vector<float> t(65536);
        for (uint32_t i = 0; i < 65536; ++i) t[i] = fp16_to_fp32_slow((uint16_t)i);
        return t;
    }();
    return table[h];
#endif
}

// float -> fp16 with round-to-nearest-even (used by tests and quantizers).
inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(sign | 0x7C00 | (mant ? 0x200 : 0));
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00);     // overflow -> inf
    if (exp <= 0) {                                        // subnormal / zero
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = mant >> shift;
        uint32_t rem  = mant & ((1u << shift) - 1);
        uint32_t mid  = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1))) ++half;
        return (uint16_t)(sign | half);
    }
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    uint32_t rem = mant & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) ++half; // may carry into exp: correct
    return (uint16_t)half;
}

} // namespace tinyml
