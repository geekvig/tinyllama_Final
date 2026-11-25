#pragma once
// =====================================================================
//  sampling.hpp  --  logits -> next token id
// =====================================================================
//  The model outputs one score ("logit") per vocabulary entry. Sampling
//  decides which token to actually emit:
//
//    repetition penalty -> temperature -> top-k -> top-p -> random draw
//
//  temperature 0 means "always take the highest logit" (greedy), which
//  is deterministic and what the tests use.
//
//  Design doc: Chapter 4.8.
// =====================================================================

#include "tensor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

namespace tinyml {

struct SamplingConfig {
    float temperature = 0.8f;   // 0 = greedy
    int   top_k = 40;           // 0 = disabled
    float top_p = 0.95f;        // 1 = disabled
    float repeat_penalty = 1.1f; // 1 = disabled
    int   repeat_last_n = 64;   // how many recent tokens the penalty looks at
};

inline int argmax(const float* v, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (v[i] > v[best]) best = i;
    return best;
}

inline int sample_token(const Tensor& logits_t, const SamplingConfig& cfg, std::mt19937& rng,
                        const std::vector<int>& history = {}) {
    const int n = (int)logits_t.numel();
    std::vector<float> logits(logits_t.data(), logits_t.data() + n);

    // Repetition penalty (llama.cpp style): make recently used tokens less likely.
    if (cfg.repeat_penalty != 1.0f && !history.empty()) {
        std::unordered_set<int> recent;
        int start = std::max(0, (int)history.size() - cfg.repeat_last_n);
        for (int i = start; i < (int)history.size(); ++i) recent.insert(history[i]);
        for (int id : recent) {
            if (id < 0 || id >= n) continue;
            logits[id] = logits[id] > 0 ? logits[id] / cfg.repeat_penalty : logits[id] * cfg.repeat_penalty;
        }
    }

    if (cfg.temperature <= 0.0f) return argmax(logits.data(), n);

    // Rank token ids by logit (descending). partial_sort is enough for top-k.
    int keep = (cfg.top_k > 0) ? std::min(cfg.top_k, n) : n;
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });

    // Softmax over the kept candidates, with temperature.
    std::vector<float> probs(keep);
    const float maxl = logits[order[0]];
    float sum = 0.0f;
    for (int i = 0; i < keep; ++i) { probs[i] = std::exp((logits[order[i]] - maxl) / cfg.temperature); sum += probs[i]; }
    for (float& p : probs) p /= sum;

    // Top-p (nucleus): smallest prefix whose probability mass reaches top_p.
    if (cfg.top_p < 1.0f) {
        float cum = 0.0f;
        int cutoff = keep;
        for (int i = 0; i < keep; ++i) { cum += probs[i]; if (cum >= cfg.top_p) { cutoff = i + 1; break; } }
        probs.resize(cutoff);   // discrete_distribution renormalises for us
    }

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return order[dist(rng)];
}

} // namespace tinyml
