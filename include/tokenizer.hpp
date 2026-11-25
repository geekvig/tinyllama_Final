#pragma once
// =====================================================================
//  tokenizer.hpp  --  SentencePiece-style (SPM) tokenizer read from GGUF
// =====================================================================
//  The neural network only understands integers ("token ids"). The
//  tokenizer turns text into ids and back.
//
//  Llama 1/2 and TinyLlama use a SentencePiece BPE vocabulary. The GGUF
//  file stores it as three parallel arrays:
//      tokenizer.ggml.tokens      ["<unk>", "<s>", "</s>", "<0x00>", ..., "▁the", ...]
//      tokenizer.ggml.scores      [0.0, 0.0, 0.0, ..., -2.5, ...]   (higher = merge first)
//      tokenizer.ggml.token_type  [2, 3, 3, 6, ..., 1, ...]          (NORMAL=1, BYTE=6 ...)
//
//  Encoding algorithm (same as llama.cpp's llm_tokenizer_spm):
//    1. Put a space in front ("add_space_prefix"), then replace every
//       ' ' with "▁" (U+2581, 3 bytes in UTF-8).
//    2. Split into single UTF-8 characters -> the initial "symbols".
//    3. Repeat: among all ADJACENT symbol pairs whose concatenation is
//       a vocab token, merge the one with the HIGHEST score (leftmost
//       on ties). Stop when no pair can merge.
//    4. Each final symbol -> its id. A symbol that is not in the vocab
//       (e.g. an emoji) falls back to one "<0xXX>" token per byte.
//    5. Put BOS (id 1, "<s>") in front.
//  Before all that, special tokens typed in the text ("</s>") are cut
//  out and emitted as single ids, and steps 1-4 run on each piece between.
//
//  Decoding: concatenate pieces, "▁" -> ' ', "<0xXX>" -> that raw byte,
//  control tokens (<s>, </s>) -> nothing.
//
//  Design doc: Chapter 4.1 and 5.7.
// =====================================================================

#include "gguf.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace tinyml {

class Tokenizer {
public:
    enum TokenType { NORMAL = 1, UNKNOWN = 2, CONTROL = 3, USER_DEFINED = 4, UNUSED = 5, BYTE = 6 };

    int bos_id = 1, eos_id = 2, unk_id = 0;
    bool add_bos = true;
    bool add_space_prefix = true;

    static Tokenizer from_gguf(const GGUFFile& g) {
        Tokenizer tk;
        const std::string model = g.get_str("tokenizer.ggml.model", "llama");
        if (model != "llama")
            throw std::runtime_error("tokenizer model '" + model + "' is not supported "
                                     "(tinyllama implements the SentencePiece 'llama' tokenizer used by "
                                     "Llama 1/2 and TinyLlama; Llama 3 uses 'gpt2' BPE)");
        const GGUFValue& toks = g.get("tokenizer.ggml.tokens");
        const size_t n = toks.arr.size();
        tk.pieces_.resize(n);
        tk.scores_.assign(n, 0.0f);
        tk.types_.assign(n, NORMAL);
        for (size_t i = 0; i < n; ++i) {
            tk.pieces_[i] = toks.arr[i].str;
            tk.piece_to_id_.emplace(tk.pieces_[i], (int)i);   // first occurrence wins
        }
        if (g.has("tokenizer.ggml.scores")) {
            const auto& s = g.get("tokenizer.ggml.scores").arr;
            for (size_t i = 0; i < n && i < s.size(); ++i) tk.scores_[i] = (float)s[i].f;
        }
        if (g.has("tokenizer.ggml.token_type")) {
            const auto& t = g.get("tokenizer.ggml.token_type").arr;
            for (size_t i = 0; i < n && i < t.size(); ++i) tk.types_[i] = (int)t[i].i;
        }
        tk.bos_id = (int)g.get_int("tokenizer.ggml.bos_token_id", 1);
        tk.eos_id = (int)g.get_int("tokenizer.ggml.eos_token_id", 2);
        tk.unk_id = (int)g.get_int("tokenizer.ggml.unknown_token_id", 0);
        tk.add_bos = g.get_bool("tokenizer.ggml.add_bos_token", true);
        tk.add_space_prefix = g.get_bool("tokenizer.ggml.add_space_prefix", true);
        for (size_t i = 0; i < n; ++i)
            if ((tk.types_[i] == CONTROL || tk.types_[i] == USER_DEFINED) && !tk.pieces_[i].empty())
                tk.special_ids_.push_back((int)i);
        return tk;
    }

    int vocab_size() const { return (int)pieces_.size(); }
    const std::string& piece(int id) const { return pieces_.at(id); }

    // Encode text. With parse_special = true, control tokens written in
    // the text (like "</s>" in a chat template) become their single id
    // instead of being spelled out letter by letter -- same as llama.cpp.
    std::vector<int> encode(const std::string& text, bool with_bos = true,
                            bool parse_special = true) const {
        std::vector<int> ids;
        if (with_bos && add_bos) ids.push_back(bos_id);
        // Split the text into [plain text] [special] [plain text] ...
        // A space prefix is added to every plain segment that starts the
        // text or follows a special token (llama.cpp behaviour, verified).
        size_t i = 0, seg_start = 0;
        auto flush = [&](size_t end) {
            if (end > seg_start) encode_plain(text.substr(seg_start, end - seg_start), ids);
        };
        while (parse_special && i < text.size()) {
            int best = -1; size_t best_len = 0;
            for (int id : special_ids_) {
                const std::string& sp = pieces_[id];
                if (sp.size() > best_len && text.compare(i, sp.size(), sp) == 0) { best = id; best_len = sp.size(); }
            }
            if (best < 0) { ++i; continue; }
            flush(i);
            ids.push_back(best);
            i += best_len;
            seg_start = i;
        }
        flush(text.size());
        return ids;
    }

    // Text for a single token (used for streaming output).
    std::string decode_piece(int id) const {
        if (id < 0 || id >= vocab_size()) return "";
        const int type = types_[id];
        const std::string& p = pieces_[id];
        if (type == CONTROL || type == UNUSED) return "";
        if (type == BYTE && p.size() == 6 && p.rfind("<0x", 0) == 0)
            return std::string(1, (char)std::stoi(p.substr(3, 2), nullptr, 16));
        std::string out;
        for (size_t i = 0; i < p.size(); ++i) {
            if (i + 2 < p.size() && (unsigned char)p[i] == 0xE2 && (unsigned char)p[i + 1] == 0x96 &&
                (unsigned char)p[i + 2] == 0x81) { out += ' '; i += 2; }
            else out += p[i];
        }
        return out;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string out;
        for (int id : ids) out += decode_piece(id);
        if (add_space_prefix && !out.empty() && out[0] == ' ') out.erase(0, 1);
        return out;
    }

private:
    std::vector<std::string> pieces_;
    std::vector<int> special_ids_;   // CONTROL / USER_DEFINED tokens, e.g. "<s>", "</s>"

    // The SentencePiece algorithm on one plain-text segment (steps 1-4).
    void encode_plain(const std::string& text, std::vector<int>& ids) const {
        // 1. whitespace escaping
        std::string s = add_space_prefix ? " " + text : text;
        std::string esc;
        for (char ch : s) { if (ch == ' ') esc += "\xE2\x96\x81"; else esc += ch; } // "▁"

        // 2. split into UTF-8 characters
        std::vector<std::string> sym;
        for (size_t i = 0; i < esc.size();) {
            size_t len = utf8_len((unsigned char)esc[i]);
            if (i + len > esc.size()) len = 1;      // broken UTF-8: take 1 byte
            sym.push_back(esc.substr(i, len));
            i += len;
        }

        // 3. greedy highest-score merging
        while (sym.size() > 1) {
            float best_score = -1e30f;
            int best_i = -1;
            for (size_t i = 0; i + 1 < sym.size(); ++i) {
                auto it = piece_to_id_.find(sym[i] + sym[i + 1]);
                if (it == piece_to_id_.end()) continue;
                float sc = scores_[it->second];
                if (sc > best_score) { best_score = sc; best_i = (int)i; } // '>' keeps leftmost on ties
            }
            if (best_i < 0) break;
            sym[best_i] += sym[best_i + 1];
            sym.erase(sym.begin() + best_i + 1);
        }

        // 4. symbols -> ids, with byte fallback
        for (const std::string& p : sym) {
            auto it = piece_to_id_.find(p);
            if (it != piece_to_id_.end()) { ids.push_back(it->second); continue; }
            for (unsigned char b : p) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
                auto bt = piece_to_id_.find(buf);
                ids.push_back(bt != piece_to_id_.end() ? bt->second : unk_id);
            }
        }
    }
    std::vector<float> scores_;
    std::vector<int> types_;
    std::unordered_map<std::string, int> piece_to_id_;

    static size_t utf8_len(unsigned char c) {
        if (c < 0x80) return 1;
        if ((c >> 5) == 0x6) return 2;
        if ((c >> 4) == 0xE) return 3;
        if ((c >> 3) == 0x1E) return 4;
        return 1;
    }
};

} // namespace tinyml
