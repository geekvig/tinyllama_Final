#pragma once
// =====================================================================
//  gguf.hpp  --  a from-scratch reader for the GGUF file format
// =====================================================================
//  GGUF ("GGML Universal File") is the single-file model format used by
//  llama.cpp. One file holds EVERYTHING needed to run a model:
//  hyper-parameters, the tokenizer vocabulary, and every weight tensor.
//
//  On-disk layout (all integers little-endian):
//
//   +--------------------------------------------------------------+
//   | magic        "GGUF"  (4 bytes)                                |
//   | version      uint32  (we accept 2 and 3)                      |
//   | tensor_count uint64                                           |
//   | kv_count     uint64                                           |
//   +--------------------------------------------------------------+
//   | kv_count x METADATA entry:                                    |
//   |    key   : gguf_string  (uint64 length + bytes, no '\0')      |
//   |    type  : uint32  (GGUFType below)                           |
//   |    value : depends on type (arrays: elem_type, count, items)  |
//   +--------------------------------------------------------------+
//   | tensor_count x TENSOR INFO:                                   |
//   |    name    : gguf_string                                      |
//   |    n_dims  : uint32                                           |
//   |    ne[i]   : uint64 x n_dims   (ne[0] = fastest-moving dim!)  |
//   |    type    : uint32  (GGMLType below)                         |
//   |    offset  : uint64  (relative to start of the data section)  |
//   +--------------------------------------------------------------+
//   | padding up to a multiple of general.alignment (default 32)    |
//   +--------------------------------------------------------------+
//   | DATA SECTION: raw tensor bytes, each tensor aligned           |
//   +--------------------------------------------------------------+
//
//  We read the whole file into one byte buffer and hand out POINTERS
//  into it ("zero-copy"). llama.cpp does the same with mmap().
//  Consequence: the GGUFFile object must outlive every pointer that
//  points into it (Bug Lab 6 is what happens when it doesn't).
//
//  Design doc: Chapter 3 (the whole chapter is about this file).
// =====================================================================

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tinyml {

// Metadata value types (gguf.h: enum gguf_type)
enum class GGUFType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5,
    FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12,
};

// Tensor element types (ggml.h: enum ggml_type). We implement the first
// F32/F16/BF16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K; the rest are listed so error
// messages can name them.
enum class GGMLType : uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15, BF16 = 30,
};

inline const char* ggml_type_name(GGMLType t) {
    switch (t) {
        case GGMLType::F32: return "F32";   case GGMLType::F16: return "F16";
        case GGMLType::Q4_0: return "Q4_0"; case GGMLType::Q4_1: return "Q4_1";
        case GGMLType::Q5_0: return "Q5_0"; case GGMLType::Q5_1: return "Q5_1";
        case GGMLType::Q8_0: return "Q8_0"; case GGMLType::Q8_1: return "Q8_1";
        case GGMLType::Q2_K: return "Q2_K"; case GGMLType::Q3_K: return "Q3_K";
        case GGMLType::Q4_K: return "Q4_K"; case GGMLType::Q5_K: return "Q5_K";
        case GGMLType::Q6_K: return "Q6_K"; case GGMLType::Q8_K: return "Q8_K";
        case GGMLType::BF16: return "BF16";
    }
    return "UNKNOWN";
}

// Bytes needed to store `n_elements` values of a given type.
// Block types store 32 elements per block.
inline uint64_t ggml_nbytes(GGMLType t, uint64_t n_elements) {
    switch (t) {
        case GGMLType::F32:  return n_elements * 4;
        case GGMLType::F16:  return n_elements * 2;
        case GGMLType::BF16: return n_elements * 2;
        case GGMLType::Q8_0: return n_elements / 32 * 34;
        case GGMLType::Q4_0: return n_elements / 32 * 18;
        case GGMLType::Q4_1: return n_elements / 32 * 20;
        case GGMLType::Q5_0: return n_elements / 32 * 22;
        case GGMLType::Q5_1: return n_elements / 32 * 24;
        case GGMLType::Q4_K: return n_elements / 256 * 144;
        case GGMLType::Q5_K: return n_elements / 256 * 176;
        case GGMLType::Q6_K: return n_elements / 256 * 210;
        default: return 0; // unknown to us -> caller must refuse
    }
}

// One metadata value. A small tagged union: `type` says which field is live.
struct GGUFValue {
    GGUFType type = GGUFType::UINT8;
    uint64_t u = 0;          // all unsigned ints + bool
    int64_t  i = 0;          // all signed ints
    double   f = 0.0;        // float32 / float64
    std::string str;
    GGUFType arr_type = GGUFType::UINT8;
    std::vector<GGUFValue> arr;
};

struct GGUFTensorInfo {
    std::string name;
    std::vector<uint64_t> ne;   // GGML order: ne[0] is the contiguous (inner) dim
    GGMLType type;
    uint64_t offset;            // relative to data section
    const uint8_t* data = nullptr; // absolute pointer, filled in after parsing
    uint64_t nbytes = 0;

    uint64_t n_elements() const { uint64_t n = 1; for (auto d : ne) n *= d; return n; }
};

class GGUFFile {
public:
    uint32_t version = 0;
    uint64_t alignment = 32;
    std::map<std::string, GGUFValue> kv;          // sorted -> nice for printing
    std::vector<GGUFTensorInfo> tensors;
    std::map<std::string, size_t> tensor_index;   // name -> index in tensors

    static std::shared_ptr<GGUFFile> open(const std::string& path) {
        auto g = std::make_shared<GGUFFile>();
        g->read_file(path);
        g->parse();
        return g;
    }

    // ---- metadata helpers ------------------------------------------
    bool has(const std::string& key) const { return kv.count(key) != 0; }

    const GGUFValue& get(const std::string& key) const {
        auto it = kv.find(key);
        if (it == kv.end()) throw std::runtime_error("GGUF: missing metadata key '" + key + "'");
        return it->second;
    }
    int64_t get_int(const std::string& key) const {
        const GGUFValue& v = get(key);
        switch (v.type) {
            case GGUFType::UINT8: case GGUFType::UINT16: case GGUFType::UINT32:
            case GGUFType::UINT64: case GGUFType::BOOL: return (int64_t)v.u;
            case GGUFType::INT8: case GGUFType::INT16: case GGUFType::INT32:
            case GGUFType::INT64: return v.i;
            default: throw std::runtime_error("GGUF: key '" + key + "' is not an integer");
        }
    }
    int64_t get_int(const std::string& key, int64_t def) const { return has(key) ? get_int(key) : def; }
    double get_float(const std::string& key, double def) const {
        if (!has(key)) return def;
        const GGUFValue& v = get(key);
        if (v.type == GGUFType::FLOAT32 || v.type == GGUFType::FLOAT64) return v.f;
        return (double)get_int(key);
    }
    std::string get_str(const std::string& key, const std::string& def = "") const {
        if (!has(key)) return def;
        const GGUFValue& v = get(key);
        if (v.type != GGUFType::STRING) throw std::runtime_error("GGUF: key '" + key + "' is not a string");
        return v.str;
    }
    bool get_bool(const std::string& key, bool def) const { return has(key) ? get(key).u != 0 : def; }

    // ---- tensor helpers --------------------------------------------
    const GGUFTensorInfo* find_tensor(const std::string& name) const {
        auto it = tensor_index.find(name);
        return it == tensor_index.end() ? nullptr : &tensors[it->second];
    }
    const GGUFTensorInfo& tensor(const std::string& name) const {
        const GGUFTensorInfo* t = find_tensor(name);
        if (!t) throw std::runtime_error("GGUF: missing tensor '" + name + "'");
        return *t;
    }

    size_t file_size() const { return bytes_.size(); }
    const uint8_t* base() const { return bytes_.data(); }

private:
    std::vector<uint8_t> bytes_;   // the ENTIRE file
    size_t pos_ = 0;               // read cursor while parsing

    void read_file(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("GGUF: cannot open '" + path + "'");
        std::streamoff size = f.tellg();
        if (size < 24) throw std::runtime_error("GGUF: file too small to be GGUF: '" + path + "'");
        bytes_.resize((size_t)size);
        f.seekg(0);
        f.read(reinterpret_cast<char*>(bytes_.data()), size);
        if (!f) throw std::runtime_error("GGUF: short read on '" + path + "'");
    }

    // ---- primitive readers: every read is bounds-checked ------------
    void need(size_t n) const {
        if (pos_ + n > bytes_.size())
            throw std::runtime_error("GGUF: unexpected end of file at byte " + std::to_string(pos_) +
                                     " (wanted " + std::to_string(n) + " more bytes)");
    }
    template <typename T> T rd() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, bytes_.data() + pos_, sizeof(T)); // memcpy: no alignment UB
        pos_ += sizeof(T);
        return v;
    }
    std::string rd_str() {
        uint64_t len = rd<uint64_t>();       // GGUF v2+: 64-bit length (Bug Lab 2!)
        need(len);
        std::string s(reinterpret_cast<const char*>(bytes_.data() + pos_), (size_t)len);
        pos_ += len;
        return s;
    }

    GGUFValue rd_value(GGUFType t) {
        GGUFValue v;
        v.type = t;
        switch (t) {
            case GGUFType::UINT8:   v.u = rd<uint8_t>();  break;
            case GGUFType::INT8:    v.i = rd<int8_t>();   break;
            case GGUFType::UINT16:  v.u = rd<uint16_t>(); break;
            case GGUFType::INT16:   v.i = rd<int16_t>();  break;
            case GGUFType::UINT32:  v.u = rd<uint32_t>(); break;
            case GGUFType::INT32:   v.i = rd<int32_t>();  break;
            case GGUFType::UINT64:  v.u = rd<uint64_t>(); break;
            case GGUFType::INT64:   v.i = rd<int64_t>();  break;
            case GGUFType::FLOAT32: v.f = rd<float>();    break;
            case GGUFType::FLOAT64: v.f = rd<double>();   break;
            case GGUFType::BOOL:    v.u = rd<uint8_t>();  break;
            case GGUFType::STRING:  v.str = rd_str();     break;
            case GGUFType::ARRAY: {
                v.arr_type = (GGUFType)rd<uint32_t>();
                uint64_t n = rd<uint64_t>();
                if (n > bytes_.size()) throw std::runtime_error("GGUF: absurd array length");
                v.arr.reserve((size_t)n);
                for (uint64_t k = 0; k < n; ++k) v.arr.push_back(rd_value(v.arr_type));
                break;
            }
            default:
                throw std::runtime_error("GGUF: unknown metadata type " + std::to_string((uint32_t)t) +
                                         " at byte " + std::to_string(pos_));
        }
        return v;
    }

    void parse() {
        // 1) header
        uint32_t magic = rd<uint32_t>();
        if (magic != 0x46554747u) // bytes 'G','G','U','F' read as little-endian uint32
            throw std::runtime_error("GGUF: bad magic (not a GGUF file)");
        version = rd<uint32_t>();
        if (version < 2 || version > 3)
            throw std::runtime_error("GGUF: unsupported version " + std::to_string(version));
        uint64_t n_tensors = rd<uint64_t>();
        uint64_t n_kv = rd<uint64_t>();

        // 2) metadata key/value pairs
        for (uint64_t k = 0; k < n_kv; ++k) {
            std::string key = rd_str();
            GGUFType t = (GGUFType)rd<uint32_t>();
            kv[key] = rd_value(t);
        }
        alignment = (uint64_t)get_int("general.alignment", 32);
        if (alignment == 0 || (alignment & (alignment - 1)))
            throw std::runtime_error("GGUF: alignment must be a power of two");

        // 3) tensor infos
        tensors.reserve((size_t)n_tensors);
        for (uint64_t k = 0; k < n_tensors; ++k) {
            GGUFTensorInfo ti;
            ti.name = rd_str();
            uint32_t n_dims = rd<uint32_t>();
            if (n_dims == 0 || n_dims > 4) throw std::runtime_error("GGUF: bad n_dims for " + ti.name);
            for (uint32_t d = 0; d < n_dims; ++d) ti.ne.push_back(rd<uint64_t>());
            ti.type = (GGMLType)rd<uint32_t>();
            ti.offset = rd<uint64_t>();
            tensor_index[ti.name] = tensors.size();
            tensors.push_back(std::move(ti));
        }

        // 4) the data section starts at the next multiple of `alignment`
        size_t data_start = (pos_ + alignment - 1) / alignment * alignment;

        // 5) turn relative offsets into pointers, and sanity-check sizes
        for (auto& t : tensors) {
            t.nbytes = ggml_nbytes(t.type, t.n_elements());
            t.data = bytes_.data() + data_start + t.offset;
            if (t.nbytes && data_start + t.offset + t.nbytes > bytes_.size())
                throw std::runtime_error("GGUF: tensor '" + t.name + "' runs past end of file");
        }
    }
};

} // namespace tinyml
