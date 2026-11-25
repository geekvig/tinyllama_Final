#pragma once
// =====================================================================
//  tensor.hpp  --  the simplest possible n-dimensional array
// =====================================================================
//  A Tensor is just:   a flat std::vector<float>   +   a shape.
//  Everything is ROW-MAJOR: in a [rows, cols] tensor, element (i, j)
//  lives at data[i * cols + j]. "Strides" tell us how far to jump in
//  the flat buffer when an index goes up by one.
//
//  Design doc: Chapter 5.1 ("Tensors: numbers in a row").
// =====================================================================

#include <vector>
#include <cstddef>
#include <sstream>
#include <string>
#include <stdexcept>

namespace tinyml {

class Tensor {
public:
    Tensor() = default;

    // Zero-filled tensor of the given shape.
    explicit Tensor(std::vector<int> shape) : shape_(std::move(shape)) {
        recompute_strides();
        data_.assign(numel(), 0.0f);
    }

    // Tensor that takes ownership of existing data (must match the shape).
    Tensor(std::vector<int> shape, std::vector<float> data)
        : data_(std::move(data)), shape_(std::move(shape)) {
        recompute_strides();
        if (data_.size() != numel()) {
            throw std::runtime_error("Tensor: data size does not match shape " + to_string());
        }
    }

    size_t numel() const {
        size_t n = 1;
        for (int s : shape_) n *= (size_t)s;
        return shape_.empty() ? 0 : n;
    }

    // ---- shape ----
    const std::vector<int>& shape() const { return shape_; }
    int ndim() const { return (int)shape_.size(); }
    int dim(int i) const { return shape_.at(i); }   // .at() = bounds-checked
    bool empty() const { return data_.empty(); }

    // ---- raw data ----
    float*       data()       { return data_.data(); }
    const float* data() const { return data_.data(); }
    float*       row(int i)       { return data_.data() + (size_t)i * strides_[0]; }
    const float* row(int i) const { return data_.data() + (size_t)i * strides_[0]; }

    // ---- element access ----
    float& at_flat(size_t idx)       { return data_[idx]; }
    float  at_flat(size_t idx) const { return data_[idx]; }
    float& at(int i, int j)       { return data_[(size_t)i * strides_[0] + j]; }
    float  at(int i, int j) const { return data_[(size_t)i * strides_[0] + j]; }

    std::string to_string() const {
        std::ostringstream os;
        os << "Tensor(shape=[";
        for (size_t i = 0; i < shape_.size(); ++i) os << shape_[i] << (i + 1 < shape_.size() ? "," : "");
        os << "])";
        return os.str();
    }

private:
    std::vector<float> data_;
    std::vector<int>   shape_;
    std::vector<size_t> strides_;

    void recompute_strides() {
        strides_.assign(shape_.size(), 1);
        for (int i = (int)shape_.size() - 2; i >= 0; --i)
            strides_[i] = strides_[i + 1] * (size_t)shape_[i + 1];
    }
};

} // namespace tinyml
