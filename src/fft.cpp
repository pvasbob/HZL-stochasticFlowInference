#include "hzl/fft.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace hzl {
Fourier2D::Fourier2D(std::size_t n) : n_(n) {
    if (n < 8 || n > 4096 || !std::has_single_bit(n))
        throw std::invalid_argument("grid must be a power of two between 8 and 4096");
    reversed_.resize(n);
    roots_.resize(n / 2);
    scratch_.resize(n);
    const auto bits = std::countr_zero(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto x = i;
        std::size_t r = 0;
        for (int b = 0; b < bits; ++b) {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        reversed_[i] = r;
    }
    for (std::size_t i = 0; i < n / 2; ++i)
        roots_[i] = std::polar(1.0, -2.0 * std::numbers::pi * static_cast<double>(i) /
                                        static_cast<double>(n));
}
void Fourier2D::line(bool inverse) {
    for (std::size_t i = 0; i < n_; ++i)
        if (i < reversed_[i])
            std::swap(scratch_[i], scratch_[reversed_[i]]);
    for (std::size_t length = 2; length <= n_; length *= 2) {
        const auto half = length / 2, stride = n_ / length;
        for (std::size_t start = 0; start < n_; start += length)
            for (std::size_t j = 0; j < half; ++j) {
                const auto root = inverse ? std::conj(roots_[j * stride]) : roots_[j * stride];
                const auto a = scratch_[start + j], b = root * scratch_[start + j + half];
                scratch_[start + j] = a + b;
                scratch_[start + j + half] = a - b;
            }
    }
}
void Fourier2D::transform(Field &data, bool inverse) {
    if (data.size() != n_ * n_)
        throw std::invalid_argument("FFT field size mismatch");
    for (std::size_t y = 0; y < n_; ++y) {
        std::copy_n(data.begin() + static_cast<std::ptrdiff_t>(y * n_), n_, scratch_.begin());
        line(inverse);
        std::copy(scratch_.begin(), scratch_.end(),
                  data.begin() + static_cast<std::ptrdiff_t>(y * n_));
    }
    for (std::size_t x = 0; x < n_; ++x) {
        for (std::size_t y = 0; y < n_; ++y)
            scratch_[y] = data[y * n_ + x];
        line(inverse);
        for (std::size_t y = 0; y < n_; ++y)
            data[y * n_ + x] = scratch_[y];
    }
    if (!inverse)
        for (auto &z : data)
            z /= static_cast<double>(n_ * n_);
}
void Fourier2D::forward(Field &data) {
    transform(data, false);
}
void Fourier2D::inverse(Field &data) {
    transform(data, true);
}
int wave_number(std::size_t index, std::size_t n) {
    return index < n / 2 ? static_cast<int>(index) : static_cast<int>(index) - static_cast<int>(n);
}
} // namespace hzl
