#pragma once
#include <complex>
#include <cstddef>
#include <vector>

namespace hzl {
using Complex = std::complex<double>;
using Field = std::vector<Complex>;
// Normalized Fourier coefficients: physical(x) = sum_k coefficient(k) exp(ikx).
class Fourier2D {
  public:
    explicit Fourier2D(std::size_t n);
    [[nodiscard]] std::size_t size() const noexcept {
        return n_;
    }
    void forward(Field &data);
    void inverse(Field &data);

  private:
    void transform(Field &data, bool inverse);
    void line(bool inverse);
    std::size_t n_;
    std::vector<std::size_t> reversed_;
    Field roots_, scratch_;
};
[[nodiscard]] int wave_number(std::size_t index, std::size_t n);
} // namespace hzl
