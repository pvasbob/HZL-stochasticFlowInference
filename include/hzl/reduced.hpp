#pragma once
#include "hzl/inference.hpp"

namespace hzl {
double spectral_inner(const Field &a, const Field &b);
struct PODBasis {
    Field mean;
    std::vector<Field> modes;
    Vector singular_values;
    double captured_variance = 0;
    [[nodiscard]] Vector project(const Field &state) const;
    [[nodiscard]] Field reconstruct(const Vector &coefficients) const;
};
// Snapshot covariance eigendecomposition. Intended for modest training sets;
// production large-scale SVD backends remain a later optimization.
PODBasis fit_pod(const std::vector<Field> &snapshots, std::size_t maximum_rank,
                 double variance_fraction = 0.9999);
class GalerkinModel {
  public:
    GalerkinModel(PODBasis basis, Solver &full_model);
    [[nodiscard]] Vector rhs(const Vector &coefficients) const;
    void step(Vector &coefficients, double dt) const;
    [[nodiscard]] const PODBasis &basis() const noexcept {
        return basis_;
    }

  private:
    PODBasis basis_;
    Vector constant_, linear_, quadratic_;
};
} // namespace hzl
