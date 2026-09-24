#pragma once
#include "hzl/inference.hpp"

namespace hzl {
class QuadraticSurrogate {
  public:
    QuadraticSurrogate(const std::vector<Vector> &inputs, const Vector &outputs, Vector center,
                       Vector scales);
    [[nodiscard]] double operator()(const Vector &input) const;
    [[nodiscard]] const Vector &coefficients() const noexcept {
        return coefficients_;
    }
    [[nodiscard]] double training_rmse() const noexcept {
        return training_rmse_;
    }

  private:
    [[nodiscard]] Vector features(const Vector &input) const;
    Vector center_, scales_, coefficients_;
    double training_rmse_ = 0;
};
class GaussianRBFSurrogate {
  public:
    GaussianRBFSurrogate(std::vector<Vector> inputs, Vector outputs, Vector center, Vector scales,
                         double shape = 1.0, double ridge = 1e-10);
    [[nodiscard]] double operator()(const Vector &input) const;
    [[nodiscard]] double training_rmse() const noexcept {
        return training_rmse_;
    }

  private:
    [[nodiscard]] Vector normalize(const Vector &input) const;
    std::vector<Vector> centers_;
    Vector center_, scales_, weights_;
    double shape_ = 1, training_rmse_ = 0;
};
struct DelayedAcceptanceResult {
    std::vector<Sample> samples;
    std::size_t stage_one_acceptances = 0, exact_evaluations = 0;
};
DelayedAcceptanceResult delayed_acceptance(const Objective &exact_log_density,
                                           const Objective &approximate_log_density, Vector initial,
                                           const Vector &proposal_scales, std::size_t count,
                                           std::uint64_t seed);
} // namespace hzl
