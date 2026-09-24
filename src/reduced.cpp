#include "hzl/reduced.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace hzl {
double spectral_inner(const Field &a, const Field &b) {
    if (a.size() != b.size())
        throw std::invalid_argument("spectral inner product size mismatch");
    double value = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        value += std::real(std::conj(a[i]) * b[i]);
    return value;
}
Vector PODBasis::project(const Field &state) const {
    if (state.size() != mean.size())
        throw std::invalid_argument("POD state size mismatch");
    auto centered = state;
    for (std::size_t i = 0; i < mean.size(); ++i)
        centered[i] -= mean[i];
    Vector result(modes.size());
    for (std::size_t j = 0; j < modes.size(); ++j)
        result[j] = spectral_inner(centered, modes[j]);
    return result;
}
Field PODBasis::reconstruct(const Vector &coefficients) const {
    if (coefficients.size() != modes.size())
        throw std::invalid_argument("POD coefficient size mismatch");
    auto state = mean;
    for (std::size_t j = 0; j < modes.size(); ++j)
        for (std::size_t i = 0; i < state.size(); ++i)
            state[i] += coefficients[j] * modes[j][i];
    return state;
}
PODBasis fit_pod(const std::vector<Field> &snapshots, std::size_t maximum_rank, double fraction) {
    if (snapshots.size() < 2 || snapshots.size() > 512 || snapshots.front().empty() ||
        maximum_rank == 0 || !std::isfinite(fraction) || fraction <= 0 || fraction > 1)
        throw std::invalid_argument(
            "POD requires 2..512 snapshots, positive rank, and variance fraction in (0,1]");
    const auto m = snapshots.size(), dimension = snapshots.front().size();
    PODBasis result;
    result.mean.assign(dimension, 0);
    for (const auto &state : snapshots) {
        if (state.size() != dimension)
            throw std::invalid_argument("inconsistent snapshot dimensions");
        for (std::size_t i = 0; i < dimension; ++i) {
            if (!std::isfinite(state[i].real()) || !std::isfinite(state[i].imag()))
                throw std::invalid_argument("nonfinite snapshot");
            result.mean[i] += state[i] / static_cast<double>(m);
        }
    }
    auto centered = snapshots;
    for (auto &state : centered)
        for (std::size_t i = 0; i < dimension; ++i)
            state[i] -= result.mean[i];
    Vector covariance(m * m), vectors(m * m, 0);
    double trace = 0;
    for (std::size_t i = 0; i < m; ++i) {
        vectors[i * m + i] = 1;
        for (std::size_t j = 0; j <= i; ++j)
            covariance[i * m + j] = covariance[j * m + i] =
                spectral_inner(centered[i], centered[j]);
        trace += covariance[i * m + i];
    }
    if (trace <= 0 || !std::isfinite(trace))
        throw std::invalid_argument("snapshots have no finite variance");
    bool converged = false;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double maximum = 0;
        for (std::size_t p = 0; p < m; ++p)
            for (std::size_t q = p + 1; q < m; ++q) {
                const double off = covariance[p * m + q];
                maximum = std::max(maximum, std::abs(off));
                if (std::abs(off) < 1e-14 * trace)
                    continue;
                const double tau = (covariance[q * m + q] - covariance[p * m + p]) / (2 * off);
                const double t = std::copysign(1.0, tau) / (std::abs(tau) + std::hypot(1.0, tau)),
                             c = 1 / std::sqrt(1 + t * t), s = t * c;
                const double pp = covariance[p * m + p], qq = covariance[q * m + q];
                covariance[p * m + p] = pp - t * off;
                covariance[q * m + q] = qq + t * off;
                covariance[p * m + q] = covariance[q * m + p] = 0;
                for (std::size_t k = 0; k < m; ++k) {
                    if (k != p && k != q) {
                        const double a = covariance[k * m + p], b = covariance[k * m + q];
                        covariance[k * m + p] = covariance[p * m + k] = c * a - s * b;
                        covariance[k * m + q] = covariance[q * m + k] = s * a + c * b;
                    }
                    const double a = vectors[k * m + p], b = vectors[k * m + q];
                    vectors[k * m + p] = c * a - s * b;
                    vectors[k * m + q] = s * a + c * b;
                }
            }
        if (maximum < 1e-12 * trace) {
            converged = true;
            break;
        }
    }
    if (!converged)
        throw std::runtime_error("snapshot Jacobi eigensolver failed to converge");
    std::vector<std::size_t> order(m);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](auto i, auto j) { return covariance[i * m + i] > covariance[j * m + j]; });
    double retained = 0;
    for (auto column : order) {
        const double eigenvalue = covariance[column * m + column];
        if (eigenvalue <= 1e-12 * trace || result.modes.size() >= maximum_rank)
            break;
        Field mode(dimension, 0);
        for (std::size_t j = 0; j < m; ++j)
            for (std::size_t i = 0; i < dimension; ++i)
                mode[i] += centered[j][i] * (vectors[j * m + column] / std::sqrt(eigenvalue));
        for (int pass = 0; pass < 2; ++pass)
            for (const auto &previous : result.modes) {
                const double overlap = spectral_inner(previous, mode);
                for (std::size_t i = 0; i < dimension; ++i)
                    mode[i] -= overlap * previous[i];
            }
        const double norm = std::sqrt(spectral_inner(mode, mode));
        if (norm < 1e-8)
            throw std::runtime_error("numerically degenerate POD mode");
        for (auto &z : mode)
            z /= norm;
        result.modes.push_back(std::move(mode));
        result.singular_values.push_back(std::sqrt(eigenvalue));
        retained += eigenvalue;
        if (retained / trace >= fraction)
            break;
    }
    if (result.modes.empty())
        throw std::runtime_error("no resolved POD modes");
    result.captured_variance = retained / trace;
    return result;
}
GalerkinModel::GalerkinModel(PODBasis basis, Solver &full) : basis_(std::move(basis)) {
    const auto r = basis_.modes.size();
    if (r == 0 || basis_.mean.size() != full.size() * full.size())
        throw std::invalid_argument("invalid Galerkin basis");
    if (full.parameters().forcing_sigma != 0)
        throw std::invalid_argument("Galerkin prototype currently models deterministic drift only");
    constant_.resize(r);
    linear_.resize(r * r);
    quadratic_.resize(r * r * r);
    const auto constant = full.drift(basis_.mean);
    std::vector<Field> individual;
    individual.reserve(r);
    for (const auto &mode : basis_.modes)
        individual.push_back(full.drift(mode));
    for (std::size_t i = 0; i < r; ++i)
        constant_[i] = spectral_inner(basis_.modes[i], constant);
    for (std::size_t j = 0; j < r; ++j) {
        const auto linear = full.linearized_drift(basis_.mean, basis_.modes[j]);
        for (std::size_t i = 0; i < r; ++i)
            linear_[i * r + j] = spectral_inner(basis_.modes[i], linear);
        auto nonlinear = individual[j];
        const auto n = full.size();
        for (std::size_t index = 0; index < nonlinear.size(); ++index) {
            const double x = wave_number(index % n, n), y = wave_number(index / n, n);
            nonlinear[index] +=
                (full.parameters().viscosity * (x * x + y * y) + full.parameters().friction) *
                basis_.modes[j][index];
        }
        for (std::size_t i = 0; i < r; ++i)
            quadratic_[(i * r + j) * r + j] = spectral_inner(basis_.modes[i], nonlinear);
        for (std::size_t k = 0; k < j; ++k) {
            auto sum = basis_.modes[j];
            for (std::size_t index = 0; index < sum.size(); ++index)
                sum[index] += basis_.modes[k][index];
            auto cross = full.drift(sum);
            for (std::size_t index = 0; index < cross.size(); ++index)
                cross[index] = 0.5 * (cross[index] - individual[j][index] - individual[k][index]);
            for (std::size_t i = 0; i < r; ++i)
                quadratic_[(i * r + j) * r + k] = quadratic_[(i * r + k) * r + j] =
                    spectral_inner(basis_.modes[i], cross);
        }
    }
}
Vector GalerkinModel::rhs(const Vector &a) const {
    const auto r = basis_.modes.size();
    if (a.size() != r)
        throw std::invalid_argument("Galerkin state dimension mismatch");
    auto result = constant_;
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < r; ++j) {
            result[i] += linear_[i * r + j] * a[j];
            for (std::size_t k = 0; k < r; ++k)
                result[i] += quadratic_[(i * r + j) * r + k] * a[j] * a[k];
        }
    return result;
}
void GalerkinModel::step(Vector &state, double dt) const {
    if (!std::isfinite(dt) || dt <= 0)
        throw std::invalid_argument("invalid Galerkin time step");
    const auto stage = [&](const Vector &slope, double scale) {
        auto result = state;
        for (std::size_t i = 0; i < state.size(); ++i)
            result[i] += scale * slope[i];
        return result;
    };
    const auto k1 = rhs(state), k2 = rhs(stage(k1, dt / 2)), k3 = rhs(stage(k2, dt / 2)),
               k4 = rhs(stage(k3, dt));
    for (std::size_t i = 0; i < state.size(); ++i) {
        state[i] += dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        if (!std::isfinite(state[i]))
            throw std::runtime_error("unstable Galerkin trajectory");
    }
}
} // namespace hzl
