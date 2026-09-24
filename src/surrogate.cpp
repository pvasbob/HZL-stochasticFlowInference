#include "hzl/surrogate.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace hzl {
Vector QuadraticSurrogate::features(const Vector &input) const {
    if (input.size() != center_.size())
        throw std::invalid_argument("surrogate input dimension mismatch");
    Vector normalized(input.size()), result{1};
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (!std::isfinite(input[i]))
            throw std::invalid_argument("nonfinite surrogate input");
        normalized[i] = (input[i] - center_[i]) / scales_[i];
        result.push_back(normalized[i]);
    }
    for (std::size_t i = 0; i < input.size(); ++i)
        for (std::size_t j = i; j < input.size(); ++j)
            result.push_back(normalized[i] * normalized[j]);
    return result;
}
QuadraticSurrogate::QuadraticSurrogate(const std::vector<Vector> &inputs, const Vector &outputs,
                                       Vector center, Vector scales)
    : center_(std::move(center)), scales_(std::move(scales)) {
    if (center_.empty() || scales_.size() != center_.size() || inputs.size() != outputs.size())
        throw std::invalid_argument("invalid surrogate training dimensions");
    for (std::size_t i = 0; i < center_.size(); ++i)
        if (!std::isfinite(center_[i]) || !std::isfinite(scales_[i]) || scales_[i] <= 0)
            throw std::invalid_argument("invalid surrogate normalization");
    const auto rows = inputs.size(), columns = features(center_).size();
    if (rows < columns)
        throw std::invalid_argument("not enough observations to fit quadratic surrogate");
    Vector matrix(rows * columns), right = outputs;
    for (std::size_t i = 0; i < rows; ++i) {
        const auto row = features(inputs[i]);
        std::copy(row.begin(), row.end(),
                  matrix.begin() + static_cast<std::ptrdiff_t>(i * columns));
        if (!std::isfinite(outputs[i]))
            throw std::invalid_argument("nonfinite surrogate target");
    }
    // Householder QR: avoid squaring the regression condition number with normal equations.
    for (std::size_t column = 0; column < columns; ++column) {
        double norm = 0;
        for (std::size_t row = column; row < rows; ++row)
            norm = std::hypot(norm, matrix[row * columns + column]);
        if (norm < 1e-12)
            throw std::invalid_argument("rank-deficient surrogate design");
        const double diagonal = -std::copysign(norm, matrix[column * columns + column]);
        Vector reflector(rows - column);
        for (std::size_t row = column; row < rows; ++row)
            reflector[row - column] = matrix[row * columns + column];
        reflector[0] -= diagonal;
        const double squared =
            std::inner_product(reflector.begin(), reflector.end(), reflector.begin(), 0.0);
        for (std::size_t j = column; j < columns; ++j) {
            double dot = 0;
            for (std::size_t row = column; row < rows; ++row)
                dot += reflector[row - column] * matrix[row * columns + j];
            for (std::size_t row = column; row < rows; ++row)
                matrix[row * columns + j] -= 2 * reflector[row - column] * dot / squared;
        }
        double dot = 0;
        for (std::size_t row = column; row < rows; ++row)
            dot += reflector[row - column] * right[row];
        for (std::size_t row = column; row < rows; ++row)
            right[row] -= 2 * reflector[row - column] * dot / squared;
        matrix[column * columns + column] = diagonal;
        for (std::size_t row = column + 1; row < rows; ++row)
            matrix[row * columns + column] = 0;
    }
    coefficients_.resize(columns);
    for (std::size_t row = columns; row > 0; --row) {
        const auto i = row - 1;
        double value = right[i];
        for (std::size_t j = i + 1; j < columns; ++j)
            value -= matrix[i * columns + j] * coefficients_[j];
        coefficients_[i] = value / matrix[i * columns + i];
    }
    double squared_error = 0;
    for (std::size_t i = 0; i < rows; ++i)
        squared_error += std::pow((*this)(inputs[i]) - outputs[i], 2);
    training_rmse_ = std::sqrt(squared_error / static_cast<double>(rows));
}
double QuadraticSurrogate::operator()(const Vector &input) const {
    const auto values = features(input);
    return std::inner_product(values.begin(), values.end(), coefficients_.begin(), 0.0);
}
GaussianRBFSurrogate::GaussianRBFSurrogate(std::vector<Vector> inputs, Vector outputs,
                                           Vector center, Vector scales, double shape, double ridge)
    : center_(std::move(center)), scales_(std::move(scales)), shape_(shape) {
    if (inputs.empty() || inputs.size() != outputs.size() || center_.empty() ||
        center_.size() != scales_.size() || !std::isfinite(shape) || shape <= 0 ||
        !std::isfinite(ridge) || ridge < 0)
        throw std::invalid_argument("invalid Gaussian RBF configuration");
    for (double scale : scales_)
        if (!std::isfinite(scale) || scale <= 0)
            throw std::invalid_argument("invalid Gaussian RBF normalization");
    centers_.reserve(inputs.size());
    for (const auto &input : inputs)
        centers_.push_back(normalize(input));
    const auto n = centers_.size();
    Vector matrix(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(outputs[i]))
            throw std::invalid_argument("nonfinite Gaussian RBF target");
        for (std::size_t j = 0; j < n; ++j) {
            double squared = 0;
            for (std::size_t k = 0; k < center_.size(); ++k)
                squared += std::pow(centers_[i][k] - centers_[j][k], 2);
            matrix[i * n + j] = std::exp(-shape_ * shape_ * squared);
        }
        matrix[i * n + i] += ridge;
    }
    weights_ = outputs;
    // Partial-pivoted elimination is adequate for this small, ridge-regularized design.
    for (std::size_t column = 0; column < n; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < n; ++row)
            if (std::abs(matrix[row * n + column]) > std::abs(matrix[pivot * n + column]))
                pivot = row;
        if (std::abs(matrix[pivot * n + column]) < 1e-14)
            throw std::invalid_argument("singular Gaussian RBF system");
        for (std::size_t j = column; j < n; ++j)
            std::swap(matrix[column * n + j], matrix[pivot * n + j]);
        std::swap(weights_[column], weights_[pivot]);
        for (std::size_t row = column + 1; row < n; ++row) {
            const double factor = matrix[row * n + column] / matrix[column * n + column];
            for (std::size_t j = column; j < n; ++j)
                matrix[row * n + j] -= factor * matrix[column * n + j];
            weights_[row] -= factor * weights_[column];
        }
    }
    for (std::size_t row = n; row > 0; --row) {
        const auto i = row - 1;
        for (std::size_t j = i + 1; j < n; ++j)
            weights_[i] -= matrix[i * n + j] * weights_[j];
        weights_[i] /= matrix[i * n + i];
    }
    double squared_error = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i)
        squared_error += std::pow((*this)(inputs[i]) - outputs[i], 2);
    training_rmse_ = std::sqrt(squared_error / static_cast<double>(inputs.size()));
}
Vector GaussianRBFSurrogate::normalize(const Vector &input) const {
    if (input.size() != center_.size())
        throw std::invalid_argument("Gaussian RBF input dimension mismatch");
    Vector result(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (!std::isfinite(input[i]))
            throw std::invalid_argument("nonfinite Gaussian RBF input");
        result[i] = (input[i] - center_[i]) / scales_[i];
    }
    return result;
}
double GaussianRBFSurrogate::operator()(const Vector &input) const {
    const auto point = normalize(input);
    double result = 0;
    for (std::size_t i = 0; i < centers_.size(); ++i) {
        double squared = 0;
        for (std::size_t j = 0; j < point.size(); ++j)
            squared += std::pow(point[j] - centers_[i][j], 2);
        result += weights_[i] * std::exp(-shape_ * shape_ * squared);
    }
    return result;
}
DelayedAcceptanceResult delayed_acceptance(const Objective &exact, const Objective &approximate,
                                           Vector current, const Vector &scales, std::size_t count,
                                           std::uint64_t seed) {
    if (current.empty() || current.size() != scales.size() || count == 0)
        throw std::invalid_argument("invalid delayed-acceptance dimensions");
    for (double scale : scales)
        if (!std::isfinite(scale) || scale <= 0)
            throw std::invalid_argument("invalid delayed-acceptance proposal");
    double exact_value = exact(current), approximate_value = approximate(current);
    if (!std::isfinite(exact_value) || !std::isfinite(approximate_value))
        throw std::invalid_argument("initial delayed-acceptance densities must be finite");
    NormalStream normal(seed);
    std::mt19937_64 random(stream_seed(seed, 1));
    const auto log_uniform = [&]() {
        return std::log((static_cast<double>(random() >> 11) + 0.5) * 0x1.0p-53);
    };
    DelayedAcceptanceResult result;
    result.exact_evaluations = 1;
    result.samples.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        auto proposal = current;
        for (std::size_t j = 0; j < proposal.size(); ++j)
            proposal[j] += scales[j] * normal();
        const double approximate_proposal = approximate(proposal);
        bool accepted = false;
        if (std::isfinite(approximate_proposal) &&
            log_uniform() < std::min(0.0, approximate_proposal - approximate_value)) {
            ++result.stage_one_acceptances;
            const double exact_proposal = exact(proposal);
            ++result.exact_evaluations;
            if (std::isfinite(exact_proposal) &&
                log_uniform() < std::min(0.0, exact_proposal - exact_value - approximate_proposal +
                                                  approximate_value)) {
                current = std::move(proposal);
                exact_value = exact_proposal;
                approximate_value = approximate_proposal;
                accepted = true;
            }
        }
        result.samples.push_back({current, exact_value, accepted});
    }
    return result;
}
} // namespace hzl
