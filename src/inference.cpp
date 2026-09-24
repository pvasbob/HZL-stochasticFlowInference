#include "hzl/inference.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace hzl {
OptimizationResult nelder_mead(const Objective &f, const Vector &initial, double step,
                               std::size_t max_iterations, double tolerance) {
    if (initial.empty() || !std::isfinite(step) || step <= 0 || !std::isfinite(tolerance) ||
        tolerance <= 0 || max_iterations == 0)
        throw std::invalid_argument("invalid simplex configuration");
    for (double x : initial)
        if (!std::isfinite(x))
            throw std::invalid_argument("nonfinite initial parameter");
    struct Vertex {
        Vector x;
        double value;
    };
    const auto dimensions = initial.size();
    std::size_t evaluations = 0;
    const auto evaluate = [&](const Vector &x) {
        ++evaluations;
        const double y = f(x);
        return std::isfinite(y) ? y : std::numeric_limits<double>::infinity();
    };
    std::vector<Vertex> simplex;
    simplex.push_back({initial, evaluate(initial)});
    for (std::size_t j = 0; j < dimensions; ++j) {
        auto x = initial;
        x[j] += step;
        simplex.push_back({x, evaluate(x)});
    }
    OptimizationResult result;
    const auto sort = [&]() {
        std::sort(simplex.begin(), simplex.end(),
                  [](const auto &a, const auto &b) { return a.value < b.value; });
    };
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
        sort();
        result.history.push_back(
            {iteration, evaluations, simplex.front().value, simplex.front().x});
        double spread = 0;
        for (const auto &v : simplex)
            for (std::size_t j = 0; j < dimensions; ++j)
                spread = std::max(spread, std::abs(v.x[j] - simplex.front().x[j]));
        if (std::isfinite(simplex.back().value) && spread < tolerance &&
            simplex.back().value - simplex.front().value <
                tolerance * (1 + std::abs(simplex.front().value))) {
            result.converged = true;
            result.reason = "simplex diameter and objective spread";
            break;
        }
        Vector centroid(dimensions, 0);
        for (std::size_t i = 0; i < dimensions; ++i)
            for (std::size_t j = 0; j < dimensions; ++j)
                centroid[j] += simplex[i].x[j] / static_cast<double>(dimensions);
        const auto candidate = [&](double scale) {
            Vector x(dimensions);
            for (std::size_t j = 0; j < dimensions; ++j)
                x[j] = centroid[j] + scale * (centroid[j] - simplex.back().x[j]);
            return Vertex{x, evaluate(x)};
        };
        auto reflected = candidate(1);
        if (reflected.value < simplex.front().value) {
            auto expanded = candidate(2);
            simplex.back() = expanded.value < reflected.value ? expanded : reflected;
        } else if (reflected.value < simplex[dimensions - 1].value)
            simplex.back() = reflected;
        else {
            const bool outside = reflected.value < simplex.back().value;
            auto contracted = candidate(outside ? 0.5 : -0.5);
            if (contracted.value < (outside ? reflected.value : simplex.back().value))
                simplex.back() = contracted;
            else
                for (std::size_t i = 1; i < simplex.size(); ++i) {
                    for (std::size_t j = 0; j < dimensions; ++j)
                        simplex[i].x[j] =
                            simplex.front().x[j] + 0.5 * (simplex[i].x[j] - simplex.front().x[j]);
                    simplex[i].value = evaluate(simplex[i].x);
                }
        }
    }
    sort();
    result.parameters = simplex.front().x;
    result.value = simplex.front().value;
    result.evaluations = evaluations;
    if (!result.converged)
        result.reason = "iteration limit";
    return result;
}
Vector central_gradient(const Objective &f, const Vector &x, double step) {
    if (!std::isfinite(step) || step <= 0)
        throw std::invalid_argument("gradient step must be positive");
    Vector gradient(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        auto plus = x, minus = x;
        plus[i] += step;
        minus[i] -= step;
        gradient[i] = (f(plus) - f(minus)) / (2 * step);
    }
    return gradient;
}
OptimizationResult bfgs(const DifferentiableObjective &f, const Vector &initial,
                        std::size_t max_iterations, double tolerance) {
    if (initial.empty() || max_iterations == 0 || !std::isfinite(tolerance) || tolerance <= 0)
        throw std::invalid_argument("invalid BFGS configuration");
    const auto n = initial.size();
    Vector x = initial;
    auto current = f(x);
    std::size_t evaluations = 1;
    Vector h(n * n, 0);
    const auto dot = [](const Vector &a, const Vector &b) {
        return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    };
    const auto reset = [&]() {
        std::fill(h.begin(), h.end(), 0);
        for (std::size_t i = 0; i < n; ++i)
            h[i * n + i] = 1;
    };
    reset();
    if (!std::isfinite(current.value) || current.gradient.size() != n)
        throw std::invalid_argument("invalid BFGS initial objective");
    OptimizationResult result;
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
        result.history.push_back({iteration, evaluations, current.value, x});
        if (std::sqrt(dot(current.gradient, current.gradient)) < tolerance) {
            result.converged = true;
            result.reason = "gradient norm";
            break;
        }
        Vector direction(n, 0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                direction[i] -= h[i * n + j] * current.gradient[j];
        double derivative = dot(direction, current.gradient);
        if (!std::isfinite(derivative) || derivative >= 0) {
            reset();
            for (std::size_t i = 0; i < n; ++i)
                direction[i] = -current.gradient[i];
            derivative = -dot(current.gradient, current.gradient);
        }
        double length = 1;
        Vector candidate(n);
        ValueGradient next;
        bool accepted = false;
        for (int trial = 0; trial < 40; ++trial) {
            for (std::size_t i = 0; i < n; ++i)
                candidate[i] = x[i] + length * direction[i];
            next = f(candidate);
            ++evaluations;
            if (std::isfinite(next.value) && next.gradient.size() == n &&
                next.value <= current.value + 1e-4 * length * derivative) {
                accepted = true;
                break;
            }
            length *= 0.5;
        }
        if (!accepted) {
            result.reason = "Armijo line search failed";
            break;
        }
        Vector s(n), y(n), hy(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            s[i] = candidate[i] - x[i];
            y[i] = next.gradient[i] - current.gradient[i];
        }
        const double ys = dot(y, s);
        if (ys > 1e-12 * std::sqrt(dot(y, y) * dot(s, s))) {
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    hy[i] += h[i * n + j] * y[j];
            const double factor = (1 + dot(y, hy) / ys) / ys;
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    h[i * n + j] += factor * s[i] * s[j] - (s[i] * hy[j] + hy[i] * s[j]) / ys;
        } else
            reset();
        x = std::move(candidate);
        current = std::move(next);
    }
    result.parameters = x;
    result.value = current.value;
    result.evaluations = evaluations;
    if (result.reason.empty())
        result.reason = "iteration limit";
    return result;
}
void Dataset::validate() const {
    Fourier2D grid(n);
    Parameters p;
    p.forcing_min = forcing_min;
    p.forcing_max = forcing_max;
    p.validate(n);
    if (n < 16 || !std::isfinite(dt) || dt <= 0 || observations.empty())
        throw std::invalid_argument("invalid observation dataset");
    std::size_t previous = 0;
    for (const auto &o : observations) {
        if (o.step < previous || o.x >= n || o.y >= n || !std::isfinite(o.value) ||
            !std::isfinite(o.standard_deviation) || o.standard_deviation <= 0)
            throw std::invalid_argument("observations must be time ordered, finite, in bounds, and "
                                        "have positive uncertainty");
        previous = o.step;
    }
}
void Dataset::save(const std::filesystem::path &path) const {
    validate();
    std::ofstream out(path);
    out << std::setprecision(17);
    out << "HZL_OBSERVATIONS_V2," << n << ',' << dt << ',' << stochastic << ',' << forcing_min
        << ',' << forcing_max << "\nstep,x,y,vorticity,standard_deviation\n";
    for (const auto &o : observations)
        out << o.step << ',' << o.x << ',' << o.y << ',' << o.value << ',' << o.standard_deviation
            << '\n';
    if (!out)
        throw std::runtime_error("observation write failed");
}
Dataset Dataset::load(const std::filesystem::path &path) {
    std::ifstream in(path);
    std::string line, magic;
    Dataset data;
    if (!std::getline(in, line))
        throw std::runtime_error("cannot read observation dataset");
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream header(line);
    if (!(header >> magic >> data.n >> data.dt) ||
        (magic != "HZL_OBSERVATIONS_V1" && magic != "HZL_OBSERVATIONS_V2"))
        throw std::runtime_error("invalid dataset header");
    if (magic == "HZL_OBSERVATIONS_V2" &&
        !(header >> data.stochastic >> data.forcing_min >> data.forcing_max))
        throw std::runtime_error("invalid stochastic dataset metadata");
    if (!std::getline(in, line) || line != "step,x,y,vorticity,standard_deviation")
        throw std::runtime_error("invalid dataset columns");
    while (std::getline(in, line)) {
        if (line.empty())
            continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        Observation o;
        std::string extra;
        if (!(row >> o.step >> o.x >> o.y >> o.value >> o.standard_deviation) || (row >> extra))
            throw std::runtime_error("invalid observation row");
        data.observations.push_back(o);
    }
    data.validate();
    return data;
}
Dataset synthetic_observations(std::size_t n, double dt, std::size_t steps, std::size_t every,
                               Parameters truth, double noise, std::uint64_t seed,
                               std::vector<Field> *truth_fields) {
    if (every == 0 || steps < every || !std::isfinite(noise) || noise <= 0)
        throw std::invalid_argument(
            "observation experiment requires steps >= every and positive noise");
    Solver solver(n, truth, stream_seed(seed, 0));
    solver.initialize("multimode");
    NormalStream random(stream_seed(seed, 1));
    Dataset data;
    data.n = n;
    data.dt = dt;
    data.stochastic = truth.forcing_sigma > 0;
    data.forcing_min = truth.forcing_min;
    data.forcing_max = truth.forcing_max;
    if (truth_fields)
        truth_fields->clear();
    for (std::size_t step = 1; step <= steps; ++step) {
        solver.step(dt);
        if (step % every == 0) {
            const auto field = solver.physical_vorticity();
            if (truth_fields)
                truth_fields->push_back(field);
            for (std::size_t j = 0; j < 8; ++j) {
                const auto x = (j * n / 8 + 1) % n, y = (j * j + 3 * j + 2) % n;
                data.observations.push_back(
                    {step, x, y, field[y * n + x].real() + noise * random(), noise});
            }
        }
    }
    data.validate();
    return data;
}
Vector predict(const Dataset &data, const Vector &q) {
    data.validate();
    if (data.stochastic)
        throw std::invalid_argument("deterministic prediction cannot marginalize stochastic "
                                    "forcing; use particle inference");
    if (q.size() != 2 || !std::isfinite(q[0]) || !std::isfinite(q[1]))
        throw std::invalid_argument("expected log(nu), log(alpha)");
    Parameters p;
    p.viscosity = std::exp(q[0]);
    p.friction = std::exp(q[1]);
    p.forcing_sigma = 0;
    p.forcing_min = data.forcing_min;
    p.forcing_max = data.forcing_max;
    Solver solver(data.n, p, 0);
    solver.initialize("multimode");
    Vector predictions;
    predictions.reserve(data.observations.size());
    std::size_t step = 0, cached_step = std::numeric_limits<std::size_t>::max();
    Field field;
    for (const auto &o : data.observations) {
        while (step < o.step) {
            solver.step(data.dt);
            ++step;
        }
        if (cached_step != step) {
            field = solver.physical_vorticity();
            cached_step = step;
        }
        predictions.push_back(field[o.y * data.n + o.x].real());
    }
    return predictions;
}
double negative_log_likelihood(const Dataset &data, const Vector &q) {
    if (q.size() != 2)
        throw std::invalid_argument("expected two parameters");
    // Explicit numerical parameter domain; invalid candidates have zero likelihood.
    for (double x : q)
        if (!std::isfinite(x) || x < -12 || x > 0)
            return std::numeric_limits<double>::infinity();
    const double largest = static_cast<double>((data.n - 1) / 3);
    if (data.dt * (2 * std::exp(q[0]) * largest * largest + std::exp(q[1])) > 2.5)
        return std::numeric_limits<double>::infinity();
    const auto model = predict(data, q);
    double objective = 0;
    for (std::size_t i = 0; i < model.size(); ++i) {
        const double residual =
            (model[i] - data.observations[i].value) / data.observations[i].standard_deviation;
        objective += 0.5 * residual * residual;
    }
    return objective; // Parameter-independent Gaussian normalization omitted.
}
ValueGradient likelihood_gradient(const Dataset &data, const Vector &q) {
    data.validate();
    if (data.stochastic || q.size() != 2)
        throw std::invalid_argument(
            "tangent likelihood requires deterministic data and two parameters");
    const double infinity = std::numeric_limits<double>::infinity();
    for (double value : q)
        if (!std::isfinite(value) || value < -12 || value > 0)
            return {infinity, Vector(2, 0)};
    Parameters p;
    p.viscosity = std::exp(q[0]);
    p.friction = std::exp(q[1]);
    p.forcing_sigma = 0;
    p.forcing_min = data.forcing_min;
    p.forcing_max = data.forcing_max;
    const double largest = static_cast<double>((data.n - 1) / 3);
    if (data.dt * (2 * p.viscosity * largest * largest + p.friction) > 2.5)
        return {infinity, Vector(2, 0)};
    Solver solver(data.n, p, 0);
    solver.initialize("multimode");
    Fourier2D fft(data.n);
    std::array<Field, 2> tangent{Field(data.n * data.n), Field(data.n * data.n)}, physical_tangent;
    std::size_t step = 0, cached = std::numeric_limits<std::size_t>::max();
    Field field;
    ValueGradient result{0, Vector(2, 0)};
    for (const auto &o : data.observations) {
        while (step < o.step) {
            solver.step_tangent(data.dt, tangent);
            ++step;
        }
        if (cached != step) {
            field = solver.physical_vorticity();
            physical_tangent = tangent;
            for (auto &t : physical_tangent)
                fft.inverse(t);
            cached = step;
        }
        const auto i = o.y * data.n + o.x;
        const double residual = (field[i].real() - o.value) / o.standard_deviation;
        result.value += 0.5 * residual * residual;
        for (std::size_t j = 0; j < 2; ++j)
            result.gradient[j] += residual * physical_tangent[j][i].real() / o.standard_deviation;
    }
    return result;
}
std::vector<Sample> metropolis(const Objective &log_density, Vector current, const Vector &scales,
                               std::size_t count, std::uint64_t seed) {
    if (current.empty() || current.size() != scales.size() || count == 0)
        throw std::invalid_argument("invalid sampler dimensions");
    for (double scale : scales)
        if (!std::isfinite(scale) || scale <= 0)
            throw std::invalid_argument("proposal scales must be positive");
    NormalStream normal(seed);
    std::mt19937_64 uniform(stream_seed(seed, 1));
    double density = log_density(current);
    if (!std::isfinite(density))
        throw std::invalid_argument("initial log density must be finite");
    std::vector<Sample> samples;
    samples.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        auto proposal = current;
        for (std::size_t j = 0; j < proposal.size(); ++j)
            proposal[j] += scales[j] * normal();
        const double proposed = log_density(proposal);
        const double log_u = std::log((static_cast<double>(uniform() >> 11) + 0.5) * 0x1.0p-53);
        const bool accepted = std::isfinite(proposed) && log_u < std::min(0.0, proposed - density);
        if (accepted) {
            current = std::move(proposal);
            density = proposed;
        }
        samples.push_back({current, density, accepted});
    }
    return samples;
}
std::vector<Sample> adaptive_metropolis(const Objective &density, Vector current, double scale,
                                        std::size_t count, std::size_t warmup, std::uint64_t seed) {
    const auto dimensions = current.size();
    if (dimensions == 0 || !std::isfinite(scale) || scale <= 0 || warmup < 50 || count <= warmup)
        throw std::invalid_argument("adaptive sampler needs positive scale, at least 50 warmup "
                                    "draws, and subsequent production draws");
    double log_density = density(current);
    if (!std::isfinite(log_density))
        throw std::invalid_argument("nonfinite initial posterior");
    NormalStream normal(seed);
    std::mt19937_64 uniform(stream_seed(seed, 1));
    Vector mean = current, m2(dimensions * dimensions, 0), lower(dimensions * dimensions, 0);
    for (std::size_t j = 0; j < dimensions; ++j)
        lower[j * dimensions + j] = scale;
    std::size_t accumulated = 1;
    std::vector<Sample> samples;
    samples.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        Vector innovation(dimensions);
        for (auto &z : innovation)
            z = normal();
        auto proposal = current;
        for (std::size_t row = 0; row < dimensions; ++row)
            for (std::size_t col = 0; col <= row; ++col)
                proposal[row] += lower[row * dimensions + col] * innovation[col];
        const double proposed = density(proposal),
                     log_u = std::log((static_cast<double>(uniform() >> 11) + 0.5) * 0x1.0p-53);
        const bool accepted =
            std::isfinite(proposed) && log_u < std::min(0.0, proposed - log_density);
        if (accepted) {
            current = std::move(proposal);
            log_density = proposed;
        }
        samples.push_back({current, log_density, accepted});
        if (i < warmup) {
            ++accumulated;
            Vector delta(dimensions);
            for (std::size_t j = 0; j < dimensions; ++j) {
                delta[j] = current[j] - mean[j];
                mean[j] += delta[j] / static_cast<double>(accumulated);
            }
            for (std::size_t row = 0; row < dimensions; ++row)
                for (std::size_t col = 0; col < dimensions; ++col)
                    m2[row * dimensions + col] += delta[row] * (current[col] - mean[col]);
            if (i >= 49 && ((i + 1) % 25 == 0 || i + 1 == warmup)) {
                const double factor = 2.38 * 2.38 / static_cast<double>(dimensions) /
                                      static_cast<double>(accumulated - 1);
                std::fill(lower.begin(), lower.end(), 0);
                for (std::size_t row = 0; row < dimensions; ++row)
                    for (std::size_t col = 0; col <= row; ++col) {
                        double value =
                            factor * 0.5 *
                                (m2[row * dimensions + col] + m2[col * dimensions + row]) +
                            (row == col ? 1e-8 : 0);
                        for (std::size_t k = 0; k < col; ++k)
                            value -= lower[row * dimensions + k] * lower[col * dimensions + k];
                        if (row == col) {
                            if (value <= 0 || !std::isfinite(value))
                                throw std::runtime_error(
                                    "adaptive proposal covariance is not positive definite");
                            lower[row * dimensions + col] = std::sqrt(value);
                        } else
                            lower[row * dimensions + col] = value / lower[col * dimensions + col];
                    }
            }
        }
    }
    return samples;
}
std::vector<Sample> hamiltonian_monte_carlo(const DifferentiableObjective &potential,
                                            Vector current, double step_size, std::size_t leapfrog,
                                            std::size_t count, std::uint64_t seed) {
    if (current.empty() || !std::isfinite(step_size) || step_size <= 0 || leapfrog == 0 ||
        count == 0)
        throw std::invalid_argument("invalid HMC configuration");
    auto value = potential(current);
    if (!std::isfinite(value.value) || value.gradient.size() != current.size())
        throw std::invalid_argument("invalid initial HMC potential");
    NormalStream normal(seed);
    std::mt19937_64 uniform(stream_seed(seed, 1));
    std::vector<Sample> samples;
    samples.reserve(count);
    const auto kinetic = [](const Vector &p) {
        return 0.5 * std::inner_product(p.begin(), p.end(), p.begin(), 0.0);
    };
    for (std::size_t iteration = 0; iteration < count; ++iteration) {
        Vector momentum(current.size());
        for (auto &p : momentum)
            p = normal();
        const double initial_energy = value.value + kinetic(momentum);
        auto proposal = current;
        auto next = value;
        bool valid = true;
        for (std::size_t j = 0; j < momentum.size(); ++j)
            momentum[j] -= 0.5 * step_size * value.gradient[j];
        for (std::size_t step = 0; step < leapfrog; ++step) {
            for (std::size_t j = 0; j < proposal.size(); ++j)
                proposal[j] += step_size * momentum[j];
            next = potential(proposal);
            if (!std::isfinite(next.value) || next.gradient.size() != proposal.size()) {
                valid = false;
                break;
            }
            for (double g : next.gradient)
                if (!std::isfinite(g))
                    valid = false;
            if (!valid)
                break;
            const double fraction = step + 1 == leapfrog ? 0.5 : 1;
            for (std::size_t j = 0; j < momentum.size(); ++j)
                momentum[j] -= fraction * step_size * next.gradient[j];
        }
        const double delta = valid ? next.value + kinetic(momentum) - initial_energy
                                   : std::numeric_limits<double>::infinity();
        const bool divergent = !std::isfinite(delta) || std::abs(delta) > 1000;
        const double log_u = std::log((static_cast<double>(uniform() >> 11) + 0.5) * 0x1.0p-53);
        const bool accepted = valid && std::isfinite(delta) && log_u < std::min(0.0, -delta);
        if (accepted) {
            current = std::move(proposal);
            value = std::move(next);
        }
        samples.push_back({current, -value.value, accepted, divergent,
                           std::isfinite(delta) ? delta : std::numeric_limits<double>::max()});
    }
    return samples;
}
void Moments::add(double x) {
    if (!std::isfinite(x))
        throw std::invalid_argument("nonfinite statistical sample");
    ++count;
    const double delta = x - mean;
    mean += delta / static_cast<double>(count);
    m2 += delta * (x - mean);
}
double Moments::variance() const {
    return count > 1 ? m2 / static_cast<double>(count - 1)
                     : std::numeric_limits<double>::quiet_NaN();
}
double Moments::standard_error() const {
    return std::sqrt(variance() / static_cast<double>(count));
}
std::vector<Diagnostics> ensemble(std::size_t n, Parameters p, double dt, std::size_t steps,
                                  std::size_t count, std::size_t workers, std::uint64_t seed) {
    if (count == 0 || workers == 0)
        throw std::invalid_argument("ensemble count and workers must be positive");
    std::vector<Diagnostics> results(count);
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex mutex;
    {
        std::vector<std::jthread> threads;
        for (std::size_t worker = 0; worker < std::min(workers, count); ++worker)
            threads.emplace_back([&]() {
                try {
                    while (!failed.load()) {
                        const auto index = next.fetch_add(1);
                        if (index >= count)
                            break;
                        Solver solver(n, p, stream_seed(seed, index));
                        solver.initialize("zero");
                        for (std::size_t step = 0; step < steps; ++step)
                            solver.step(dt);
                        results[index] = solver.diagnostics();
                    }
                } catch (...) {
                    std::lock_guard lock(mutex);
                    if (!error)
                        error = std::current_exception();
                    failed.store(true);
                }
            });
    }
    if (error)
        std::rethrow_exception(error);
    return results;
}
} // namespace hzl
