#include "hzl/monte_carlo.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <stdexcept>

namespace hzl {
namespace {
struct LinearCoefficients {
    Vector a, b, k2;
};
LinearCoefficients coefficients(std::size_t n, Parameters p, double dt) {
    Fourier2D grid(n);
    p.validate(n);
    if (!std::isfinite(dt) || dt <= 0)
        throw std::invalid_argument("invalid control time step");
    LinearCoefficients result{Vector(n * n), Vector(n * n), Vector(n * n)};
    for (std::size_t y = 0; y < n; ++y)
        for (std::size_t x = 0; x < n; ++x) {
            const auto i = y * n + x;
            const double kx = wave_number(x, n), ky = wave_number(y, n);
            result.k2[i] = kx * kx + ky * ky;
            const double z = dt * (p.viscosity * result.k2[i] + p.friction);
            result.a[i] = 1 - z + z * z / 2 - z * z * z / 6 + z * z * z * z / 24;
            result.b[i] = dt * (1 - z / 2 + z * z / 6 - z * z * z / 24);
        }
    return result;
}
} // namespace
CoupledEnergy coupled_energy(std::size_t n, Parameters p, double dt, std::size_t steps,
                             std::uint64_t seed) {
    Solver full(n, p, seed);
    full.initialize("zero");
    const auto c = coefficients(n, p, dt);
    Field linear(n * n, 0);
    for (std::size_t step = 0; step < steps; ++step) {
        full.step(dt);
        for (std::size_t i = 0; i < linear.size(); ++i)
            linear[i] = c.a[i] * linear[i] + c.b[i] * full.midpoint_forcing()[i];
    }
    double energy = 0;
    for (std::size_t i = 1; i < linear.size(); ++i)
        energy += 2 * std::numbers::pi * std::numbers::pi * std::norm(linear[i]) / c.k2[i];
    return {full.diagnostics().energy, energy};
}
double expected_linear_energy(std::size_t n, Parameters p, double dt, std::size_t steps) {
    const auto c = coefficients(n, p, dt);
    const double variance = p.forcing_sigma * p.forcing_sigma * p.forcing_tau / 2,
                 rho = std::exp(-dt / p.forcing_tau);
    double energy = 0;
    for (std::size_t y = 0; y < n; ++y)
        for (std::size_t x = 0; x < n; ++x) {
            const auto i = y * n + x;
            const int kx = wave_number(x, n), ky = wave_number(y, n);
            if (3 * std::abs(kx) >= static_cast<int>(n) ||
                3 * std::abs(ky) >= static_cast<int>(n) ||
                c.k2[i] < p.forcing_min * p.forcing_min || c.k2[i] > p.forcing_max * p.forcing_max)
                continue;
            double state_variance = 0, covariance = 0;
            for (std::size_t step = 0; step < steps; ++step) {
                state_variance = c.a[i] * c.a[i] * state_variance + c.b[i] * c.b[i] * variance +
                                 2 * c.a[i] * c.b[i] * rho * covariance;
                covariance = c.a[i] * rho * covariance + c.b[i] * variance;
            }
            energy += 2 * std::numbers::pi * std::numbers::pi * state_variance / c.k2[i];
        }
    return energy;
}
ControlVariateResult control_variate_experiment(std::size_t n, Parameters p, double dt,
                                                std::size_t steps, std::size_t pilot_count,
                                                std::size_t count, std::uint64_t seed) {
    if (pilot_count < 3 || count < 2 || steps == 0 || p.forcing_sigma <= 0)
        throw std::invalid_argument("control variates require at least 3 pilot and 2 production "
                                    "samples, positive forcing, and nonzero duration");
    ControlVariateResult result;
    result.known_control_mean = expected_linear_energy(n, p, dt, steps);
    std::vector<CoupledEnergy> pilot;
    pilot.reserve(pilot_count);
    Moments full, linear;
    const auto pilot_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < pilot_count; ++i) {
        const auto pair = coupled_energy(n, p, dt, steps, stream_seed(seed, i));
        pilot.push_back(pair);
        full.add(pair.full);
        linear.add(pair.linear);
    }
    double covariance = 0;
    for (const auto &pair : pilot)
        covariance += (pair.full - full.mean) * (pair.linear - linear.mean) /
                      static_cast<double>(pilot_count - 1);
    if (linear.variance() <= 0)
        throw std::runtime_error("control has no sampled variance");
    result.beta = covariance / linear.variance();
    result.pilot_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - pilot_start).count();
    Moments raw, corrected;
    result.production.reserve(count);
    const auto production_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < count; ++i) {
        const auto pair = coupled_energy(n, p, dt, steps, stream_seed(seed, pilot_count + i));
        result.production.push_back(pair);
        raw.add(pair.full);
        corrected.add(pair.full - result.beta * (pair.linear - result.known_control_mean));
    }
    result.production_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - production_start).count();
    result.raw_mean = raw.mean;
    result.corrected_mean = corrected.mean;
    result.raw_variance = raw.variance();
    result.corrected_variance = corrected.variance();
    result.raw_standard_error = raw.standard_error();
    result.corrected_standard_error = corrected.standard_error();
    return result;
}
SplittingResult dissipation_splitting(std::size_t n, Parameters p, double dt,
                                      std::size_t segment_steps, std::size_t maximum_levels,
                                      std::size_t count, double threshold, std::uint64_t seed) {
    if (count < 4 || segment_steps == 0 || maximum_levels == 0 || !std::isfinite(dt) || dt <= 0 ||
        !std::isfinite(threshold) || threshold <= 0)
        throw std::invalid_argument("invalid trajectory-splitting configuration");
    std::vector<Solver> particles;
    particles.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        particles.emplace_back(n, p, stream_seed(seed, i));
        particles.back().initialize("zero");
    }
    Vector scores(count, 0);
    std::vector<std::size_t> steps_used(count, 0);
    const auto total_steps = segment_steps * maximum_levels;
    SplittingResult result;
    double accumulated = 1;
    std::mt19937_64 resampling(stream_seed(seed, std::numeric_limits<std::uint64_t>::max()));
    for (std::size_t level = 0; level < maximum_levels; ++level) {
        const double level_threshold =
            threshold * static_cast<double>(level + 1) / static_cast<double>(maximum_levels);
        std::vector<std::size_t> survivor_indices;
        double crossing_time = 0;
        for (std::size_t i = 0; i < count; ++i) {
            while (steps_used[i] < total_steps && scores[i] < level_threshold) {
                particles[i].step(dt);
                scores[i] = std::max(scores[i], particles[i].diagnostics().dissipation);
                ++steps_used[i];
            }
            if (scores[i] >= level_threshold) {
                survivor_indices.push_back(i);
                crossing_time += static_cast<double>(steps_used[i]) * dt;
            }
        }
        const auto survivors = survivor_indices.size();
        accumulated *= static_cast<double>(survivors) / static_cast<double>(count);
        result.levels.push_back({level + 1, count, survivors,
                                 level + 1 == maximum_levels ? survivors : 0,
                                 survivors > 0 ? crossing_time / static_cast<double>(survivors)
                                               : static_cast<double>(total_steps) * dt,
                                 level_threshold, accumulated});
        if (survivors == 0 || level + 1 == maximum_levels) {
            result.probability_estimate = accumulated;
            result.threshold_reached = survivors > 0 && level + 1 == maximum_levels;
            return result;
        }
        std::vector<Solver> next;
        Vector next_scores;
        std::vector<std::size_t> next_steps;
        next.reserve(count);
        next_scores.reserve(count);
        next_steps.reserve(count);
        std::uniform_int_distribution<std::size_t> choose(0, survivor_indices.size() - 1);
        for (std::size_t i = 0; i < count; ++i) {
            const auto parent = survivor_indices[choose(resampling)];
            next.push_back(particles[parent]);
            next.back().reseed_future(stream_seed(seed, (level + 1) * count + i));
            next_scores.push_back(scores[parent]);
            next_steps.push_back(steps_used[parent]);
        }
        particles = std::move(next);
        scores = std::move(next_scores);
        steps_used = std::move(next_steps);
    }
    return result;
}
AntitheticResult parameter_antithetic_experiment(std::size_t n, Parameters center, double log_sd,
                                                 double dt, std::size_t steps,
                                                 std::size_t pair_count, std::uint64_t seed) {
    if (center.forcing_sigma != 0 || !std::isfinite(log_sd) || log_sd <= 0 || dt <= 0 ||
        steps == 0 || pair_count < 2)
        throw std::invalid_argument("invalid antithetic parameter experiment");
    const auto evaluate = [&](double z0, double z1) {
        auto p = center;
        p.viscosity *= std::exp(log_sd * z0);
        p.friction *= std::exp(log_sd * z1);
        Solver solver(n, p, 0);
        solver.initialize("multimode");
        for (std::size_t step = 0; step < steps; ++step)
            solver.step(dt);
        return solver.diagnostics().energy;
    };
    const auto start = std::chrono::steady_clock::now();
    NormalStream raw_random(stream_seed(seed, 0)), paired_random(stream_seed(seed, 1));
    Moments raw, paired;
    for (std::size_t i = 0; i < 2 * pair_count; ++i)
        raw.add(evaluate(raw_random(), raw_random()));
    for (std::size_t i = 0; i < pair_count; ++i) {
        const double z0 = paired_random(), z1 = paired_random();
        paired.add(0.5 * (evaluate(z0, z1) + evaluate(-z0, -z1)));
    }
    return {raw.mean,
            paired.mean,
            raw.standard_error(),
            paired.standard_error(),
            raw.variance(),
            paired.variance(),
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()};
}
} // namespace hzl
