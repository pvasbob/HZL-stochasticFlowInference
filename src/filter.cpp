#include "hzl/filter.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>

namespace hzl {
namespace {
constexpr double negative_infinity = -std::numeric_limits<double>::infinity();
double uniform(std::mt19937_64 &engine) {
    return static_cast<double>(engine() >> 11) * 0x1.0p-53;
}
Parameters decode(const Dataset &data, const Vector &q) {
    Parameters p;
    p.viscosity = std::exp(q[0]);
    p.friction = std::exp(q[1]);
    p.forcing_sigma = std::exp(q[2]);
    p.forcing_tau = std::exp(q[3]);
    p.forcing_min = data.forcing_min;
    p.forcing_max = data.forcing_max;
    return p;
}
double gaspari_cohn(double distance, double radius) {
    if (radius <= 0)
        return 1;
    const double x = distance / radius;
    if (x >= 2)
        return 0;
    if (x <= 1)
        return 1 - 5.0 / 3 * x * x + 5.0 / 8 * x * x * x + 0.5 * std::pow(x, 4) -
               0.25 * std::pow(x, 5);
    return 4 - 5 * x + 5.0 / 3 * x * x + 5.0 / 8 * x * x * x - 0.5 * std::pow(x, 4) +
           std::pow(x, 5) / 12 - 2.0 / (3 * x);
}
} // namespace
FilterResult bootstrap_filter(const Dataset &data, Parameters p, std::size_t count,
                              std::uint64_t seed, bool retain_states) {
    data.validate();
    p.forcing_min = data.forcing_min;
    p.forcing_max = data.forcing_max;
    p.validate(data.n);
    if (count < 2)
        throw std::invalid_argument("particle filter needs at least two particles");
    std::vector<Solver> particles;
    particles.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        particles.emplace_back(data.n, p, stream_seed(seed, i));
        particles.back().initialize("multimode");
    }
    Vector weights(count, 1 / static_cast<double>(count)), logs(count);
    std::mt19937_64 resampling(stream_seed(seed, std::numeric_limits<std::uint64_t>::max()));
    FilterResult result;
    std::size_t current_step = 0, begin = 0, group = 0;
    Fourier2D transform(data.n);
    while (begin < data.observations.size()) {
        const auto target = data.observations[begin].step;
        auto end = begin;
        while (end < data.observations.size() && data.observations[end].step == target)
            ++end;
        for (std::size_t i = 0; i < count; ++i) {
            for (auto step = current_step; step < target; ++step)
                particles[i].step(data.dt);
            const auto field = particles[i].physical_vorticity();
            double value = weights[i] > 0 ? std::log(weights[i]) : negative_infinity;
            for (auto index = begin; index < end; ++index) {
                const auto &o = data.observations[index];
                const double residual =
                    (field[o.y * data.n + o.x].real() - o.value) / o.standard_deviation;
                value -= 0.5 * residual * residual + std::log(o.standard_deviation) +
                         0.5 * std::log(2 * std::numbers::pi);
            }
            logs[i] = value;
        }
        const double maximum = *std::max_element(logs.begin(), logs.end());
        if (!std::isfinite(maximum)) {
            result.log_likelihood = negative_infinity;
            return result;
        }
        double total = 0;
        for (auto log : logs)
            total += std::exp(log - maximum);
        const double increment = maximum + std::log(total);
        result.log_likelihood += increment;
        double squared_weights = 0, max_weight = 0;
        for (std::size_t i = 0; i < count; ++i) {
            weights[i] = std::exp(logs[i] - increment);
            squared_weights += weights[i] * weights[i];
            max_weight = std::max(max_weight, weights[i]);
        }
        const double effective = 1 / squared_weights;
        FilterRecord record{target, increment, effective, max_weight, {}};
        if (retain_states) {
            record.mean_vorticity.assign(data.n * data.n, 0);
            for (std::size_t i = 0; i < count; ++i)
                for (std::size_t j = 0; j < record.mean_vorticity.size(); ++j)
                    record.mean_vorticity[j] += weights[i] * particles[i].coefficients()[j];
            transform.inverse(record.mean_vorticity);
        }
        result.history.push_back(std::move(record));
        if (effective < static_cast<double>(count) / 2 && end < data.observations.size()) {
            std::vector<Solver> offspring;
            offspring.reserve(count);
            std::size_t ancestor = 0;
            double cumulative = weights[0];
            const double start = uniform(resampling) / static_cast<double>(count);
            for (std::size_t i = 0; i < count; ++i) {
                const double position = start + static_cast<double>(i) / static_cast<double>(count);
                while (position > cumulative && ancestor + 1 < count)
                    cumulative += weights[++ancestor];
                offspring.push_back(particles[ancestor]);
                // Cloned particles retain the current latent forcing, but must have
                // independent future innovations to avoid duplicated trajectories.
                offspring.back().reseed_future(stream_seed(seed, (group + 1) * count + i));
            }
            particles = std::move(offspring);
            std::fill(weights.begin(), weights.end(), 1 / static_cast<double>(count));
            ++result.resampling_events;
        }
        current_step = target;
        begin = end;
        ++group;
    }
    return result;
}
double stochastic_log_prior(const Vector &q) {
    if (q.size() != 4)
        throw std::invalid_argument(
            "stochastic inference expects log(nu), log(alpha), log(sigma), log(tau)");
    const double centers[] = {std::log(0.03), std::log(0.1), std::log(0.2), std::log(0.5)};
    double result = 0;
    for (std::size_t i = 0; i < q.size(); ++i) {
        if (!std::isfinite(q[i]) || q[i] < -10 || q[i] > (i < 2 ? 0 : 2))
            return negative_infinity;
        const double z = (q[i] - centers[i]) / 1.5;
        result -= 0.5 * z * z;
    }
    return result;
}
double particle_log_likelihood(const Dataset &data, const Vector &q, std::size_t particles,
                               std::uint64_t seed) {
    if (!std::isfinite(stochastic_log_prior(q)))
        return negative_infinity;
    const auto p = decode(data, q);
    const double largest = static_cast<double>((data.n - 1) / 3);
    if (data.dt * (2 * p.viscosity * largest * largest + p.friction) > 2.5)
        return negative_infinity;
    return bootstrap_filter(data, p, particles, seed, false).log_likelihood;
}
double averaged_particle_log_likelihood(const Dataset &data, const Vector &q, std::size_t particles,
                                        std::size_t replicates, std::uint64_t seed) {
    if (replicates == 0)
        throw std::invalid_argument("averaged particle likelihood needs at least one replicate");
    Vector values(replicates);
    for (std::size_t i = 0; i < replicates; ++i)
        values[i] = particle_log_likelihood(data, q, particles, stream_seed(seed, i));
    const double maximum = *std::max_element(values.begin(), values.end());
    if (!std::isfinite(maximum))
        return negative_infinity;
    double total = 0;
    for (double value : values)
        total += std::exp(value - maximum);
    return maximum + std::log(total / static_cast<double>(replicates));
}
std::vector<ParticleSample> particle_metropolis(const Dataset &data, Vector current, double scale,
                                                std::size_t count, std::size_t particles,
                                                std::uint64_t seed) {
    if (!data.stochastic)
        throw std::invalid_argument("particle sampler expects a stochastic observation dataset");
    if (count == 0 || !std::isfinite(scale) || scale <= 0)
        throw std::invalid_argument("invalid particle sampler configuration");
    NormalStream normal(seed);
    std::mt19937_64 random(stream_seed(seed, 1));
    auto likelihood_seed = random();
    double likelihood = particle_log_likelihood(data, current, particles, likelihood_seed);
    double density = likelihood + stochastic_log_prior(current);
    if (!std::isfinite(density))
        throw std::invalid_argument("initial particle posterior estimate is not finite");
    std::vector<ParticleSample> chain;
    chain.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        auto proposal = current;
        for (auto &q : proposal)
            q += scale * normal();
        const auto proposed_seed = random();
        const double proposed_likelihood =
            particle_log_likelihood(data, proposal, particles, proposed_seed);
        const double proposed_density = proposed_likelihood + stochastic_log_prior(proposal);
        const double draw = uniform(random);
        const bool accepted = std::isfinite(proposed_density) &&
                              std::log(std::max(draw, std::numeric_limits<double>::min())) <
                                  std::min(0.0, proposed_density - density);
        if (accepted) {
            current = std::move(proposal);
            likelihood = proposed_likelihood;
            density = proposed_density;
            likelihood_seed = proposed_seed;
        }
        chain.push_back({current, likelihood, density, likelihood_seed, accepted});
    }
    return chain;
}
std::vector<EnsembleFilterRecord> ensemble_kalman_filter(const Dataset &data, Parameters p,
                                                         std::size_t count, std::uint64_t seed,
                                                         EnKFOptions options) {
    data.validate();
    if (count < 2)
        throw std::invalid_argument("EnKF requires at least two ensemble members");
    if (!std::isfinite(options.inflation) || options.inflation < 1 ||
        !std::isfinite(options.localization_radius) || options.localization_radius < 0)
        throw std::invalid_argument(
            "EnKF inflation must be >= 1 and localization radius nonnegative");
    if (options.forecast_every == 0)
        throw std::invalid_argument("EnKF forecast output interval must be positive");
    p.forcing_min = data.forcing_min;
    p.forcing_max = data.forcing_max;
    std::vector<Solver> members;
    members.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        members.emplace_back(data.n, p, stream_seed(seed, i));
        members.back().initialize("multimode");
    }
    NormalStream perturbations(stream_seed(seed, count));
    std::vector<EnsembleFilterRecord> history;
    std::size_t step = 0, begin = 0;
    const auto dimension = data.n * data.n;
    Fourier2D transform(data.n);
    while (begin < data.observations.size()) {
        const auto target = data.observations[begin].step;
        auto end = begin;
        while (end < data.observations.size() && data.observations[end].step == target)
            ++end;
        for (auto &member : members)
            for (auto current = step; current < target; ++current)
                member.step(data.dt);
        if (options.inflation > 1) {
            Field mean_state(dimension, 0), mean_forcing(dimension, 0);
            for (const auto &member : members)
                for (std::size_t j = 0; j < dimension; ++j) {
                    mean_state[j] += member.coefficients()[j] / static_cast<double>(count);
                    mean_forcing[j] += member.forcing()[j] / static_cast<double>(count);
                }
            for (auto &member : members) {
                auto state = member.coefficients(), forcing = member.forcing();
                for (std::size_t j = 0; j < dimension; ++j) {
                    state[j] = mean_state[j] + options.inflation * (state[j] - mean_state[j]);
                    forcing[j] =
                        mean_forcing[j] + options.inflation * (forcing[j] - mean_forcing[j]);
                }
                member.update_latent_state(state, forcing);
            }
        }
        double forecast_error = 0;
        // Forecast diagnostics are measured before assimilating any sensor at this time.
        Field forecast(dimension, 0);
        for (const auto &member : members)
            for (std::size_t j = 0; j < dimension; ++j)
                forecast[j] += member.coefficients()[j] / static_cast<double>(count);
        transform.inverse(forecast);
        for (auto index = begin; index < end; ++index) {
            const auto &o = data.observations[index];
            forecast_error += std::pow(forecast[o.y * data.n + o.x].real() - o.value, 2);
        }
        for (auto index = begin; index < end; ++index) {
            const auto &o = data.observations[index];
            Vector predictions(count);
            double mean_prediction = 0;
            Field mean_state(dimension, 0), mean_forcing(dimension, 0);
            for (std::size_t i = 0; i < count; ++i) {
                predictions[i] = members[i].physical_vorticity()[o.y * data.n + o.x].real();
                mean_prediction += predictions[i] / static_cast<double>(count);
                for (std::size_t j = 0; j < dimension; ++j) {
                    mean_state[j] += members[i].coefficients()[j] / static_cast<double>(count);
                    mean_forcing[j] += members[i].forcing()[j] / static_cast<double>(count);
                }
            }
            double variance = 0;
            Field state_covariance(dimension, 0), forcing_covariance(dimension, 0);
            for (std::size_t i = 0; i < count; ++i) {
                const double delta = predictions[i] - mean_prediction;
                variance += delta * delta / static_cast<double>(count - 1);
                for (std::size_t j = 0; j < dimension; ++j) {
                    state_covariance[j] += (members[i].coefficients()[j] - mean_state[j]) *
                                           (delta / static_cast<double>(count - 1));
                    forcing_covariance[j] += (members[i].forcing()[j] - mean_forcing[j]) *
                                             (delta / static_cast<double>(count - 1));
                }
            }
            if (options.localization_radius > 0) {
                transform.inverse(state_covariance);
                transform.inverse(forcing_covariance);
                for (std::size_t y = 0; y < data.n; ++y)
                    for (std::size_t x = 0; x < data.n; ++x) {
                        const double dx = static_cast<double>(
                                         std::min(x > o.x ? x - o.x : o.x - x,
                                                  data.n - (x > o.x ? x - o.x : o.x - x))),
                                     dy = static_cast<double>(
                                         std::min(y > o.y ? y - o.y : o.y - y,
                                                  data.n - (y > o.y ? y - o.y : o.y - y))),
                                     taper = gaspari_cohn(std::hypot(dx, dy),
                                                          options.localization_radius);
                        state_covariance[y * data.n + x] *= taper;
                        forcing_covariance[y * data.n + x] *= taper;
                    }
                transform.forward(state_covariance);
                transform.forward(forcing_covariance);
            }
            const double denominator = variance + o.standard_deviation * o.standard_deviation;
            for (std::size_t i = 0; i < count; ++i) {
                const double innovation =
                    (o.value + o.standard_deviation * perturbations() - predictions[i]) /
                    denominator;
                auto state = members[i].coefficients(), forcing = members[i].forcing();
                for (std::size_t j = 0; j < dimension; ++j) {
                    state[j] += state_covariance[j] * innovation;
                    forcing[j] += forcing_covariance[j] * innovation;
                }
                members[i].update_latent_state(state, forcing);
            }
        }
        Field mean(dimension, 0);
        for (const auto &member : members)
            for (std::size_t j = 0; j < dimension; ++j)
                mean[j] += member.coefficients()[j] / static_cast<double>(count);
        double variance = 0;
        for (const auto &member : members)
            for (std::size_t j = 0; j < dimension; ++j)
                variance +=
                    std::norm(member.coefficients()[j] - mean[j]) / static_cast<double>(count - 1);
        transform.inverse(mean);
        history.push_back({target, std::sqrt(forecast_error / static_cast<double>(end - begin)),
                           std::sqrt(variance), std::move(mean), false});
        step = target;
        begin = end;
    }
    for (std::size_t horizon = 1; horizon <= options.forecast_steps; ++horizon) {
        for (auto &member : members)
            member.step(data.dt);
        if (horizon % options.forecast_every != 0 && horizon != options.forecast_steps)
            continue;
        Field mean(dimension, 0);
        for (const auto &member : members)
            for (std::size_t j = 0; j < dimension; ++j)
                mean[j] += member.coefficients()[j] / static_cast<double>(count);
        double variance = 0;
        for (const auto &member : members)
            for (std::size_t j = 0; j < dimension; ++j)
                variance +=
                    std::norm(member.coefficients()[j] - mean[j]) / static_cast<double>(count - 1);
        transform.inverse(mean);
        history.push_back({step + horizon, std::numeric_limits<double>::quiet_NaN(),
                           std::sqrt(variance), std::move(mean), true});
    }
    return history;
}
} // namespace hzl
