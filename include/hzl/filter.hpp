#pragma once
#include "hzl/inference.hpp"

namespace hzl {
struct FilterRecord {
    std::size_t step{};
    double log_likelihood_increment{}, effective_particles{}, maximum_weight{};
    Field mean_vorticity;
};
struct FilterResult {
    double log_likelihood = 0;
    std::size_t resampling_events = 0;
    std::vector<FilterRecord> history;
};
// Bootstrap particle filter with systematic resampling below N/2 ESS.
// The likelihood estimate (before taking logs) is unbiased for the discrete model.
FilterResult bootstrap_filter(const Dataset &data, Parameters p, std::size_t particles,
                              std::uint64_t seed, bool retain_states = true);
double particle_log_likelihood(const Dataset &data, const Vector &log_parameters,
                               std::size_t particles, std::uint64_t seed);
// Log of an arithmetic mean of independent likelihood estimates. The mean on
// the likelihood scale remains unbiased; fixed seeds provide a repeatable CRN objective.
double averaged_particle_log_likelihood(const Dataset &data, const Vector &log_parameters,
                                        std::size_t particles, std::size_t replicates,
                                        std::uint64_t seed);
struct ParticleSample {
    Vector parameters;
    double log_likelihood{}, log_posterior{};
    std::uint64_t likelihood_seed{};
    bool accepted{};
};
// Pseudo-marginal MH: retain the current likelihood estimate on rejection.
std::vector<ParticleSample> particle_metropolis(const Dataset &data, Vector initial,
                                                double proposal_scale, std::size_t samples,
                                                std::size_t particles, std::uint64_t seed);
double stochastic_log_prior(const Vector &log_parameters);
struct EnsembleFilterRecord {
    std::size_t step{};
    double forecast_sensor_rmse{}, posterior_spread{};
    Field mean_vorticity;
    bool is_forecast = false;
};
struct EnKFOptions {
    // Multiplicative anomaly inflation applied once before each observation time.
    double inflation = 1.0;
    // Gaspari--Cohn half-support radius in grid cells; zero disables localization.
    double localization_radius = 0.0;
    std::size_t forecast_steps = 0;
    std::size_t forecast_every = 1;
};
// Serial perturbed-observation EnKF on joint vorticity and OU forcing state.
std::vector<EnsembleFilterRecord> ensemble_kalman_filter(const Dataset &data, Parameters p,
                                                         std::size_t members, std::uint64_t seed,
                                                         EnKFOptions options = {});
} // namespace hzl
