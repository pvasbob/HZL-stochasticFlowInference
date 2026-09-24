#pragma once
#include "hzl/inference.hpp"

namespace hzl {
struct CoupledEnergy {
    double full{}, linear{};
};
// Linear Stokes control driven by exactly the same midpoint OU coefficients.
CoupledEnergy coupled_energy(std::size_t n, Parameters p, double dt, std::size_t steps,
                             std::uint64_t seed);
double expected_linear_energy(std::size_t n, Parameters p, double dt, std::size_t steps);
struct ControlVariateResult {
    double beta{}, known_control_mean{}, raw_mean{}, corrected_mean{}, raw_variance{},
        corrected_variance{}, raw_standard_error{}, corrected_standard_error{}, pilot_seconds{},
        production_seconds{};
    std::vector<CoupledEnergy> production;
};
ControlVariateResult control_variate_experiment(std::size_t n, Parameters p, double dt,
                                                std::size_t steps, std::size_t pilot_count,
                                                std::size_t count, std::uint64_t seed);
struct SplittingLevel {
    std::size_t level{}, trajectories{}, survivors{}, threshold_hits{};
    double time{}, score_cutoff{}, probability_factor{};
};
struct SplittingResult {
    double probability_estimate{};
    bool threshold_reached = false;
    std::vector<SplittingLevel> levels;
};
// Adaptive trajectory splitting for the path maximum of viscous dissipation.
SplittingResult dissipation_splitting(std::size_t n, Parameters p, double dt,
                                      std::size_t segment_steps, std::size_t maximum_levels,
                                      std::size_t particles, double threshold, std::uint64_t seed);
struct AntitheticResult {
    double raw_mean{}, antithetic_mean{}, raw_standard_error{}, antithetic_standard_error{},
        raw_variance{}, paired_variance{}, wall_seconds{};
};
// Equal-cost propagation of Gaussian log-parameter uncertainty into final energy.
AntitheticResult parameter_antithetic_experiment(std::size_t n, Parameters center, double log_sd,
                                                 double dt, std::size_t steps,
                                                 std::size_t pair_count, std::uint64_t seed);
} // namespace hzl
