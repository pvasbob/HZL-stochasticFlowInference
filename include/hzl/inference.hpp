#pragma once
#include "hzl/solver.hpp"
#include <functional>
#include <vector>

namespace hzl {
using Vector = std::vector<double>;
using Objective = std::function<double(const Vector &)>;
struct OptimizationRecord {
    std::size_t iteration{}, evaluations{};
    double value{};
    Vector parameters;
};
struct OptimizationResult {
    Vector parameters;
    double value{};
    std::size_t evaluations{};
    bool converged = false;
    std::string reason;
    std::vector<OptimizationRecord> history;
};
OptimizationResult nelder_mead(const Objective &objective, const Vector &initial,
                               double initial_step = 0.25, std::size_t max_iterations = 200,
                               double tolerance = 1e-6);
Vector central_gradient(const Objective &objective, const Vector &x, double step);
struct ValueGradient {
    double value{};
    Vector gradient;
};
using DifferentiableObjective = std::function<ValueGradient(const Vector &)>;
OptimizationResult bfgs(const DifferentiableObjective &objective, const Vector &initial,
                        std::size_t max_iterations = 100, double gradient_tolerance = 1e-5);
struct Observation {
    std::size_t step{}, x{}, y{};
    double value{}, standard_deviation{};
};
struct Dataset {
    std::size_t n = 16;
    double dt = 0.01;
    bool stochastic = false;
    int forcing_min = 2, forcing_max = 3;
    std::vector<Observation> observations;
    void validate() const;
    void save(const std::filesystem::path &path) const;
    static Dataset load(const std::filesystem::path &path);
};
Dataset synthetic_observations(std::size_t n, double dt, std::size_t steps, std::size_t every,
                               Parameters truth, double noise, std::uint64_t seed,
                               std::vector<Field> *truth_fields = nullptr);
// Initial inverse problem: deterministic dynamics, known multimode initial state,
// independent Gaussian vorticity sensor noise, unknown positive nu and alpha.
Vector predict(const Dataset &data, const Vector &log_parameters);
double negative_log_likelihood(const Dataset &data, const Vector &log_parameters);
ValueGradient likelihood_gradient(const Dataset &data, const Vector &log_parameters);
struct AdjointEvaluation {
    ValueGradient objective;
    std::size_t checkpoint_fields = 0, peak_block_fields = 0, replayed_steps = 0;
};
AdjointEvaluation likelihood_adjoint(const Dataset &data, const Vector &log_parameters,
                                     std::size_t checkpoint_stride = 20);
struct Sample {
    Vector parameters;
    double log_density{};
    bool accepted{};
    bool divergent = false;
    double energy_error = 0;
};
std::vector<Sample> metropolis(const Objective &log_density, Vector initial,
                               const Vector &proposal_scales, std::size_t count,
                               std::uint64_t seed);
std::vector<Sample> adaptive_metropolis(const Objective &log_density, Vector initial,
                                        double initial_scale, std::size_t count, std::size_t warmup,
                                        std::uint64_t seed);
std::vector<Sample> hamiltonian_monte_carlo(const DifferentiableObjective &negative_log_density,
                                            Vector initial, double step_size,
                                            std::size_t leapfrog_steps, std::size_t count,
                                            std::uint64_t seed);
struct Moments {
    std::size_t count = 0;
    double mean = 0, m2 = 0;
    void add(double x);
    [[nodiscard]] double variance() const;
    [[nodiscard]] double standard_error() const;
};
std::vector<Diagnostics> ensemble(std::size_t n, Parameters p, double dt, std::size_t steps,
                                  std::size_t count, std::size_t workers, std::uint64_t seed);
} // namespace hzl
