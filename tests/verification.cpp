#include "hzl/filter.hpp"
#include "hzl/inference.hpp"
#include "hzl/monte_carlo.hpp"
#include "hzl/reduced.hpp"
#include "hzl/surrogate.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
double error(const hzl::Field &a, const hzl::Field &b) {
    double numerator = 0, denominator = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        numerator += std::norm(a[i] - b[i]);
        denominator += std::norm(b[i]);
    }
    return std::sqrt(numerator / std::max(denominator, 1e-30));
}
void fourier() {
    constexpr std::size_t n = 32;
    hzl::Fourier2D fft(n);
    hzl::Field field(n * n);
    for (std::size_t y = 0; y < n; ++y)
        for (std::size_t x = 0; x < n; ++x)
            field[y * n + x] = std::sin(3 * 2 * std::numbers::pi * x / n) +
                               0.7 * std::cos(2 * 2 * std::numbers::pi * y / n);
    const auto original = field;
    fft.forward(field);
    require(std::abs(field[3] - hzl::Complex{0, -0.5}) < 1e-14,
            "Fourier coefficient normalization");
    fft.inverse(field);
    require(error(field, original) < 1e-14, "FFT round trip");
    std::cout << "PASS Fourier normalization and analytic modes\n";
}
void decay() {
    hzl::Parameters p;
    p.viscosity = 0.7;
    p.friction = 0.6;
    p.forcing_sigma = 0;
    std::vector<double> errors;
    for (double dt : {0.04, 0.02, 0.01}) {
        hzl::Solver solver(16, p, 1);
        solver.initialize("taylor-green");
        auto exact = solver.coefficients();
        for (auto &z : exact)
            z *= std::exp(-2.0 * 0.4);
        for (int i = 0; i < static_cast<int>(std::lround(0.4 / dt)); ++i)
            solver.step(dt);
        errors.push_back(error(solver.coefficients(), exact));
        const auto d = solver.diagnostics();
        const double expected = std::numbers::pi * std::numbers::pi * std::exp(-4.0 * 0.4);
        require(std::abs(d.energy / expected - 1) < 1e-6, "Taylor-Green energy normalization");
        require(std::abs(d.dissipation - 2 * p.viscosity * d.enstrophy) < 1e-13,
                "dissipation identity");
    }
    require(errors[0] / errors[1] > 14 && errors[1] / errors[2] > 14, "RK4 temporal convergence");
    std::vector<double> ssp_errors;
    for (double dt : {0.04, 0.02, 0.01}) {
        hzl::Solver solver(16, p, 1);
        solver.initialize("taylor-green");
        auto exact = solver.coefficients();
        for (auto &z : exact)
            z *= std::exp(-2.0 * 0.4);
        for (int i = 0; i < static_cast<int>(std::lround(0.4 / dt)); ++i)
            solver.step(dt, hzl::TimeIntegrator::ssprk3);
        ssp_errors.push_back(error(solver.coefficients(), exact));
    }
    require(ssp_errors[0] / ssp_errors[1] > 7 && ssp_errors[1] / ssp_errors[2] > 7,
            "SSPRK3 temporal convergence");
    require(hzl::parse_integrator("rk4") == hzl::TimeIntegrator::rk4 &&
                hzl::parse_integrator("ssprk3") == hzl::TimeIntegrator::ssprk3,
            "time integrator parsing");
    std::cout << "PASS Taylor-Green decay; temporal errors=" << errors[0] << ',' << errors[1] << ','
              << errors[2] << "; SSPRK3=" << ssp_errors[0] << ',' << ssp_errors[1] << ','
              << ssp_errors[2] << '\n';
}
void nonlinear() {
    hzl::Parameters p;
    p.viscosity = 0;
    p.friction = 0;
    p.forcing_sigma = 0;
    hzl::Solver solver(32, p, 1);
    solver.initialize("multimode");
    const auto before = solver.diagnostics();
    for (int i = 0; i < 100; ++i)
        solver.step(0.005);
    const auto after = solver.diagnostics();
    require(std::abs(after.energy / before.energy - 1) < 1e-9, "inviscid energy conservation");
    require(std::abs(after.enstrophy / before.enstrophy - 1) < 1e-9,
            "inviscid enstrophy conservation");
    const auto bins = solver.energy_spectrum();
    require(std::abs(std::accumulate(bins.begin(), bins.end(), 0.0) - after.energy) < 1e-12,
            "spectrum energy sum");
    for (auto z : solver.physical_vorticity())
        require(std::abs(z.imag()) < 1e-12, "real physical vorticity");
    std::cout << "PASS nonlinear inviscid energy/enstrophy conservation; relative drift="
              << after.energy / before.energy - 1 << ',' << after.enstrophy / before.enstrophy - 1
              << '\n';
}
void stochastic() {
    hzl::Parameters p;
    p.forcing_sigma = 0.5;
    p.forcing_tau = 0.3;
    hzl::Moments initial, final;
    double cross = 0;
    constexpr int count = 2048;
    for (int j = 0; j < count; ++j) {
        hzl::Solver solver(16, p, hzl::stream_seed(123, static_cast<std::uint64_t>(j)));
        const double a = solver.forcing()[2].real();
        solver.step(0.05);
        const double b = solver.forcing()[2].real();
        initial.add(a);
        final.add(b);
        cross += a * b;
        if (j == 0)
            for (auto z : solver.physical_vorticity())
                require(std::abs(z.imag()) < 1e-12, "forcing conjugate symmetry");
    }
    const double variance = p.forcing_sigma * p.forcing_sigma * p.forcing_tau / 4;
    // About 5 standard errors for a Gaussian sample variance at this sample count.
    const double tolerance = 5 * std::sqrt(2.0 / (count - 1));
    require(std::abs(initial.variance() / variance - 1) < tolerance,
            "OU initial stationary variance");
    require(std::abs(final.variance() / variance - 1) < tolerance,
            "OU evolved stationary variance");
    const double correlation = (cross / count - initial.mean * final.mean) /
                               std::sqrt(initial.variance() * final.variance());
    require(std::abs(correlation - std::exp(-0.05 / p.forcing_tau)) < 0.04,
            "OU transition correlation");
    std::cout << "PASS OU stationary variance and transition correlation=" << correlation << '\n';
}
void checkpoint() {
    hzl::Solver solver(16, {}, 123);
    solver.initialize("multimode");
    for (int i = 0; i < 7; ++i)
        solver.step(0.01);
    const auto path =
        std::filesystem::temp_directory_path() /
        ("hzl-checkpoint-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".txt");
    solver.save(path);
    auto restored = hzl::Solver::load(path);
    std::filesystem::remove(path);
    for (int i = 0; i < 5; ++i) {
        solver.step(0.01);
        restored.step(0.01);
    }
    require(solver.coefficients() == restored.coefficients() &&
                solver.forcing() == restored.forcing(),
            "checkpoint exact continuation");
    const auto a = hzl::ensemble(16, {}, 0.01, 4, 6, 1, 77),
               b = hzl::ensemble(16, {}, 0.01, 4, 6, 3, 77);
    for (std::size_t i = 0; i < a.size(); ++i)
        require(a[i].energy == b[i].energy, "ensemble worker independence");
    std::cout << "PASS exact checkpoint continuation and scheduling-independent ensembles\n";
}
void inference() {
    const hzl::Objective rosenbrock = [](const auto &x) {
        return std::pow(1 - x[0], 2) + 100 * std::pow(x[1] - x[0] * x[0], 2);
    };
    const auto optimum = hzl::nelder_mead(rosenbrock, {-1.2, 1}, 0.3, 500, 1e-7);
    require(optimum.converged && optimum.value < 1e-12, "Nelder-Mead Rosenbrock recovery");
    const auto gradient = hzl::central_gradient(rosenbrock, {-0.5, 0.8}, 1e-5);
    require(std::abs(gradient[0] - 107) < 1e-6 && std::abs(gradient[1] - 110) < 1e-6,
            "finite difference analytic gradient comparison");
    const auto samples =
        hzl::metropolis([](const auto &x) { return -0.5 * x[0] * x[0]; }, {0}, {1.5}, 50000, 42);
    hzl::Moments moments;
    for (std::size_t i = 5000; i < samples.size(); ++i)
        moments.add(samples[i].parameters[0]);
    require(std::abs(moments.mean) < 0.06 && std::abs(moments.variance() - 1) < 0.08,
            "Metropolis Gaussian target");
    std::cout << "PASS optimizer, analytic gradient comparison, Gaussian posterior sampler\n";
}
void particles() {
    hzl::Parameters p;
    p.viscosity = 0.03;
    p.friction = 0.1;
    p.forcing_sigma = 0;
    const auto deterministic = hzl::synthetic_observations(16, 0.01, 30, 10, p, 0.2, 987);
    const auto result = hzl::bootstrap_filter(deterministic, p, 8, 555);
    double expected =
        -hzl::negative_log_likelihood(deterministic, {std::log(p.viscosity), std::log(p.friction)});
    for (const auto &o : deterministic.observations)
        expected -= std::log(o.standard_deviation) + 0.5 * std::log(2 * std::numbers::pi);
    require(std::abs(result.log_likelihood - expected) < 1e-10,
            "particle likelihood deterministic limit");
    require(result.resampling_events == 0, "identical particles do not need resampling");
    const hzl::Vector deterministic_q{std::log(p.viscosity), std::log(p.friction), std::log(1e-8),
                                      std::log(p.forcing_tau)};
    const double single = hzl::particle_log_likelihood(deterministic, deterministic_q, 8, 555),
                 averaged_one = hzl::averaged_particle_log_likelihood(deterministic,
                                                                      deterministic_q, 8, 1, 555);
    require(single == averaged_one, "one-replicate averaged particle likelihood identity");
    const auto kalman = hzl::ensemble_kalman_filter(deterministic, p, 8, 555);
    require(kalman.size() == result.history.size(), "EnKF observation times");
    for (std::size_t i = 0; i < kalman.size(); ++i) {
        require(error(kalman[i].mean_vorticity, result.history[i].mean_vorticity) < 1e-12,
                "EnKF deterministic limit");
        require(kalman[i].posterior_spread < 1e-12, "EnKF identical-state spread");
    }
    p.forcing_sigma = 0.2;
    const auto data = hzl::synthetic_observations(16, 0.01, 30, 10, p, 0.2, 987);
    const auto first = hzl::bootstrap_filter(data, p, 32, 555),
               second = hzl::bootstrap_filter(data, p, 32, 555);
    require(first.log_likelihood == second.log_likelihood, "particle filter reproducibility");
    for (const auto &r : first.history)
        require(r.effective_particles >= 1 - 1e-10 && r.effective_particles <= 32 + 1e-10,
                "particle ESS bounds");
    const auto localized = hzl::ensemble_kalman_filter(data, p, 12, 99, {1.02, 4.0, 2, 1});
    require(localized.size() == 5 && localized[3].is_forecast && localized[4].is_forecast,
            "localized inflated EnKF forecast horizon");
    for (const auto &record : localized)
        for (const auto &value : record.mean_vorticity)
            require(std::isfinite(value.real()), "finite localized EnKF state");
    const auto chain = hzl::particle_metropolis(
        data, {std::log(0.03), std::log(0.1), std::log(0.2), std::log(0.5)}, 0.5, 16, 8, 42);
    std::size_t rejections = 0;
    for (std::size_t i = 1; i < chain.size(); ++i)
        if (!chain[i].accepted) {
            ++rejections;
            require(chain[i].likelihood_seed == chain[i - 1].likelihood_seed &&
                        chain[i].log_likelihood == chain[i - 1].log_likelihood &&
                        chain[i].parameters == chain[i - 1].parameters,
                    "pseudo-marginal rejection retains likelihood estimate");
        }
    require(rejections > 0, "exercise pseudo-marginal rejection");
    std::cout << "PASS particle likelihood deterministic limit, filter reproducibility, and "
                 "pseudo-marginal rejection\n";
}
void tangent() {
    hzl::Parameters p;
    p.forcing_sigma = 0;
    const auto data = hzl::synthetic_observations(16, 0.01, 30, 10, p, 0.02, 123);
    const hzl::Vector q{std::log(0.03), std::log(0.2)};
    const auto exact = hzl::likelihood_gradient(data, q);
    for (std::size_t stride : {1U, 7U, 100U}) {
        const auto adjoint = hzl::likelihood_adjoint(data, q, stride);
        require(std::abs(adjoint.objective.value - exact.value) < 1e-10,
                "adjoint objective matches forward solve");
        for (std::size_t j = 0; j < 2; ++j)
            require(std::abs(adjoint.objective.gradient[j] - exact.gradient[j]) <
                        1e-9 * (1 + std::abs(exact.gradient[j])),
                    "checkpointed discrete adjoint matches tangent gradient");
    }
    const auto finite = hzl::central_gradient(
        [&](const auto &x) { return hzl::negative_log_likelihood(data, x); }, q, 1e-5);
    require(std::abs(exact.value - hzl::negative_log_likelihood(data, q)) < 1e-10,
            "tangent state matches standard RK4");
    for (std::size_t j = 0; j < 2; ++j)
        require(std::abs(exact.gradient[j] - finite[j]) < 1e-6 * (1 + std::abs(exact.gradient[j])),
                "full PDE likelihood gradient");
    const auto result = hzl::bfgs(
        [](const auto &x) {
            return hzl::ValueGradient{
                std::pow(1 - x[0], 2) + 100 * std::pow(x[1] - x[0] * x[0], 2),
                {2 * (x[0] - 1) - 400 * x[0] * (x[1] - x[0] * x[0]), 200 * (x[1] - x[0] * x[0])}};
        },
        {-1.2, 1}, 200, 1e-6);
    require(result.converged && result.value < 1e-12, "BFGS Rosenbrock recovery");
    std::cout << "PASS differentiated RK4 PDE gradient and BFGS optimizer\n";
}
void reduced() {
    std::vector<hzl::Field> analytic;
    for (int i = 0; i < 30; ++i) {
        const double t = 2 * std::numbers::pi * i / 30;
        analytic.push_back({1.0 + std::sin(t), 2.0 + 2 * std::cos(t), 3.0});
    }
    const auto exact = hzl::fit_pod(analytic, 3, 1);
    require(exact.modes.size() == 2, "POD identifies rank-two snapshots");
    for (const auto &state : analytic)
        require(error(exact.reconstruct(exact.project(state)), state) < 1e-12,
                "POD low-rank reconstruction");
    hzl::Parameters p;
    p.forcing_sigma = 0;
    hzl::Solver full(16, p, 1);
    full.initialize("multimode");
    std::vector<hzl::Field> snapshots;
    for (int i = 0; i < 20; ++i) {
        snapshots.push_back(full.coefficients());
        full.step(0.01);
    }
    const auto basis = hzl::fit_pod(snapshots, 4, 0.9999999);
    hzl::GalerkinModel model(basis, full);
    auto a = basis.project(full.coefficients());
    for (auto &x : a)
        x += 0.01;
    const auto drift = full.drift(basis.reconstruct(a));
    const auto modal = model.rhs(a);
    for (std::size_t i = 0; i < a.size(); ++i) {
        require(std::abs(modal[i] - hzl::spectral_inner(basis.modes[i], drift)) < 1e-10,
                "Galerkin tensor matches projected PDE drift");
        for (std::size_t j = 0; j < a.size(); ++j)
            require(std::abs(hzl::spectral_inner(basis.modes[i], basis.modes[j]) -
                             (i == j ? 1 : 0)) < 1e-12,
                    "POD orthonormality");
    }
    const auto samples = hzl::adaptive_metropolis(
        [](const auto &x) { return -0.5 * (x[0] * x[0] + std::pow((x[1] - 0.8 * x[0]) / 0.6, 2)); },
        {0, 0}, 0.3, 40000, 2000, 77);
    hzl::Moments first, second;
    for (std::size_t i = 2000; i < samples.size(); ++i) {
        first.add(samples[i].parameters[0]);
        second.add(samples[i].parameters[1]);
    }
    require(std::abs(first.mean) < 0.08 && std::abs(second.mean) < 0.08 &&
                std::abs(first.variance() - 1) < 0.1 && std::abs(second.variance() - 1) < 0.1,
            "adaptive Metropolis correlated Gaussian target");
    const auto hmc = hzl::hamiltonian_monte_carlo(
        [](const auto &x) { return hzl::ValueGradient{0.5 * x[0] * x[0], {x[0]}}; }, {0}, 0.2, 8,
        12000, 77);
    hzl::Moments target;
    std::size_t accepted = 0;
    for (std::size_t i = 1000; i < hmc.size(); ++i) {
        target.add(hmc[i].parameters[0]);
        accepted += hmc[i].accepted;
        require(!hmc[i].divergent, "Gaussian HMC should not diverge");
    }
    require(std::abs(target.mean) < 0.05 && std::abs(target.variance() - 1) < 0.07 &&
                accepted > 10000,
            "HMC Gaussian target and energy control");
    std::cout << "PASS POD rank/reconstruction, projected Galerkin dynamics, and adaptive Gaussian "
                 "sampling\n";
}
void control() {
    hzl::Parameters p;
    p.forcing_min = 2;
    p.forcing_max = 2;
    const auto pair = hzl::coupled_energy(16, p, 0.01, 30, 77);
    require(std::abs(pair.full - pair.linear) < 1e-12,
            "single-shell nonlinear flow matches linear Stokes control");
    p.forcing_max = 3;
    hzl::Solver full(16, p, 555);
    full.initialize("zero");
    hzl::ShellForcing forcing(16, p, 555);
    for (int step = 0; step < 6; ++step) {
        full.step(0.01);
        const auto &midpoint = forcing.midpoint_step(0.01);
        require(midpoint == full.midpoint_forcing() && forcing.coefficients() == full.forcing(),
                "standalone forcing matches solver path exactly");
    }
    hzl::Moments values;
    for (std::uint64_t i = 0; i < 512; ++i)
        values.add(hzl::coupled_energy(16, p, 0.05, 6, hzl::stream_seed(44, i)).linear);
    const double expected = hzl::expected_linear_energy(16, p, 0.05, 6);
    require(std::abs(values.mean - expected) < 5 * values.standard_error(),
            "analytic discrete linear-control mean");
    std::cout << "PASS shared-forcing linear Stokes control and analytic mean\n";
    const auto splitting = hzl::dissipation_splitting(16, p, 0.01, 5, 3, 16, 0.1, 91);
    const auto repeated = hzl::dissipation_splitting(16, p, 0.01, 5, 3, 16, 0.1, 91);
    require(splitting.probability_estimate == repeated.probability_estimate &&
                splitting.probability_estimate >= 0 && splitting.probability_estimate <= 1,
            "trajectory splitting reproducibility and probability bounds");
    hzl::Parameters deterministic;
    deterministic.forcing_sigma = 0;
    const auto antithetic =
        hzl::parameter_antithetic_experiment(16, deterministic, 0.15, 0.01, 5, 8, 12);
    require(std::isfinite(antithetic.antithetic_mean) && antithetic.antithetic_standard_error >= 0,
            "antithetic parameter propagation finite statistics");
}
} // namespace
void surrogate() {
    std::vector<hzl::Vector> inputs;
    hzl::Vector outputs;
    const auto polynomial = [](const auto &x) {
        return 2 + 3 * x[0] - 4 * x[1] + 2 * x[0] * x[0] + 0.5 * x[0] * x[1] - x[1] * x[1];
    };
    for (int i = -2; i <= 2; ++i)
        for (int j = -2; j <= 2; ++j) {
            inputs.push_back({i / 2.0, j / 2.0});
            outputs.push_back(polynomial(inputs.back()));
        }
    const hzl::QuadraticSurrogate model(inputs, outputs, {0, 0}, {1, 1});
    require(model.training_rmse() < 1e-12 &&
                std::abs(model({0.3, -0.7}) - polynomial(hzl::Vector{0.3, -0.7})) < 1e-12,
            "Householder QR quadratic recovery");
    const hzl::GaussianRBFSurrogate rbf(inputs, outputs, {0, 0}, {1, 1}, 0.8, 1e-12);
    require(rbf.training_rmse() < 1e-8, "regularized Gaussian RBF interpolation");
    const auto chain = hzl::delayed_acceptance([](const auto &x) { return -0.5 * x[0] * x[0]; },
                                               [](const auto &x) { return -0.25 * x[0] * x[0]; },
                                               {0}, {1.5}, 40000, 77);
    hzl::Moments moments;
    for (std::size_t i = 2000; i < chain.samples.size(); ++i)
        moments.add(chain.samples[i].parameters[0]);
    require(std::abs(moments.mean) < 0.06 && std::abs(moments.variance() - 1) < 0.1,
            "delayed acceptance corrects inaccurate surrogate");
    require(chain.exact_evaluations < chain.samples.size() &&
                chain.exact_evaluations == chain.stage_one_acceptances + 1,
            "delayed acceptance avoids full evaluations");
    std::cout << "PASS quadratic regression and corrected surrogate sampling\n";
}
int main() {
    try {
        fourier();
        decay();
        nonlinear();
        stochastic();
        checkpoint();
        inference();
        particles();
        tangent();
        reduced();
        control();
        surrogate();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
