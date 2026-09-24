#include "hzl/filter.hpp"
#include "hzl/inference.hpp"
#include "hzl/monte_carlo.hpp"
#include "hzl/reduced.hpp"
#include "hzl/surrogate.hpp"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace {
class Options {
  public:
    Options(int argc, char **argv) : command_(argv[1]) {
        for (int i = 2; i < argc; i += 2) {
            const std::string key = argv[i];
            if (!key.starts_with("--") || i + 1 >= argc || values_.contains(key))
                throw std::invalid_argument("options require unique --name value pairs");
            values_[key] = argv[i + 1];
        }
    }
    std::string get(const std::string &key, const std::string &fallback) {
        used_.insert(key);
        auto i = values_.find(key);
        const auto value = i == values_.end() ? fallback : i->second;
        resolved_[key.substr(2)] = value;
        return value;
    }
    template <class T> T number(const std::string &key, T fallback) {
        const auto text = get(key, std::to_string(fallback));
        T value{};
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (error != std::errc{} || end != text.data() + text.size())
            throw std::invalid_argument("invalid value for " + key);
        if constexpr (std::is_floating_point_v<T>)
            if (!std::isfinite(value))
                throw std::invalid_argument("nonfinite value for " + key);
        return value;
    }
    void finish() const {
        for (const auto &[key, value] : values_)
            if (!used_.contains(key))
                throw std::invalid_argument("unknown option " + key);
    }
    bool supplied(const std::string &key) const {
        return values_.contains(key);
    }
    void record(const std::filesystem::path &path) const {
        const auto quote = [](std::ostream &out, const std::string &value) {
            out << '"';
            for (unsigned char c : value) {
                if (c == '"' || c == '\\')
                    out << '\\' << static_cast<char>(c);
                else if (c < 32)
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned>(c) << std::dec;
                else
                    out << static_cast<char>(c);
            }
            out << '"';
        };
        std::ofstream out(path);
        out << "{\"schema\":1,\"command\":";
        quote(out, command_);
        out << ",\"options\":{";
        bool first = true;
        for (const auto &[key, value] : resolved_) {
            if (!first)
                out << ',';
            first = false;
            quote(out, key);
            out << ':';
            quote(out, value);
        }
        out << "}}\n";
        if (!out)
            throw std::runtime_error("cannot record invocation");
    }

  private:
    std::string command_;
    std::map<std::string, std::string> values_, resolved_;
    std::set<std::string> used_;
};
std::ofstream output(const std::filesystem::path &path) {
    std::ofstream out(path);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << std::setprecision(17);
    return out;
}
std::filesystem::path directory(Options &args, const std::string &name) {
    const std::filesystem::path path = args.get("--output", "results/" + name);
    if (std::filesystem::exists(path) && !std::filesystem::is_empty(path))
        throw std::invalid_argument(
            "output directory is not empty; choose a new --output directory");
    args.finish();
    std::filesystem::create_directories(path);
    args.record(path / "invocation.json");
    return path;
}
hzl::Parameters parameters(Options &args) {
    hzl::Parameters p;
    p.viscosity = args.number("--nu", p.viscosity);
    p.friction = args.number("--alpha", p.friction);
    p.forcing_sigma = args.number("--sigma", p.forcing_sigma);
    p.forcing_tau = args.number("--tau", p.forcing_tau);
    p.forcing_min = args.number("--kmin", p.forcing_min);
    p.forcing_max = args.number("--kmax", p.forcing_max);
    return p;
}
void metadata(const std::filesystem::path &path, const hzl::Solver &solver, double dt,
              std::size_t steps, std::uint64_t seed, const std::string &initial, double seconds,
              double final_time = -1, hzl::TimeIntegrator integrator = hzl::TimeIntegrator::rk4) {
    auto out = output(path / "metadata.json");
    const auto &p = solver.parameters();
    out << "{\n  \"schema\": 1,\n  \"version\": \"0.1.0\",\n  \"backend\": "
           "\"cpu-native-radix2-double\",\n  \"compiler\": \""
        << __VERSION__ << "\",\n"
        << "  \"n\": " << solver.size() << ",\n  \"dt\": " << dt << ",\n  \"integrator\": \""
        << hzl::integrator_name(integrator) << "\""
        << ",\n  \"steps_this_run\": " << steps
        << ",\n  \"seed\": " << (initial == "checkpoint" ? "null" : std::to_string(seed)) << ",\n"
        << "  \"initial\": \"" << initial << "\",\n  \"nu\": " << p.viscosity
        << ",\n  \"alpha\": " << p.friction << ",\n"
        << "  \"sigma\": " << p.forcing_sigma << ",\n  \"tau\": " << p.forcing_tau
        << ",\n  \"kmin\": " << p.forcing_min << ",\n  \"kmax\": " << p.forcing_max << ",\n"
        << "  \"final_time\": " << (final_time < 0 ? solver.time() : final_time)
        << ",\n  \"wall_seconds_including_diagnostics\": " << seconds << "\n}\n";
}
void simulate(Options &args) {
    const auto n = args.number<std::size_t>("--n", 32),
               steps = args.number<std::size_t>("--steps", 1000),
               every = args.number<std::size_t>("--every", 10);
    const double dt = args.number("--dt", 0.005);
    const auto seed = args.number<std::uint64_t>("--seed", 42);
    const auto integrator = hzl::parse_integrator(args.get("--integrator", "rk4"));
    const auto initial = args.get("--initial", "multimode"), restart = args.get("--restart", "");
    const auto p = parameters(args);
    if (!restart.empty())
        for (const auto *key : {"--n", "--nu", "--alpha", "--sigma", "--tau", "--kmin", "--kmax",
                                "--seed", "--initial"})
            if (args.supplied(key))
                throw std::invalid_argument(std::string("checkpoint supplies ") + key +
                                            "; omit this option when restarting");
    const auto path = directory(args, "simulation");
    args.finish();
    if (every == 0 || steps == 0)
        throw std::invalid_argument("steps and every must be positive");
    auto solver = restart.empty() ? hzl::Solver(n, p, seed) : hzl::Solver::load(restart);
    if (!restart.empty())
        std::filesystem::copy_file(restart, path / "restart_input.txt");
    if (restart.empty())
        solver.initialize(initial);
    auto out = output(path / "diagnostics.csv");
    out << "time,energy,enstrophy,dissipation,injection,cfl\n";
    const auto record = [&]() {
        const auto d = solver.diagnostics();
        out << d.time << ',' << d.energy << ',' << d.enstrophy << ',' << d.dissipation << ','
            << d.injection << ',' << dt * d.cfl_rate << '\n';
    };
    const auto start = std::chrono::steady_clock::now();
    record();
    for (std::size_t step = 1; step <= steps; ++step) {
        solver.step(dt, integrator);
        if (step % every == 0 || step == steps)
            record();
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    solver.save(path / "checkpoint.txt");
    const auto field = solver.physical_vorticity();
    auto fields = output(path / "vorticity.csv");
    fields << "ix,iy,vorticity\n";
    for (std::size_t y = 0; y < solver.size(); ++y)
        for (std::size_t x = 0; x < solver.size(); ++x)
            fields << x << ',' << y << ',' << field[y * solver.size() + x].real() << '\n';
    auto spectrum = output(path / "spectrum.csv");
    spectrum << "k,energy\n";
    const auto bins = solver.energy_spectrum();
    for (std::size_t k = 0; k < bins.size(); ++k)
        spectrum << k << ',' << bins[k] << '\n';
    metadata(path, solver, dt, steps, seed, restart.empty() ? initial : "checkpoint", seconds, -1,
             integrator);
    std::cout << "simulation complete: " << path << "; " << seconds << " seconds\n";
}
void observe(Options &args) {
    const auto n = args.number<std::size_t>("--n", 16),
               steps = args.number<std::size_t>("--steps", 150),
               every = args.number<std::size_t>("--every", 10);
    const double dt = args.number("--dt", 0.01), noise = args.number("--noise", 0.02);
    const auto seed = args.number<std::uint64_t>("--seed", 314159);
    auto p = parameters(args);
    const auto path = directory(args, "observations");
    args.finish();
    std::vector<hzl::Field> truth_fields;
    auto data = hzl::synthetic_observations(n, dt, steps, every, p, noise, seed, &truth_fields);
    auto train = data, heldout = data;
    train.observations.clear();
    heldout.observations.clear();
    const auto split = steps * 2 / 3;
    for (const auto &o : data.observations)
        (o.step <= split ? train.observations : heldout.observations).push_back(o);
    train.save(path / "training.csv");
    heldout.save(path / "validation.csv");
    data.save(path / "observations.csv");
    auto fields = output(path / "truth_fields.csv");
    fields << "step,ix,iy,vorticity\n";
    for (std::size_t j = 0; j < truth_fields.size(); ++j)
        for (std::size_t y = 0; y < n; ++y)
            for (std::size_t x = 0; x < n; ++x)
                fields << (j + 1) * every << ',' << x << ',' << y << ','
                       << truth_fields[j][y * n + x].real() << '\n';
    auto truth = output(path / "truth.json");
    truth << "{\"nu\":" << p.viscosity << ",\"alpha\":" << p.friction
          << ",\"sigma\":" << p.forcing_sigma << ",\"tau\":" << p.forcing_tau
          << ",\"seed\":" << seed << ",\"noise_sd\":" << noise << ",\"initial\":\"multimode\"}\n";
    std::cout << "observations complete: " << path
              << "; truth is stored separately from calibration inputs\n";
}
void calibrate(Options &args) {
    const auto input = args.get("--data", "results/observations/training.csv"),
               validation = args.get("--validation", "");
    const double nu = args.number("--initial-nu", 0.08),
                 alpha = args.number("--initial-alpha", 0.3);
    const auto method = args.get("--method", "nelder-mead");
    const auto gradient_method = args.get("--gradient", "tangent");
    const auto checkpoint_stride = args.number<std::size_t>("--checkpoint-stride", 20);
    const auto iterations = args.number<std::size_t>("--iterations", 160);
    const auto path = directory(args, "calibration");
    args.finish();
    if (nu <= 0 || alpha <= 0)
        throw std::invalid_argument("initial parameters must be positive");
    const auto data = hzl::Dataset::load(input);
    data.save(path / "training.csv");
    const hzl::Objective objective = [&](const auto &q) {
        return hzl::negative_log_likelihood(data, q);
    };
    const hzl::Vector initial{std::log(nu), std::log(alpha)};
    if (method != "nelder-mead" && method != "bfgs")
        throw std::invalid_argument("method must be nelder-mead or bfgs");
    if (gradient_method != "tangent" && gradient_method != "adjoint")
        throw std::invalid_argument("gradient must be tangent or adjoint");
    const hzl::DifferentiableObjective differentiated = [&](const auto &q) {
        return gradient_method == "adjoint"
                   ? hzl::likelihood_adjoint(data, q, checkpoint_stride).objective
                   : hzl::likelihood_gradient(data, q);
    };
    const auto start = std::chrono::steady_clock::now();
    if (method == "bfgs") {
        const auto exact = differentiated(initial);
        const auto finite = hzl::central_gradient(objective, initial, 1e-4);
        auto checks = output(path / "initial_gradient_check.csv");
        checks << "parameter,computed_gradient,finite_difference\n";
        for (std::size_t j = 0; j < 2; ++j) {
            checks << j << ',' << exact.gradient[j] << ',' << finite[j] << '\n';
            if (std::abs(exact.gradient[j] - finite[j]) > 1e-5 * (1 + std::abs(exact.gradient[j])))
                throw std::runtime_error("PDE gradient check failed before BFGS");
        }
    }
    const auto result = method == "bfgs" ? hzl::bfgs(differentiated, initial, iterations)
                                         : hzl::nelder_mead(objective, initial, 0.3, iterations);
    const auto gradient = hzl::central_gradient(objective, result.parameters, 1e-4);
    auto history = output(path / "history.csv");
    history << "iteration,evaluations,nll,nu,alpha\n";
    for (const auto &r : result.history)
        history << r.iteration << ',' << r.evaluations << ',' << r.value << ','
                << std::exp(r.parameters[0]) << ',' << std::exp(r.parameters[1]) << '\n';
    auto predictions = output(path / "predictions.csv");
    predictions << "split,step,x,y,observed,predicted,standard_deviation\n";
    const auto write_predictions = [&](const hzl::Dataset &dataset, const std::string &split) {
        const auto values = hzl::predict(dataset, result.parameters);
        for (std::size_t i = 0; i < values.size(); ++i) {
            const auto &o = dataset.observations[i];
            predictions << split << ',' << o.step << ',' << o.x << ',' << o.y << ',' << o.value
                        << ',' << values[i] << ',' << o.standard_deviation << '\n';
        }
    };
    write_predictions(data, "training");
    double heldout = 0;
    if (!validation.empty()) {
        const auto test = hzl::Dataset::load(validation);
        test.save(path / "validation.csv");
        write_predictions(test, "validation");
        heldout = hzl::negative_log_likelihood(test, result.parameters);
    }
    auto summary = output(path / "fit.json");
    summary << "{\n\"nu\":" << std::exp(result.parameters[0])
            << ",\n\"alpha\":" << std::exp(result.parameters[1])
            << ",\n\"initial_nll\":" << objective(initial) << ",\n\"final_nll\":" << result.value
            << ",\n\"converged\":" << (result.converged ? "true" : "false") << ",\n\"method\":\""
            << method << "\",\n\"reason\":\"" << result.reason
            << "\",\n\"optimizer_evaluations\":" << result.evaluations
            << ",\n\"gradient_log_nu\":" << gradient[0]
            << ",\n\"gradient_log_alpha\":" << gradient[1]
            << ",\n\"validation_nll\":" << (validation.empty() ? "null" : std::to_string(heldout))
            << ",\n\"wall_seconds\":"
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
            << "\n}\n";
    std::cout << "calibration " << (result.converged ? "converged" : "stopped")
              << ": nu=" << std::exp(result.parameters[0])
              << ", alpha=" << std::exp(result.parameters[1]) << ", nll=" << result.value << "; "
              << result.reason << '\n';
}
void sample(Options &args) {
    const auto input = args.get("--data", "results/observations/training.csv");
    const auto count = args.number<std::size_t>("--samples", 1000);
    const auto seed = args.number<std::uint64_t>("--seed", 123);
    const double nu = args.number("--initial-nu", 0.03),
                 alpha = args.number("--initial-alpha", 0.1),
                 scale = args.number("--proposal", 0.03);
    const auto method = args.get("--method", "random-walk");
    const auto warmup = args.number<std::size_t>("--warmup", 500);
    const double hmc_step = args.number("--step-size", 0.005);
    const auto leapfrog = args.number<std::size_t>("--leapfrog", 10);
    const auto path = directory(args, "posterior");
    args.finish();
    if (nu <= 0 || alpha <= 0)
        throw std::invalid_argument("initial parameters must be positive");
    const auto data = hzl::Dataset::load(input);
    data.save(path / "training.csv");
    const hzl::Objective density = [&](const auto &q) {
        const double a = (q[0] - std::log(0.03)) / 1.5, b = (q[1] - std::log(0.1)) / 1.5;
        return -hzl::negative_log_likelihood(data, q) - 0.5 * (a * a + b * b);
    };
    if (method != "random-walk" && method != "adaptive" && method != "hmc")
        throw std::invalid_argument("sampler method must be random-walk, adaptive, or hmc");
    const hzl::Vector initial{std::log(nu), std::log(alpha)};
    std::vector<hzl::Sample> chain;
    if (method == "hmc") {
        const hzl::DifferentiableObjective potential = [&](const auto &q) {
            auto result = hzl::likelihood_gradient(data, q);
            const double center[] = {std::log(0.03), std::log(0.1)};
            for (std::size_t j = 0; j < 2; ++j) {
                const double delta = q[j] - center[j];
                result.value += 0.5 * delta * delta / 2.25;
                result.gradient[j] += delta / 2.25;
            }
            return result;
        };
        const auto exact = potential(initial);
        const auto finite =
            hzl::central_gradient([&](const auto &q) { return -density(q); }, initial, 1e-4);
        for (std::size_t j = 0; j < 2; ++j)
            if (std::abs(exact.gradient[j] - finite[j]) > 1e-5 * (1 + std::abs(exact.gradient[j])))
                throw std::runtime_error("posterior gradient check failed before HMC");
        chain = hzl::hamiltonian_monte_carlo(potential, initial, hmc_step, leapfrog, count, seed);
    } else if (method == "adaptive")
        chain = hzl::adaptive_metropolis(density, initial, scale, count, warmup, seed);
    else
        chain = hzl::metropolis(density, initial, {scale, scale}, count, seed);
    auto out = output(path / "chain.csv");
    out << "iteration,nu,alpha,log_posterior,accepted,divergent,energy_error\n";
    std::size_t accepted = 0;
    for (std::size_t i = 0; i < chain.size(); ++i) {
        const auto &s = chain[i];
        accepted += s.accepted;
        out << i << ',' << std::exp(s.parameters[0]) << ',' << std::exp(s.parameters[1]) << ','
            << s.log_density << ',' << s.accepted << ',' << s.divergent << ',' << s.energy_error
            << '\n';
    }
    auto meta = output(path / "metadata.json");
    meta << "{\"seed\":" << seed << ",\"samples\":" << count << ",\"method\":\"" << method
         << "\",\"adaptation_warmup\":" << (method == "adaptive" ? warmup : 0)
         << ",\"proposal_log_sd\":" << scale << ",\"initial_nu\":" << nu
         << ",\"initial_alpha\":" << alpha << ",\"prior_log_nu_mean\":" << std::log(0.03)
         << ",\"prior_log_alpha_mean\":" << std::log(0.1)
         << ",\"prior_log_sd\":1.5,\"acceptance_rate\":"
         << static_cast<double>(accepted) / static_cast<double>(count) << "}\n";
    std::cout
        << "chain written: " << path
        << "; assess burn-in and multi-chain convergence before interpreting posterior estimates\n";
}
void ensembles(Options &args) {
    const auto n = args.number<std::size_t>("--n", 32),
               steps = args.number<std::size_t>("--steps", 200),
               count = args.number<std::size_t>("--count", 16),
               workers = args.number<std::size_t>("--workers", 4);
    const double dt = args.number("--dt", 0.005);
    const auto seed = args.number<std::uint64_t>("--seed", 42);
    const auto p = parameters(args);
    const auto path = directory(args, "ensemble");
    args.finish();
    const auto start = std::chrono::steady_clock::now();
    const auto results = hzl::ensemble(n, p, dt, steps, count, workers, seed);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    auto out = output(path / "members.csv");
    out << "member,seed,time,energy,enstrophy,dissipation\n";
    hzl::Moments energy;
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto &d = results[i];
        energy.add(d.energy);
        out << i << ',' << hzl::stream_seed(seed, i) << ',' << d.time << ',' << d.energy << ','
            << d.enstrophy << ',' << d.dissipation << '\n';
    }
    hzl::Solver prototype(n, p, seed);
    metadata(path, prototype, dt, steps, seed, "zero", seconds, dt * static_cast<double>(steps));
    auto summary = output(path / "summary.json");
    summary << "{\"count\":" << count << ",\"workers\":" << workers
            << ",\"mean_energy\":" << energy.mean << ",\"energy_standard_error\":"
            << (count > 1 ? std::to_string(energy.standard_error()) : "null")
            << ",\"wall_seconds\":" << seconds << "}\n";
    std::cout << "ensemble complete: " << path << "; energy mean=" << energy.mean
              << ", SE=" << energy.standard_error() << ", seconds=" << seconds << '\n';
}
void filter(Options &args) {
    const auto input = args.get("--data", "results/observations/observations.csv");
    const auto count = args.number<std::size_t>("--particles", 64);
    const auto method = args.get("--method", "particle");
    const auto seed = args.number<std::uint64_t>("--seed", 2026);
    const hzl::EnKFOptions enkf_options{args.number("--inflation", 1.0),
                                        args.number("--localization-radius", 0.0),
                                        args.number<std::size_t>("--forecast-steps", 0),
                                        args.number<std::size_t>("--forecast-every", 1)};
    const auto p = parameters(args);
    const auto path = directory(args, "filter");
    args.finish();
    const auto data = hzl::Dataset::load(input);
    data.save(path / "observations.csv");
    if (method == "enkf") {
        const auto start = std::chrono::steady_clock::now();
        const auto history = hzl::ensemble_kalman_filter(data, p, count, seed, enkf_options);
        auto diagnostics = output(path / "filter.csv"),
             fields = output(path / "reconstruction.csv");
        diagnostics << "step,phase,forecast_sensor_rmse,posterior_spread\n";
        fields << "step,phase,ix,iy,vorticity\n";
        for (const auto &r : history) {
            diagnostics << r.step << ',' << (r.is_forecast ? "forecast" : "analysis") << ',';
            if (!r.is_forecast)
                diagnostics << r.forecast_sensor_rmse;
            diagnostics << ',' << r.posterior_spread << '\n';
            for (std::size_t y = 0; y < data.n; ++y)
                for (std::size_t x = 0; x < data.n; ++x)
                    fields << r.step << ',' << (r.is_forecast ? "forecast" : "analysis") << ',' << x
                           << ',' << y << ',' << r.mean_vorticity[y * data.n + x].real() << '\n';
        }
        auto summary = output(path / "summary.json");
        summary << "{\"method\":\"serial perturbed-observation EnKF\",\"members\":" << count
                << ",\"seed\":" << seed << ",\"inflation\":" << enkf_options.inflation
                << ",\"localization_radius_grid_cells\":" << enkf_options.localization_radius
                << ",\"forecast_steps\":" << enkf_options.forecast_steps
                << ",\"forecast_every\":" << enkf_options.forecast_every << ",\"wall_seconds\":"
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                << "}\n";
        std::cout << "ensemble Kalman reconstruction complete: " << path << '\n';
        return;
    }
    if (method != "particle")
        throw std::invalid_argument("filter method must be particle or enkf");
    const auto start = std::chrono::steady_clock::now();
    const auto result = hzl::bootstrap_filter(data, p, count, seed);
    if (!std::isfinite(result.log_likelihood))
        throw std::runtime_error("particle likelihood collapsed numerically");
    auto diagnostics = output(path / "filter.csv"), fields = output(path / "reconstruction.csv");
    diagnostics << "step,log_likelihood_increment,effective_particles,maximum_weight\n";
    fields << "step,ix,iy,vorticity\n";
    for (const auto &r : result.history) {
        diagnostics << r.step << ',' << r.log_likelihood_increment << ',' << r.effective_particles
                    << ',' << r.maximum_weight << '\n';
        for (std::size_t y = 0; y < data.n; ++y)
            for (std::size_t x = 0; x < data.n; ++x)
                fields << r.step << ',' << x << ',' << y << ','
                       << r.mean_vorticity[y * data.n + x].real() << '\n';
    }
    auto summary = output(path / "summary.json");
    summary << "{\"particles\":" << count << ",\"seed\":" << seed
            << ",\"log_likelihood_estimate\":" << result.log_likelihood
            << ",\"resampling_events\":" << result.resampling_events << ",\"wall_seconds\":"
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
            << "}\n";
    std::cout << "particle filter complete: " << path
              << "; log likelihood estimate=" << result.log_likelihood
              << ", resampling events=" << result.resampling_events << '\n';
}
hzl::Vector stochastic_initial(Options &args) {
    const double nu = args.number("--initial-nu", 0.03),
                 alpha = args.number("--initial-alpha", 0.1),
                 sigma = args.number("--initial-sigma", 0.2),
                 tau = args.number("--initial-tau", 0.5);
    if (nu <= 0 || alpha <= 0 || sigma <= 0 || tau <= 0)
        throw std::invalid_argument("initial stochastic parameters must be positive");
    return {std::log(nu), std::log(alpha), std::log(sigma), std::log(tau)};
}
void particle_sample(Options &args) {
    const auto input = args.get("--data", "results/observations/training.csv");
    const auto count = args.number<std::size_t>("--samples", 500),
               particles = args.number<std::size_t>("--particles", 32);
    const auto seed = args.number<std::uint64_t>("--seed", 123);
    const double scale = args.number("--proposal", 0.05);
    const auto initial = stochastic_initial(args);
    const auto path = directory(args, "particle-posterior");
    args.finish();
    const auto data = hzl::Dataset::load(input);
    data.save(path / "training.csv");
    const auto chain = hzl::particle_metropolis(data, initial, scale, count, particles, seed);
    auto out = output(path / "chain.csv");
    out << "iteration,nu,alpha,sigma,tau,log_likelihood,log_posterior,likelihood_seed,accepted\n";
    std::size_t accepted = 0;
    for (std::size_t i = 0; i < chain.size(); ++i) {
        const auto &s = chain[i];
        accepted += s.accepted;
        out << i;
        for (double q : s.parameters)
            out << ',' << std::exp(q);
        out << ',' << s.log_likelihood << ',' << s.log_posterior << ',' << s.likelihood_seed << ','
            << s.accepted << '\n';
    }
    auto summary = output(path / "summary.json");
    summary << "{\"samples\":" << count << ",\"particles\":" << particles
            << ",\"acceptance_rate\":" << static_cast<double>(accepted) / static_cast<double>(count)
            << ",\"method\":\"pseudo-marginal random-walk Metropolis\"}\n";
    std::cout << "particle posterior chain written: " << path
              << "; acceptance=" << static_cast<double>(accepted) / static_cast<double>(count)
              << '\n';
}
void stochastic_calibrate(Options &args) {
    const auto input = args.get("--data", "results/observations/training.csv");
    const auto particles = args.number<std::size_t>("--particles", 32),
               iterations = args.number<std::size_t>("--iterations", 100),
               objective_replicates = args.number<std::size_t>("--objective-replicates", 3),
               checks_count = args.number<std::size_t>("--replicates", 8),
               starts = args.number<std::size_t>("--starts", 4),
               profile_points = args.number<std::size_t>("--profile-points", 7);
    const double profile_width = args.number("--profile-width", 0.7);
    const auto seed = args.number<std::uint64_t>("--seed", 123);
    const auto initial = stochastic_initial(args);
    const auto path = directory(args, "stochastic-calibration");
    args.finish();
    if (checks_count < 2 || objective_replicates == 0 || starts == 0 || profile_points < 3 ||
        profile_width <= 0)
        throw std::invalid_argument(
            "invalid stochastic calibration replication/profile configuration");
    const auto data = hzl::Dataset::load(input);
    if (!data.stochastic)
        throw std::invalid_argument("use calibrate for deterministic data");
    data.save(path / "training.csv");
    const hzl::Objective objective = [&](const auto &q) {
        return -hzl::averaged_particle_log_likelihood(data, q, particles, objective_replicates,
                                                      seed) -
               hzl::stochastic_log_prior(q);
    };
    const auto start = std::chrono::steady_clock::now();
    hzl::NormalStream start_random(hzl::stream_seed(seed, 900));
    std::vector<hzl::OptimizationResult> fits;
    fits.reserve(starts);
    for (std::size_t run = 0; run < starts; ++run) {
        auto point = initial;
        if (run == 1)
            point = {std::log(0.03), std::log(0.1), std::log(0.2), std::log(0.5)};
        else if (run > 1)
            for (std::size_t j = 0; j < point.size(); ++j) {
                point[j] += 0.45 * start_random();
                point[j] = std::clamp(point[j], -9.5, j < 2 ? -0.1 : 1.5);
            }
        fits.push_back(hzl::nelder_mead(objective, point, 0.3, iterations, 1e-3));
    }
    const auto best = std::min_element(
        fits.begin(), fits.end(), [](const auto &a, const auto &b) { return a.value < b.value; });
    const auto &result = *best;
    auto history = output(path / "history.csv");
    history << "start,iteration,evaluations,negative_log_posterior_estimate,nu,alpha,sigma,tau\n";
    for (std::size_t run = 0; run < fits.size(); ++run)
        for (const auto &r : fits[run].history) {
            history << run << ',' << r.iteration << ',' << r.evaluations << ',' << r.value;
            for (double q : r.parameters)
                history << ',' << std::exp(q);
            history << '\n';
        }
    auto starts_file = output(path / "starts.csv");
    starts_file << "start,negative_log_posterior_estimate,converged,nu,alpha,sigma,tau\n";
    for (std::size_t run = 0; run < fits.size(); ++run) {
        starts_file << run << ',' << fits[run].value << ',' << fits[run].converged;
        for (double q : fits[run].parameters)
            starts_file << ',' << std::exp(q);
        starts_file << '\n';
    }
    auto profiles = output(path / "conditional_profiles.csv");
    profiles << "parameter,offset_log,value,negative_log_posterior,delta_from_best\n";
    const std::array<std::string_view, 4> names{"nu", "alpha", "sigma", "tau"};
    for (std::size_t j = 0; j < 4; ++j)
        for (std::size_t i = 0; i < profile_points; ++i) {
            const double offset =
                profile_width *
                (2 * static_cast<double>(i) / static_cast<double>(profile_points - 1) - 1);
            auto q = result.parameters;
            q[j] += offset;
            const double value = objective(q);
            profiles << names[j] << ',' << offset << ',' << std::exp(q[j]) << ',' << value << ','
                     << value - result.value << '\n';
        }
    auto checks = output(path / "likelihood_checks.csv");
    checks << "seed,initial_log_likelihood,fitted_log_likelihood,paired_improvement\n";
    hzl::Moments likelihood, improvement;
    for (std::size_t i = 0; i < checks_count; ++i) {
        const auto check_seed = hzl::stream_seed(seed, 1000 + i);
        const double initial_value =
                         hzl::particle_log_likelihood(data, initial, particles, check_seed),
                     fitted_value = hzl::particle_log_likelihood(data, result.parameters, particles,
                                                                 check_seed);
        likelihood.add(fitted_value);
        improvement.add(fitted_value - initial_value);
        checks << check_seed << ',' << initial_value << ',' << fitted_value << ','
               << fitted_value - initial_value << '\n';
    }
    const bool independently_improved = improvement.mean > 2 * improvement.standard_error();
    auto summary = output(path / "fit.json");
    summary << "{\"nu\":" << std::exp(result.parameters[0])
            << ",\"alpha\":" << std::exp(result.parameters[1])
            << ",\"sigma\":" << std::exp(result.parameters[2])
            << ",\"tau\":" << std::exp(result.parameters[3])
            << ",\"converged\":" << (result.converged ? "true" : "false") << ",\"reason\":\""
            << result.reason << "\",\"negative_log_posterior_estimate\":" << result.value
            << ",\"independent_log_likelihood_mean\":" << likelihood.mean
            << ",\"independent_log_likelihood_sd\":" << std::sqrt(likelihood.variance())
            << ",\"paired_improvement_mean\":" << improvement.mean
            << ",\"paired_improvement_standard_error\":" << improvement.standard_error()
            << ",\"independently_improved\":" << (independently_improved ? "true" : "false")
            << ",\"particles\":" << particles
            << ",\"objective_replicates\":" << objective_replicates << ",\"starts\":" << starts
            << ",\"independent_checks\":" << checks_count << ",\"wall_seconds\":"
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
            << ",\"qualification\":\"multi-start common-random-number particle MAP; objective "
               "uses the log of an arithmetic mean of unbiased likelihood estimates; conditional "
               "slices are not nuisance-parameter profiles\"}\n";
    std::cout << "stochastic calibration " << (result.converged ? "converged" : "stopped") << ": "
              << path << "; independent log-likelihood SD=" << std::sqrt(likelihood.variance())
              << '\n';
}
void gradient_check(Options &args) {
    const auto input = args.get("--data", "results/observations/training.csv");
    const double nu = args.number("--nu", 0.05), alpha = args.number("--alpha", 0.2);
    const auto stride = args.number<std::size_t>("--checkpoint-stride", 20);
    const auto path = directory(args, "gradient-check");
    args.finish();
    if (nu <= 0 || alpha <= 0)
        throw std::invalid_argument("parameters must be positive");
    const auto data = hzl::Dataset::load(input);
    data.save(path / "training.csv");
    const hzl::Vector q{std::log(nu), std::log(alpha)};
    const auto tangent_start = std::chrono::steady_clock::now();
    const auto tangent = hzl::likelihood_gradient(data, q);
    const double tangent_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - tangent_start).count();
    const auto adjoint_start = std::chrono::steady_clock::now();
    const auto adjoint = hzl::likelihood_adjoint(data, q, stride);
    const double adjoint_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - adjoint_start).count();
    for (std::size_t j = 0; j < q.size(); ++j)
        if (std::abs(tangent.gradient[j] - adjoint.objective.gradient[j]) >
            1e-9 * (1 + std::abs(tangent.gradient[j])))
            throw std::runtime_error("tangent/adjoint gradient mismatch");
    auto summary = output(path / "timing_and_storage.json");
    summary << "{\"tangent_seconds\":" << tangent_seconds
            << ",\"adjoint_seconds\":" << adjoint_seconds << ",\"checkpoint_stride\":" << stride
            << ",\"checkpoint_fields\":" << adjoint.checkpoint_fields
            << ",\"peak_replay_block_fields\":" << adjoint.peak_block_fields
            << ",\"replayed_steps\":" << adjoint.replayed_steps
            << ",\"storage_note\":\"field counts exclude solver and RK4 workspaces\"}\n";
    const hzl::Objective objective = [&](const auto &x) {
        return hzl::negative_log_likelihood(data, x);
    };
    auto out = output(path / "gradients.csv");
    out << "h,parameter,tangent,adjoint,finite_difference,relative_error\n";
    double best = 1;
    for (double h : {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7}) {
        const auto finite = hzl::central_gradient(objective, q, h);
        double worst = 0;
        for (std::size_t j = 0; j < q.size(); ++j) {
            const double error =
                std::abs(tangent.gradient[j] - finite[j]) / (1 + std::abs(tangent.gradient[j]));
            worst = std::max(worst, error);
            out << h << ',' << j << ',' << tangent.gradient[j] << ','
                << adjoint.objective.gradient[j] << ',' << finite[j] << ',' << error << '\n';
        }
        best = std::min(best, worst);
    }
    if (best > 1e-6)
        throw std::runtime_error("PDE tangent/finite-difference comparison failed");
    std::cout << "PDE gradient check passed: " << path << "; best maximum relative error=" << best
              << '\n';
}
void reduced(Options &args) {
    const auto n = args.number<std::size_t>("--n", 32),
               training = args.number<std::size_t>("--training-steps", 200),
               validation = args.number<std::size_t>("--validation-steps", 200),
               every = args.number<std::size_t>("--every", 5),
               rank = args.number<std::size_t>("--rank", 8);
    const double dt = args.number("--dt", 0.005), fraction = args.number("--variance", 0.999999);
    const auto p = parameters(args);
    auto validation_parameters = p;
    validation_parameters.viscosity = args.number("--validation-nu", p.viscosity);
    validation_parameters.friction = args.number("--validation-alpha", p.friction);
    const auto path = directory(args, "reduced-model");
    args.finish();
    if (p.forcing_sigma != 0 || every == 0 || training < 2 * every || validation == 0 ||
        training / every + 1 > 512)
        throw std::invalid_argument(
            "ROM requires --sigma 0, 2..512 training snapshots, and positive validation duration");
    hzl::Solver full(n, p, 0);
    full.initialize("multimode");
    std::vector<hzl::Field> snapshots{full.coefficients()};
    const auto training_start = std::chrono::steady_clock::now();
    for (std::size_t step = 1; step <= training; ++step) {
        full.step(dt);
        if (step % every == 0)
            snapshots.push_back(full.coefficients());
    }
    const double training_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - training_start).count();
    const auto build_start = std::chrono::steady_clock::now();
    auto basis = hzl::fit_pod(snapshots, rank, fraction);
    hzl::Solver validation_full(n, validation_parameters, 0);
    validation_full.initialize("zero");
    validation_full.update_latent_state(full.coefficients(), hzl::Field(n * n));
    hzl::GalerkinModel model(basis, validation_full);
    const double build_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start).count();
    auto coefficients = basis.project(validation_full.coefficients());
    std::vector<hzl::Field> reference{validation_full.coefficients()};
    std::vector<std::size_t> times{0};
    const auto full_start = std::chrono::steady_clock::now();
    for (std::size_t step = 1; step <= validation; ++step) {
        validation_full.step(dt);
        if (step % every == 0 || step == validation) {
            reference.push_back(validation_full.coefficients());
            times.push_back(step);
        }
    }
    const double full_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - full_start).count();
    std::vector<hzl::Vector> reduced_states{coefficients};
    const auto reduced_start = std::chrono::steady_clock::now();
    for (std::size_t step = 1; step <= validation; ++step) {
        model.step(coefficients, dt);
        if (step % every == 0 || step == validation)
            reduced_states.push_back(coefficients);
    }
    const double reduced_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - reduced_start).count();
    auto comparisons = output(path / "validation.csv");
    comparisons << "time,projection_relative_l2,trajectory_relative_l2\n";
    double maximum = 0;
    const auto relative = [](hzl::Field estimate, const hzl::Field &truth) {
        for (std::size_t i = 0; i < truth.size(); ++i)
            estimate[i] -= truth[i];
        return std::sqrt(hzl::spectral_inner(estimate, estimate) /
                         hzl::spectral_inner(truth, truth));
    };
    const auto reconstruction_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double projection =
                         relative(basis.reconstruct(basis.project(reference[i])), reference[i]),
                     trajectory = relative(basis.reconstruct(reduced_states[i]), reference[i]);
        maximum = std::max(maximum, trajectory);
        comparisons << dt * static_cast<double>(training + times[i]) << ',' << projection << ','
                    << trajectory << '\n';
    }
    const double reconstruction_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - reconstruction_start)
            .count();
    auto singular = output(path / "singular_values.csv");
    singular << "mode,singular_value\n";
    for (std::size_t i = 0; i < basis.singular_values.size(); ++i)
        singular << i << ',' << basis.singular_values[i] << '\n';
    auto modes = output(path / "basis.csv");
    modes << "mode,fourier_index,real,imag\n";
    for (std::size_t i = 0; i < basis.mean.size(); ++i)
        modes << -1 << ',' << i << ',' << basis.mean[i].real() << ',' << basis.mean[i].imag()
              << '\n';
    for (std::size_t j = 0; j < basis.modes.size(); ++j)
        for (std::size_t i = 0; i < basis.mean.size(); ++i)
            modes << j << ',' << i << ',' << basis.modes[j][i].real() << ','
                  << basis.modes[j][i].imag() << '\n';
    auto summary = output(path / "summary.json");
    summary << "{\"rank\":" << basis.modes.size()
            << ",\"captured_training_variance\":" << basis.captured_variance
            << ",\"training_seconds\":" << training_seconds
            << ",\"pod_and_galerkin_build_seconds\":" << build_seconds
            << ",\"full_evolution_and_snapshots_seconds\":" << full_seconds
            << ",\"reduced_evolution_and_modal_snapshots_seconds\":" << reduced_seconds
            << ",\"comparison_and_reconstruction_seconds\":" << reconstruction_seconds
            << ",\"maximum_heldout_trajectory_relative_l2\":" << maximum
            << ",\"training_nu\":" << p.viscosity << ",\"training_alpha\":" << p.friction
            << ",\"validation_nu\":" << validation_parameters.viscosity
            << ",\"validation_alpha\":" << validation_parameters.friction
            << ",\"scope\":\"deterministic temporal holdout after an optional parameter jump; "
               "basis remains fixed and no closure is used\"}\n";
    std::cout << "POD/Galerkin experiment complete: " << path << "; rank=" << basis.modes.size()
              << ", maximum held-out error=" << maximum << '\n';
}
void posterior_predict(Options &args) {
    const auto input = args.get("--data", "results/observations/validation.csv"),
               chain_path = args.get("--chain", "results/posterior/chain.csv");
    const auto burn = args.number<std::size_t>("--burn", 500),
               draws = args.number<std::size_t>("--draws", 200);
    const auto seed = args.number<std::uint64_t>("--seed", 777);
    const auto path = directory(args, "posterior-predictive");
    args.finish();
    if (draws < 20)
        throw std::invalid_argument("at least 20 predictive draws are required");
    const auto data = hzl::Dataset::load(input);
    if (data.stochastic)
        throw std::invalid_argument("current posterior predictive command requires deterministic "
                                    "flow with known initial state");
    data.save(path / "observations.csv");
    std::ifstream chain(chain_path);
    std::string line;
    std::getline(chain, line);
    if (!line.starts_with("iteration,nu,alpha,log_posterior,"))
        throw std::invalid_argument("expected a deterministic parameter chain");
    std::vector<hzl::Vector> parameters;
    std::size_t iteration = 0;
    while (std::getline(chain, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        std::size_t stored_iteration = 0;
        double nu = 0, alpha = 0;
        if (!(row >> stored_iteration >> nu >> alpha) || !std::isfinite(nu) ||
            !std::isfinite(alpha) || nu <= 0 || alpha <= 0)
            throw std::runtime_error("invalid chain row");
        if (iteration++ >= burn)
            parameters.push_back({std::log(nu), std::log(alpha)});
    }
    if (parameters.size() < draws)
        throw std::invalid_argument("not enough posterior samples after burn-in");
    std::mt19937_64 selection(seed);
    std::shuffle(parameters.begin(), parameters.end(), selection);
    parameters.resize(draws);
    std::vector<hzl::Vector> states(data.observations.size()),
        observations(data.observations.size());
    hzl::NormalStream noise(hzl::stream_seed(seed, 1));
    auto selected = output(path / "parameter_draws.csv");
    selected << "draw,nu,alpha\n";
    for (std::size_t draw = 0; draw < draws; ++draw) {
        selected << draw << ',' << std::exp(parameters[draw][0]) << ','
                 << std::exp(parameters[draw][1]) << '\n';
        const auto prediction = hzl::predict(data, parameters[draw]);
        for (std::size_t i = 0; i < prediction.size(); ++i) {
            states[i].push_back(prediction[i]);
            observations[i].push_back(prediction[i] +
                                      data.observations[i].standard_deviation * noise());
        }
    }
    const auto quantile = [](hzl::Vector values, double probability) {
        std::sort(values.begin(), values.end());
        const double position = probability * static_cast<double>(values.size() - 1);
        const auto index = static_cast<std::size_t>(position);
        const auto next = std::min(index + 1, values.size() - 1);
        return values[index] +
               (position - static_cast<double>(index)) * (values[next] - values[index]);
    };
    auto out = output(path / "predictions.csv");
    out << "step,x,y,observed,state_mean,state_sd,state_q025,state_q975,observation_q025,"
           "observation_q975\n";
    std::size_t covered = 0;
    double squared_error = 0;
    for (std::size_t i = 0; i < states.size(); ++i) {
        hzl::Moments moments;
        for (auto value : states[i])
            moments.add(value);
        const auto &o = data.observations[i];
        const double lower = quantile(observations[i], 0.025),
                     upper = quantile(observations[i], 0.975);
        covered += o.value >= lower && o.value <= upper;
        squared_error += std::pow(moments.mean - o.value, 2);
        out << o.step << ',' << o.x << ',' << o.y << ',' << o.value << ',' << moments.mean << ','
            << std::sqrt(moments.variance()) << ',' << quantile(states[i], 0.025) << ','
            << quantile(states[i], 0.975) << ',' << lower << ',' << upper << '\n';
    }
    auto summary = output(path / "summary.json");
    summary << "{\"draws\":" << draws << ",\"observations\":" << states.size()
            << ",\"empirical_95pct_observation_coverage\":"
            << static_cast<double>(covered) / static_cast<double>(states.size())
            << ",\"predictive_mean_rmse\":"
            << std::sqrt(squared_error / static_cast<double>(states.size()))
            << ",\"scope\":\"parameter uncertainty and observation noise; known initial state and "
               "deterministic dynamics\"}\n";
    std::cout << "posterior predictive experiment complete: " << path << "; covered " << covered
              << '/' << states.size() << " observations\n";
}
void control_variates(Options &args) {
    const auto n = args.number<std::size_t>("--n", 16),
               steps = args.number<std::size_t>("--steps", 100),
               pilot = args.number<std::size_t>("--pilot", 32),
               count = args.number<std::size_t>("--count", 128);
    const double dt = args.number("--dt", 0.01);
    const auto seed = args.number<std::uint64_t>("--seed", 123);
    const auto p = parameters(args);
    const auto path = directory(args, "control-variates");
    args.finish();
    const auto result = hzl::control_variate_experiment(n, p, dt, steps, pilot, count, seed);
    auto out = output(path / "samples.csv");
    out << "member,full_energy,linear_energy,corrected_energy\n";
    for (std::size_t i = 0; i < result.production.size(); ++i) {
        const auto &pair = result.production[i];
        out << i << ',' << pair.full << ',' << pair.linear << ','
            << pair.full - result.beta * (pair.linear - result.known_control_mean) << '\n';
    }
    auto summary = output(path / "summary.json");
    summary << "{\"pilot_count\":" << pilot << ",\"production_count\":" << count
            << ",\"beta\":" << result.beta << ",\"known_linear_mean\":" << result.known_control_mean
            << ",\"raw_mean\":" << result.raw_mean
            << ",\"corrected_mean\":" << result.corrected_mean
            << ",\"raw_standard_error\":" << result.raw_standard_error
            << ",\"corrected_standard_error\":" << result.corrected_standard_error
            << ",\"raw_variance\":" << result.raw_variance
            << ",\"corrected_variance\":" << result.corrected_variance
            << ",\"pilot_seconds\":" << result.pilot_seconds
            << ",\"production_seconds\":" << result.production_seconds
            << ",\"scope\":\"finite-time energy; independent pilot; exact mean of the matching "
               "discrete linear Stokes control\"}\n";
    std::cout << "control variate experiment complete: " << path
              << "; raw SE=" << result.raw_standard_error
              << ", corrected SE=" << result.corrected_standard_error << '\n';
}
void antithetic_parameters(Options &args) {
    const auto n = args.number<std::size_t>("--n", 16),
               steps = args.number<std::size_t>("--steps", 100),
               pairs = args.number<std::size_t>("--pairs", 64),
               seed = args.number<std::uint64_t>("--seed", 123);
    const double dt = args.number("--dt", 0.01), log_sd = args.number("--log-sd", 0.2);
    auto p = parameters(args);
    p.forcing_sigma = 0;
    const auto path = directory(args, "antithetic-parameters");
    const auto result = hzl::parameter_antithetic_experiment(n, p, log_sd, dt, steps, pairs, seed);
    auto summary = output(path / "summary.json");
    summary << "{\"quantity\":\"final kinetic energy under Gaussian log-parameter uncertainty\","
               "\"pair_count\":"
            << pairs << ",\"full_solves_per_method\":" << 2 * pairs << ",\"log_sd\":" << log_sd
            << ",\"raw_mean\":" << result.raw_mean
            << ",\"antithetic_mean\":" << result.antithetic_mean
            << ",\"raw_standard_error\":" << result.raw_standard_error
            << ",\"antithetic_standard_error\":" << result.antithetic_standard_error
            << ",\"raw_variance\":" << result.raw_variance
            << ",\"paired_variance\":" << result.paired_variance << ",\"estimated_variance_ratio\":"
            << result.raw_standard_error * result.raw_standard_error /
                   (result.antithetic_standard_error * result.antithetic_standard_error)
            << ",\"wall_seconds\":" << result.wall_seconds
            << ",\"scope\":\"parameter uncertainty only; deterministic flow and equal full-solve "
               "counts\"}\n";
    std::cout << "antithetic parameter experiment complete: " << path << '\n';
}
void stress(Options &args) {
    const auto n = args.number<std::size_t>("--n", 16),
               steps = args.number<std::size_t>("--steps", 100),
               count = args.number<std::size_t>("--count", 64),
               workers = args.number<std::size_t>("--workers", 6);
    const double dt = args.number("--dt", 0.01),
                 threshold = args.number("--dissipation-threshold", 0.08);
    const auto seed = args.number<std::uint64_t>("--seed", 123);
    const auto baseline = parameters(args);
    const auto path = directory(args, "stress");
    args.finish();
    if (count < 2 || steps == 0 || threshold < 0)
        throw std::invalid_argument("stress study requires at least two samples, nonzero duration, "
                                    "and nonnegative threshold");
    std::vector<std::pair<std::string, hzl::Parameters>> scenarios{{"baseline", baseline}};
    auto changed = baseline;
    changed.forcing_sigma *= 2;
    scenarios.emplace_back("double_forcing", changed);
    changed = baseline;
    changed.viscosity *= 0.5;
    scenarios.emplace_back("half_viscosity", changed);
    changed = baseline;
    changed.friction *= 0.5;
    scenarios.emplace_back("half_friction", changed);
    changed = baseline;
    changed.forcing_tau *= 2;
    scenarios.emplace_back("double_correlation_time", changed);
    auto summary = output(path / "scenarios.csv"), members = output(path / "members.csv");
    summary << "scenario,nu,alpha,sigma,tau,energy_mean,energy_standard_error,dissipation_mean,"
               "exceedances,count,tail_probability,wilson95_lower,wilson95_upper,wall_seconds\n";
    members << "scenario,member,seed,energy,enstrophy,dissipation,exceeds_threshold\n";
    for (const auto &[name, p] : scenarios) {
        const auto start = std::chrono::steady_clock::now();
        const auto values = hzl::ensemble(n, p, dt, steps, count, workers, seed);
        hzl::Moments energy, dissipation;
        std::size_t events = 0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            const auto &d = values[i];
            energy.add(d.energy);
            dissipation.add(d.dissipation);
            const bool event = d.dissipation > threshold;
            events += event;
            members << name << ',' << i << ',' << hzl::stream_seed(seed, i) << ',' << d.energy
                    << ',' << d.enstrophy << ',' << d.dissipation << ',' << event << '\n';
        }
        const double total = static_cast<double>(count),
                     probability = static_cast<double>(events) / total, z = 1.959963984540054;
        const double denominator = 1 + z * z / total,
                     center = (probability + z * z / (2 * total)) / denominator,
                     radius = z / denominator *
                              std::sqrt(probability * (1 - probability) / total +
                                        z * z / (4 * total * total));
        summary << name << ',' << p.viscosity << ',' << p.friction << ',' << p.forcing_sigma << ','
                << p.forcing_tau << ',' << energy.mean << ',' << energy.standard_error() << ','
                << dissipation.mean << ',' << events << ',' << count << ',' << probability << ','
                << std::max(0.0, center - radius) << ',' << std::min(1.0, center + radius) << ','
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                << '\n';
    }
    auto scope = output(path / "scope.json");
    scope << "{\"event\":\"final-time viscous energy dissipation exceeds threshold\",\"threshold\":"
          << threshold
          << ",\"method\":\"direct Monte Carlo with Wilson 95% binomial "
             "intervals\",\"coupling\":\"common logical random streams across "
             "scenarios\",\"limitation\":\"finite-time study, not stationary or path-maximum "
             "rare-event inference; sigma is held fixed when tau changes, so stationary forcing "
             "variance changes\"}\n";
    std::cout << "parameter stress study complete: " << path << '\n';
}
void surrogate_experiment(Options &args) {
    const auto input = args.get("--data", "results/observations/training.csv");
    const auto count = args.number<std::size_t>("--samples", 1000);
    const auto seed = args.number<std::uint64_t>("--seed", 123);
    const double width = args.number("--width", 0.15), proposal = args.number("--proposal", 0.08);
    const auto surrogate_type = args.get("--surrogate", "rbf");
    const double rbf_shape = args.number("--rbf-shape", 0.8),
                 rbf_ridge = args.number("--rbf-ridge", 1e-10);
    const auto path = directory(args, "surrogate");
    args.finish();
    if (width <= 0 || count == 0 || (surrogate_type != "quadratic" && surrogate_type != "rbf"))
        throw std::invalid_argument("surrogate width and samples must be positive");
    const auto data = hzl::Dataset::load(input);
    data.save(path / "training_observations.csv");
    const hzl::Objective nll = [&](const auto &q) { return hzl::negative_log_likelihood(data, q); };
    const auto start = std::chrono::steady_clock::now();
    const auto fit = hzl::nelder_mead(nll, {std::log(0.03), std::log(0.1)}, 0.2, 160);
    if (!fit.converged)
        throw std::runtime_error("surrogate center calibration failed to converge");
    std::vector<hzl::Vector> inputs;
    hzl::Vector targets;
    auto training = output(path / "training_design.csv");
    training << "log_nu,log_alpha,nll\n";
    for (int i = -2; i <= 2; ++i)
        for (int j = -2; j <= 2; ++j) {
            hzl::Vector q{fit.parameters[0] + width * i / 2, fit.parameters[1] + width * j / 2};
            const double value = nll(q);
            inputs.push_back(q);
            targets.push_back(value);
            training << q[0] << ',' << q[1] << ',' << value << '\n';
        }
    hzl::QuadraticSurrogate model(inputs, targets, fit.parameters, {width, width});
    hzl::GaussianRBFSurrogate rbf(inputs, targets, fit.parameters, {width, width}, rbf_shape,
                                  rbf_ridge);
    const double training_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    auto validation = output(path / "validation.csv");
    validation
        << "region,log_nu,log_alpha,exact_nll,quadratic_nll,quadratic_error,rbf_nll,rbf_error\n";
    std::mt19937_64 random(seed);
    double interpolation_squared = 0, extrapolation_squared = 0, rbf_interpolation_squared = 0,
           rbf_extrapolation_squared = 0;
    const auto uniform = [&]() { return static_cast<double>(random() >> 11) * 0x1.0p-53; };
    for (int i = 0; i < 24; ++i) {
        const bool outside = i >= 12;
        const double radius = outside ? 2.0 : 1.0;
        hzl::Vector q{fit.parameters[0] + radius * width * (2 * uniform() - 1),
                      fit.parameters[1] + radius * width * (2 * uniform() - 1)};
        if (outside)
            q[static_cast<std::size_t>(i % 2)] =
                fit.parameters[static_cast<std::size_t>(i % 2)] + radius * width;
        const double exact = nll(q), approximate = model(q), error = approximate - exact,
                     rbf_approximate = rbf(q), rbf_error = rbf_approximate - exact;
        if (outside) {
            extrapolation_squared += error * error;
            rbf_extrapolation_squared += rbf_error * rbf_error;
        } else {
            interpolation_squared += error * error;
            rbf_interpolation_squared += rbf_error * rbf_error;
        }
        validation << (outside ? "extrapolation" : "interpolation") << ',' << q[0] << ',' << q[1]
                   << ',' << exact << ',' << approximate << ',' << error << ',' << rbf_approximate
                   << ',' << rbf_error << '\n';
    }
    const hzl::Objective prior = [](const auto &q) {
        if (q.size() != 2)
            throw std::invalid_argument("two log parameters required");
        for (double x : q)
            if (!std::isfinite(x) || x < -12 || x > 0)
                return -std::numeric_limits<double>::infinity();
        const double a = (q[0] - std::log(0.03)) / 1.5, b = (q[1] - std::log(0.1)) / 1.5;
        return -0.5 * (a * a + b * b);
    };
    const hzl::Objective exact = [&](const auto &q) { return -nll(q) + prior(q); },
                         approximate = [&](const auto &q) {
                             const double value = prior(q);
                             const double surrogate_nll =
                                 surrogate_type == "rbf" ? rbf(q) : model(q);
                             return std::isfinite(value) ? -surrogate_nll + value : value;
                         };
    const auto sample_start = std::chrono::steady_clock::now();
    const auto chain = hzl::delayed_acceptance(exact, approximate, fit.parameters,
                                               {proposal, proposal}, count, seed);
    const double sample_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - sample_start).count();
    auto out = output(path / "chain.csv");
    out << "iteration,nu,alpha,log_posterior,accepted\n";
    std::size_t accepted = 0;
    for (std::size_t i = 0; i < chain.samples.size(); ++i) {
        const auto &s = chain.samples[i];
        accepted += s.accepted;
        out << i << ',' << std::exp(s.parameters[0]) << ',' << std::exp(s.parameters[1]) << ','
            << s.log_density << ',' << s.accepted << '\n';
    }
    auto model_file = output(path / "model.json");
    model_file << "{\"schema\":\"HZL_QUADRATIC_V1\",\"center\":[" << fit.parameters[0] << ','
               << fit.parameters[1] << "],\"scales\":[" << width << ',' << width
               << "],\"features\":\"1,x0,x1,x0^2,x0*x1,x1^2\",\"coefficients\":[";
    for (std::size_t i = 0; i < model.coefficients().size(); ++i) {
        if (i)
            model_file << ',';
        model_file << model.coefficients()[i];
    }
    model_file << "]}\n";
    auto summary = output(path / "summary.json");
    summary
        << "{\"training_rmse\":" << model.training_rmse()
        << ",\"interpolation_rmse\":" << std::sqrt(interpolation_squared / 12)
        << ",\"extrapolation_rmse\":" << std::sqrt(extrapolation_squared / 12)
        << ",\"rbf_training_rmse\":" << rbf.training_rmse()
        << ",\"rbf_interpolation_rmse\":" << std::sqrt(rbf_interpolation_squared / 12)
        << ",\"rbf_extrapolation_rmse\":" << std::sqrt(rbf_extrapolation_squared / 12)
        << ",\"sampling_surrogate\":\"" << surrogate_type << "\""
        << ",\"centering_evaluations\":" << fit.evaluations
        << ",\"design_evaluations\":25,\"validation_evaluations\":24,\"centering_and_training_"
           "seconds\":"
        << training_seconds << ",\"samples\":" << count
        << ",\"stage_one_acceptances\":" << chain.stage_one_acceptances
        << ",\"sampling_exact_evaluations\":" << chain.exact_evaluations
        << ",\"acceptance_rate\":" << static_cast<double>(accepted) / static_cast<double>(count)
        << ",\"sampling_seconds\":" << sample_seconds
        << ",\"correction\":\"two-stage delayed acceptance with full likelihood correction\"}\n";
    std::cout << "surrogate experiment complete: " << path
              << "; exact sampling evaluations=" << chain.exact_evaluations << " for " << count
              << " proposals\n";
}
double relative_field_error(const hzl::Field &value, const hzl::Field &reference) {
    if (value.size() != reference.size())
        throw std::invalid_argument("field error size mismatch");
    double numerator = 0, denominator = 0;
    for (std::size_t i = 0; i < value.size(); ++i) {
        numerator += std::norm(value[i] - reference[i]);
        denominator += std::norm(reference[i]);
    }
    return std::sqrt(numerator / denominator);
}
hzl::Field smooth_vorticity(std::size_t n) {
    hzl::Field field(n * n);
    for (std::size_t y = 0; y < n; ++y)
        for (std::size_t x = 0; x < n; ++x) {
            const double px = 2 * std::numbers::pi * static_cast<double>(x) / n,
                         py = 2 * std::numbers::pi * static_cast<double>(y) / n;
            field[y * n + x] =
                std::exp(0.7 * std::cos(px) + 0.4 * std::sin(2 * py) + 0.25 * std::cos(px - py));
        }
    return field;
}
hzl::Field restrict_modes(const hzl::Field &fine, std::size_t fine_n, std::size_t coarse_n) {
    hzl::Field result(coarse_n * coarse_n);
    for (std::size_t y = 0; y < coarse_n; ++y)
        for (std::size_t x = 0; x < coarse_n; ++x) {
            const int kx = hzl::wave_number(x, coarse_n), ky = hzl::wave_number(y, coarse_n);
            const auto fx = static_cast<std::size_t>(kx >= 0 ? kx : static_cast<int>(fine_n) + kx),
                       fy = static_cast<std::size_t>(ky >= 0 ? ky : static_cast<int>(fine_n) + ky);
            result[y * coarse_n + x] = fine[fy * fine_n + fx];
        }
    return result;
}
void convergence_experiment(Options &args) {
    const double duration = args.number("--duration", 0.4), base_dt = args.number("--dt", 0.04);
    const auto levels = args.number<std::size_t>("--levels", 4),
               reference_n = args.number<std::size_t>("--reference-n", 128);
    auto p = parameters(args);
    p.forcing_sigma = 0;
    const auto path = directory(args, "convergence");
    if (duration <= 0 || base_dt <= 0 || levels < 3 || reference_n < 32 ||
        (reference_n & (reference_n - 1)) != 0)
        throw std::invalid_argument("convergence requires positive times, at least three levels, "
                                    "and a power-of-two reference grid >= 32");

    auto temporal = output(path / "temporal.csv");
    temporal << "integrator,dt,steps,relative_error,observed_order,wall_seconds\n";
    for (auto integrator : {hzl::TimeIntegrator::ssprk3, hzl::TimeIntegrator::rk4}) {
        double previous = std::numeric_limits<double>::quiet_NaN();
        for (std::size_t level = 0; level < levels; ++level) {
            const double requested = std::ldexp(base_dt, -static_cast<int>(level));
            const auto steps = static_cast<std::size_t>(std::ceil(duration / requested));
            const double dt = duration / static_cast<double>(steps);
            hzl::Solver solver(16, p, 1);
            solver.initialize("taylor-green");
            auto exact = solver.coefficients();
            const double decay = std::exp(-(2 * p.viscosity + p.friction) * duration);
            for (auto &value : exact)
                value *= decay;
            const auto start = std::chrono::steady_clock::now();
            for (std::size_t step = 0; step < steps; ++step)
                solver.step(dt, integrator);
            const double seconds =
                             std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                                 .count(),
                         error = relative_field_error(solver.coefficients(), exact),
                         order = std::isfinite(previous) ? std::log2(previous / error)
                                                         : std::numeric_limits<double>::quiet_NaN();
            temporal << hzl::integrator_name(integrator) << ',' << dt << ',' << steps << ','
                     << error << ',';
            if (std::isfinite(order))
                temporal << order;
            temporal << ',' << seconds << '\n';
            previous = error;
        }
    }

    hzl::Solver reference(reference_n, p, 1);
    reference.initialize_physical(smooth_vorticity(reference_n));
    const auto reference_drift = reference.drift(reference.coefficients());
    auto spatial = output(path / "spatial.csv");
    spatial << "n,relative_drift_error,wall_seconds\n";
    for (std::size_t n = 16; n < reference_n; n *= 2) {
        const auto start = std::chrono::steady_clock::now();
        hzl::Solver solver(n, p, 1);
        solver.initialize_physical(smooth_vorticity(n));
        const auto drift = solver.drift(solver.coefficients());
        spatial << n << ','
                << relative_field_error(drift, restrict_modes(reference_drift, reference_n, n))
                << ','
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                << '\n';
    }
    auto scope = output(path / "scope.json");
    scope << "{\"temporal_solution\":\"Taylor-Green linear "
             "decay\",\"spatial_quantity\":\"instantaneous deterministic drift of a smooth "
             "non-bandlimited field\",\"reference_n\":"
          << reference_n
          << ",\"forcing_sigma\":0,\"limitation\":\"self-convergence against a finite spectral "
             "reference is not an independent manufactured-solution proof\"}\n";
    std::cout << "convergence experiment complete: " << path << '\n';
}
std::pair<double, double> wilson_interval(std::size_t successes, std::size_t count) {
    const double n = static_cast<double>(count), p = static_cast<double>(successes) / n,
                 z = 1.959963984540054, denominator = 1 + z * z / n,
                 center = (p + z * z / (2 * n)) / denominator,
                 radius = z / denominator * std::sqrt(p * (1 - p) / n + z * z / (4 * n * n));
    return {std::max(0.0, center - radius), std::min(1.0, center + radius)};
}
void rare_event_experiment(Options &args) {
    const auto n = args.number<std::size_t>("--n", 16),
               segment_steps = args.number<std::size_t>("--segment-steps", 25),
               levels = args.number<std::size_t>("--levels", 6),
               particles = args.number<std::size_t>("--particles", 64),
               splitting_replicates = args.number<std::size_t>("--splitting-replicates", 8),
               direct_count = args.number<std::size_t>("--direct-count", 128),
               seed = args.number<std::uint64_t>("--seed", 4242);
    const double dt = args.number("--dt", 0.01),
                 threshold = args.number("--dissipation-threshold", 0.12);
    const auto p = parameters(args);
    const auto path = directory(args, "rare-event");
    if (splitting_replicates < 2 || direct_count == 0)
        throw std::invalid_argument(
            "rare-event study needs at least two splitting replicates and direct samples");
    const auto start = std::chrono::steady_clock::now();
    auto level_file = output(path / "levels.csv");
    level_file << "replicate,level,trajectories,survivors,threshold_hits,time,score_cutoff,"
                  "probability_factor\n";
    hzl::Moments splitting_estimates;
    std::size_t completed_splitting = 0;
    for (std::size_t replicate = 0; replicate < splitting_replicates; ++replicate) {
        const auto splitting =
            hzl::dissipation_splitting(n, p, dt, segment_steps, levels, particles, threshold,
                                       hzl::stream_seed(seed, replicate));
        splitting_estimates.add(splitting.probability_estimate);
        completed_splitting += splitting.threshold_reached;
        for (const auto &level : splitting.levels)
            level_file << replicate << ',' << level.level << ',' << level.trajectories << ','
                       << level.survivors << ',' << level.threshold_hits << ',' << level.time << ','
                       << level.score_cutoff << ',' << level.probability_factor << '\n';
    }
    std::size_t direct_hits = 0;
    auto direct_file = output(path / "direct.csv");
    direct_file << "member,seed,maximum_dissipation,exceeded\n";
    const auto total_steps = segment_steps * levels;
    for (std::size_t member = 0; member < direct_count; ++member) {
        const auto member_seed = hzl::stream_seed(seed, 100000 + member);
        hzl::Solver solver(n, p, member_seed);
        solver.initialize("zero");
        double maximum = 0;
        for (std::size_t step = 0; step < total_steps; ++step) {
            solver.step(dt);
            maximum = std::max(maximum, solver.diagnostics().dissipation);
        }
        const bool exceeded = maximum >= threshold;
        direct_hits += exceeded;
        direct_file << member << ',' << member_seed << ',' << maximum << ',' << exceeded << '\n';
    }
    const auto direct_interval = wilson_interval(direct_hits, direct_count);
    auto summary = output(path / "summary.json");
    summary << "{\"event\":\"path-maximum viscous dissipation exceeds threshold\",\"threshold\":"
            << threshold << ",\"splitting_probability_mean\":" << splitting_estimates.mean
            << ",\"splitting_probability_standard_error\":" << splitting_estimates.standard_error()
            << ",\"splitting_replicates\":" << splitting_replicates
            << ",\"splitting_threshold_completions\":" << completed_splitting
            << ",\"direct_hits\":" << direct_hits << ",\"direct_count\":" << direct_count
            << ",\"direct_probability\":" << static_cast<double>(direct_hits) / direct_count
            << ",\"direct_wilson95\":[" << direct_interval.first << ',' << direct_interval.second
            << "],\"wall_seconds\":"
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
            << ",\"qualification\":\"fixed-level splitting with random cloning at first "
               "threshold-hitting states; finite-population bias is not corrected\"}\n";
    std::cout << "rare-event experiment complete: " << path
              << "; splitting estimate=" << splitting_estimates.mean << " +/- "
              << splitting_estimates.standard_error() << '\n';
}
void coverage_experiment(Options &args) {
    const auto count = args.number<std::size_t>("--replicates", 24),
               n = args.number<std::size_t>("--n", 16),
               steps = args.number<std::size_t>("--steps", 90),
               every = args.number<std::size_t>("--every", 10),
               seed = args.number<std::uint64_t>("--seed", 20260917);
    const double dt = args.number("--dt", 0.01), noise = args.number("--noise", 0.08),
                 nu = args.number("--nu", 0.035), alpha = args.number("--alpha", 0.12),
                 hessian_step = args.number("--hessian-step", 0.002);
    const auto path = directory(args, "coverage");
    if (count < 4 || steps == 0 || every == 0 || dt <= 0 || noise <= 0 || nu <= 0 || alpha <= 0 ||
        hessian_step <= 0)
        throw std::invalid_argument("invalid coverage-study configuration");
    hzl::Parameters truth;
    truth.viscosity = nu;
    truth.friction = alpha;
    truth.forcing_sigma = 0;
    const hzl::Vector true_q{std::log(nu), std::log(alpha)};
    auto trials = output(path / "trials.csv");
    trials << "replicate,seed,converged,valid_hessian,nu_hat,alpha_hat,se_log_nu,se_log_alpha,"
              "nu_covered,alpha_covered,joint_covered,nll,evaluations\n";
    std::size_t converged = 0, valid = 0, nu_covered = 0, alpha_covered = 0, joint_covered = 0;
    hzl::Moments nu_estimates, alpha_estimates;
    const auto begin = std::chrono::steady_clock::now();
    for (std::size_t replicate = 0; replicate < count; ++replicate) {
        const auto replicate_seed = hzl::stream_seed(seed, replicate);
        const auto data =
            hzl::synthetic_observations(n, dt, steps, every, truth, noise, replicate_seed);
        const auto fit = hzl::bfgs([&](const auto &q) { return hzl::likelihood_gradient(data, q); },
                                   {std::log(0.03), std::log(0.1)}, 100, 1e-6);
        converged += fit.converged;
        const auto &q = fit.parameters;
        double hessian[2][2]{};
        for (std::size_t column = 0; column < 2; ++column) {
            auto plus = q, minus = q;
            plus[column] += hessian_step;
            minus[column] -= hessian_step;
            const auto gp = hzl::likelihood_gradient(data, plus).gradient,
                       gm = hzl::likelihood_gradient(data, minus).gradient;
            for (std::size_t row = 0; row < 2; ++row)
                hessian[row][column] = (gp[row] - gm[row]) / (2 * hessian_step);
        }
        const double cross = 0.5 * (hessian[0][1] + hessian[1][0]);
        hessian[0][1] = hessian[1][0] = cross;
        const double determinant = hessian[0][0] * hessian[1][1] - cross * cross;
        const bool valid_hessian = fit.converged && determinant > 0 && hessian[0][0] > 0 &&
                                   hessian[1][1] > 0 && std::isfinite(determinant);
        double se0 = std::numeric_limits<double>::quiet_NaN(),
               se1 = std::numeric_limits<double>::quiet_NaN();
        bool cover0 = false, cover1 = false, cover_joint = false;
        if (valid_hessian) {
            ++valid;
            se0 = std::sqrt(hessian[1][1] / determinant);
            se1 = std::sqrt(hessian[0][0] / determinant);
            cover0 = std::abs(q[0] - true_q[0]) <= 1.959963984540054 * se0;
            cover1 = std::abs(q[1] - true_q[1]) <= 1.959963984540054 * se1;
            const double d0 = q[0] - true_q[0], d1 = q[1] - true_q[1];
            cover_joint =
                d0 * (hessian[0][0] * d0 + cross * d1) + d1 * (cross * d0 + hessian[1][1] * d1) <=
                5.991464547107979;
            nu_covered += cover0;
            alpha_covered += cover1;
            joint_covered += cover_joint;
        }
        nu_estimates.add(std::exp(q[0]));
        alpha_estimates.add(std::exp(q[1]));
        trials << replicate << ',' << replicate_seed << ',' << fit.converged << ',' << valid_hessian
               << ',' << std::exp(q[0]) << ',' << std::exp(q[1]) << ',';
        if (valid_hessian)
            trials << se0 << ',' << se1;
        else
            trials << ',';
        trials << ',' << cover0 << ',' << cover1 << ',' << cover_joint << ',' << fit.value << ','
               << fit.evaluations << '\n';
    }
    if (valid == 0)
        throw std::runtime_error("coverage study produced no converged positive-definite fits");
    const auto nu_interval = wilson_interval(nu_covered, valid),
               alpha_interval = wilson_interval(alpha_covered, valid),
               joint_interval = wilson_interval(joint_covered, valid);
    auto summary = output(path / "summary.json");
    summary << "{\"replicates\":" << count << ",\"optimizer_converged\":" << converged
            << ",\"valid_hessians\":" << valid
            << ",\"nominal_marginal_coverage\":0.95,"
               "\"nu_coverage\":"
            << static_cast<double>(nu_covered) / valid << ",\"nu_wilson95\":[" << nu_interval.first
            << ',' << nu_interval.second
            << "],\"alpha_coverage\":" << static_cast<double>(alpha_covered) / valid
            << ",\"alpha_wilson95\":[" << alpha_interval.first << ',' << alpha_interval.second
            << "],\"nominal_joint_coverage\":0.95,\"joint_coverage\":"
            << static_cast<double>(joint_covered) / valid << ",\"joint_wilson95\":["
            << joint_interval.first << ',' << joint_interval.second
            << "],\"nu_mean\":" << nu_estimates.mean << ",\"alpha_mean\":" << alpha_estimates.mean
            << ",\"truth\":{" << "\"nu\":" << nu << ",\"alpha\":" << alpha << "},\"wall_seconds\":"
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()
            << ",\"method\":\"observed-Hessian Laplace/Wald intervals in log parameters\","
               "\"limitation\":\"finite repeated-data study; intervals are local Gaussian "
               "approximations, not MCMC credible intervals\"}\n";
    std::cout << "coverage experiment complete: " << path << "; valid fits=" << valid << '/'
              << count << '\n';
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc < 2 || std::string_view(argv[1]) == "--help") {
            std::cout
                << "HZL stochastic turbulence engine 0.1.0\n"
                << "Commands: simulate, observe, calibrate, sample, ensemble, filter, "
                   "particle-sample, calibrate-stochastic, gradient-check, reduced, "
                   "posterior-predict, control-variates, antithetic-parameters, stress, surrogate, "
                   "convergence, coverage, rare-event\n"
                << "Simulation: --n 32 --steps 1000 --dt 0.005 --every 10 --seed 42\n"
                << "Parameters: --nu 0.01 --alpha 0.1 --sigma 0.2 --tau 0.5 --kmin 2 --kmax 3\n"
                << "simulate: --initial multimode|taylor-green|zero --integrator rk4|ssprk3 "
                   "--restart checkpoint.txt\n"
                << "observe: --noise 0.02 (use --sigma 0 for a deterministic inverse experiment)\n"
                << "calibrate: --data training.csv --validation validation.csv --initial-nu 0.08 "
                   "--initial-alpha 0.3 --iterations 160 --method nelder-mead|bfgs\n"
                << "  BFGS: --gradient tangent|adjoint --checkpoint-stride 20\n"
                << "gradient-check: --data training.csv --nu 0.05 --alpha 0.2 --checkpoint-stride "
                   "20\n"
                << "surrogate: --data training.csv --samples 1000 --width 0.15 --proposal 0.08 "
                   "--seed 123\n"
                << "convergence: --duration 0.4 --dt 0.04 --levels 4 --reference-n 128 and "
                   "physical parameters\n"
                << "coverage: --replicates 24 --steps 90 --every 10 --noise 0.08 --nu 0.035 "
                   "--alpha 0.12 --hessian-step 0.002\n"
                << "rare-event: --particles 64 --segment-steps 25 --levels 6 "
                   "--splitting-replicates 8 --direct-count 128 --dissipation-threshold 0.12\n"
                << "reduced: --sigma 0 --n 32 --training-steps 200 --validation-steps 200 --dt "
                   "0.005 --every 5 --rank 8 --variance 0.999999\n"
                << "posterior-predict: --chain chain.csv --data validation.csv --burn 500 --draws "
                   "200 --seed 777\n"
                << "control-variates: --n 16 --steps 100 --dt 0.01 --pilot 32 --count 128 --seed "
                   "123 and physical parameters\n"
                << "antithetic-parameters: --n 16 --steps 100 --pairs 64 --log-sd 0.2 --sigma 0\n"
                << "stress: --n 16 --steps 100 --dt 0.01 --count 64 --workers 6 "
                   "--dissipation-threshold 0.08 and physical parameters\n"
                << "sample: --data training.csv --samples 1000 --proposal 0.03 --initial-nu 0.03 "
                   "--initial-alpha 0.1 --seed 123 --method random-walk|adaptive|hmc --warmup 500 "
                   "--step-size 0.005 --leapfrog 10\n"
                << "ensemble: --count 16 --workers 4 (zero initial vorticity)\n"
                << "filter: --data observations.csv --particles 64 --method particle|enkf --seed "
                   "2026 and physical parameters\n"
                << "particle-sample: --data training.csv --particles 32 --samples 500 --proposal "
                   "0.05 --seed 123\n"
                << "calibrate-stochastic: --data training.csv --particles 32 --iterations 60 "
                   "--objective-replicates 3 --replicates 8 --starts 4 --profile-points 7 "
                   "--profile-width 0.7 --seed 123\n"
                << "Stochastic inference initial guesses: --initial-nu 0.03 --initial-alpha 0.1 "
                   "--initial-sigma 0.2 --initial-tau 0.5\n"
                << "All commands: --output NEW_DIRECTORY\n";
            return 0;
        }
        Options args(argc, argv);
        const std::string command = argv[1];
        if (command == "simulate")
            simulate(args);
        else if (command == "observe")
            observe(args);
        else if (command == "calibrate")
            calibrate(args);
        else if (command == "sample")
            sample(args);
        else if (command == "ensemble")
            ensembles(args);
        else if (command == "filter")
            filter(args);
        else if (command == "particle-sample")
            particle_sample(args);
        else if (command == "calibrate-stochastic")
            stochastic_calibrate(args);
        else if (command == "gradient-check")
            gradient_check(args);
        else if (command == "reduced")
            reduced(args);
        else if (command == "posterior-predict")
            posterior_predict(args);
        else if (command == "control-variates")
            control_variates(args);
        else if (command == "antithetic-parameters")
            antithetic_parameters(args);
        else if (command == "surrogate")
            surrogate_experiment(args);
        else if (command == "stress")
            stress(args);
        else if (command == "convergence")
            convergence_experiment(args);
        else if (command == "coverage")
            coverage_experiment(args);
        else if (command == "rare-event")
            rare_event_experiment(args);
        else
            throw std::invalid_argument("unknown command: " + command);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
