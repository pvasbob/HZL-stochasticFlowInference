#pragma once
#include "hzl/fft.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <random>
#include <string>
#include <string_view>

namespace hzl {
enum class TimeIntegrator { rk4, ssprk3 };
std::string_view integrator_name(TimeIntegrator integrator);
TimeIntegrator parse_integrator(std::string_view name);
struct Parameters {
    double viscosity = 0.01;
    double friction = 0.1;
    double forcing_sigma = 0.2;
    double forcing_tau = 0.5;
    int forcing_min = 2;
    int forcing_max = 3;
    void validate(std::size_t n) const;
};
struct Diagnostics {
    double time{}, energy{}, enstrophy{}, dissipation{}, injection{}, cfl_rate{};
};
class NormalStream {
  public:
    explicit NormalStream(std::uint64_t seed) : engine_(seed) {}
    double operator()();
    void write(std::ostream &out) const;
    void read(std::istream &in);

  private:
    std::mt19937_64 engine_;
    bool has_spare_ = false;
    double spare_ = 0;
};
// Stable logical stream identities, independent of worker scheduling.
std::uint64_t stream_seed(std::uint64_t base, std::uint64_t index);
using ModePairs = std::vector<std::pair<std::size_t, std::size_t>>;
ModePairs forcing_pairs(std::size_t n, const Parameters &parameters);
void initialize_ou(Field &field, const ModePairs &pairs, const Parameters &parameters,
                   NormalStream &random);
void transition_ou(Field &field, const ModePairs &pairs, const Parameters &parameters,
                   NormalStream &random, double dt);
class ShellForcing {
  public:
    ShellForcing(std::size_t n, Parameters parameters, std::uint64_t seed);
    const Field &midpoint_step(double dt);
    [[nodiscard]] const Field &coefficients() const noexcept {
        return field_;
    }

  private:
    Parameters parameters_;
    NormalStream random_;
    ModePairs pairs_;
    Field field_, midpoint_;
};
class Solver {
  public:
    Solver(std::size_t n, Parameters parameters, std::uint64_t seed);
    void initialize(const std::string &name);
    void initialize_physical(const Field &vorticity);
    void step(double dt, TimeIntegrator integrator = TimeIntegrator::rk4);
    // Deterministic log-parameter tangent equations, differentiated through RK4.
    void step_tangent(double dt, std::array<Field, 2> &sensitivities);
    [[nodiscard]] Field drift(const Field &state);
    [[nodiscard]] Field linearized_drift(const Field &state, const Field &direction);
    [[nodiscard]] Field adjoint_drift(const Field &state, const Field &cotangent);
    // Change future innovations only; used after particle-filter resampling.
    void reseed_future(std::uint64_t seed) {
        random_ = NormalStream(seed);
    }
    void update_latent_state(const Field &vorticity, const Field &forcing);
    [[nodiscard]] Diagnostics diagnostics();
    [[nodiscard]] Field physical_vorticity();
    [[nodiscard]] std::vector<double> energy_spectrum() const;
    [[nodiscard]] const Field &coefficients() const noexcept {
        return omega_;
    }
    [[nodiscard]] const Field &forcing() const noexcept {
        return forcing_;
    }
    [[nodiscard]] const Field &midpoint_forcing() const noexcept {
        return midpoint_forcing_;
    }
    [[nodiscard]] double time() const noexcept {
        return time_;
    }
    [[nodiscard]] std::size_t size() const noexcept {
        return n_;
    }
    [[nodiscard]] const Parameters &parameters() const noexcept {
        return parameters_;
    }
    void save(const std::filesystem::path &path) const;
    static Solver load(const std::filesystem::path &path);

  private:
    void project(Field &field) const;
    void advance_forcing(double dt);
    void velocity(const Field &omega);
    void rhs(const Field &state, Field &result);
    void validate_step(double dt);
    std::size_t n_;
    Parameters parameters_;
    Fourier2D fft_;
    NormalStream random_;
    double time_ = 0;
    Field omega_, forcing_, midpoint_forcing_, u_, v_, wx_, wy_, nonlinear_, temporary_, k1_, k2_,
        k3_, k4_;
    std::vector<double> kx_, ky_, ksq_;
    std::vector<bool> active_;
    std::vector<std::pair<std::size_t, std::size_t>> forced_pairs_;
};
} // namespace hzl
