#include "hzl/solver.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace hzl {
namespace {
constexpr double pi = std::numbers::pi;
constexpr Complex imaginary{0, 1};
} // namespace
std::string_view integrator_name(TimeIntegrator integrator) {
    switch (integrator) {
    case TimeIntegrator::rk4:
        return "rk4";
    case TimeIntegrator::ssprk3:
        return "ssprk3";
    }
    throw std::invalid_argument("unknown time integrator");
}
TimeIntegrator parse_integrator(std::string_view name) {
    if (name == "rk4")
        return TimeIntegrator::rk4;
    if (name == "ssprk3")
        return TimeIntegrator::ssprk3;
    throw std::invalid_argument("integrator must be rk4 or ssprk3");
}
void Parameters::validate(std::size_t n) const {
    if (!std::isfinite(viscosity) || viscosity < 0 || !std::isfinite(friction) || friction < 0 ||
        !std::isfinite(forcing_sigma) || forcing_sigma < 0 || !std::isfinite(forcing_tau) ||
        forcing_tau <= 0)
        throw std::invalid_argument(
            "nu, alpha, sigma must be finite and nonnegative; tau must be positive");
    if (forcing_min < 1 || forcing_max < forcing_min ||
        3 * static_cast<long long>(forcing_max) >= static_cast<long long>(n))
        throw std::invalid_argument("forcing shell must lie strictly inside the dealiased grid");
}
double NormalStream::operator()() {
    if (has_spare_) {
        has_spare_ = false;
        return spare_;
    }
    const auto uniform = [this]() {
        return (static_cast<double>(engine_() >> 11) + 0.5) * 0x1.0p-53;
    };
    const double r = std::sqrt(-2 * std::log(uniform())), angle = 2 * pi * uniform();
    spare_ = r * std::sin(angle);
    has_spare_ = true;
    return r * std::cos(angle);
}
void NormalStream::write(std::ostream &out) const {
    out << engine_ << '\n' << has_spare_ << ' ' << spare_ << '\n';
}
void NormalStream::read(std::istream &in) {
    if (!(in >> engine_ >> has_spare_ >> spare_) || !std::isfinite(spare_))
        throw std::runtime_error("invalid checkpoint random stream");
}
std::uint64_t stream_seed(std::uint64_t base, std::uint64_t index) {
    auto x = base + 0x9e3779b97f4a7c15ULL * (index + 1);
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
ModePairs forcing_pairs(std::size_t n, const Parameters &p) {
    p.validate(n);
    ModePairs pairs;
    for (std::size_t y = 0; y < n; ++y)
        for (std::size_t x = 0; x < n; ++x) {
            const int a = wave_number(x, n), b = wave_number(y, n);
            const auto i = y * n + x, partner = ((n - y) % n) * n + (n - x) % n;
            const int k2 = a * a + b * b;
            if (3 * std::abs(a) < static_cast<int>(n) && 3 * std::abs(b) < static_cast<int>(n) &&
                i < partner && k2 >= p.forcing_min * p.forcing_min &&
                k2 <= p.forcing_max * p.forcing_max)
                pairs.emplace_back(i, partner);
        }
    if (pairs.empty())
        throw std::invalid_argument("empty forcing shell");
    return pairs;
}
void initialize_ou(Field &field, const ModePairs &pairs, const Parameters &p,
                   NormalStream &random) {
    const double scale = p.forcing_sigma * std::sqrt(p.forcing_tau / 4);
    for (auto [i, j] : pairs) {
        const double re = random(), im = random();
        field[i] = scale * Complex{re, im};
        field[j] = std::conj(field[i]);
    }
}
void transition_ou(Field &field, const ModePairs &pairs, const Parameters &p, NormalStream &random,
                   double dt) {
    if (!std::isfinite(dt) || dt <= 0)
        throw std::invalid_argument("invalid OU transition interval");
    const double decay = std::exp(-dt / p.forcing_tau),
                 scale = p.forcing_sigma *
                         std::sqrt(-p.forcing_tau * std::expm1(-2 * dt / p.forcing_tau) / 4);
    for (auto [i, j] : pairs) {
        const double re = random(), im = random();
        field[i] = decay * field[i] + scale * Complex{re, im};
        field[j] = std::conj(field[i]);
    }
}
ShellForcing::ShellForcing(std::size_t n, Parameters p, std::uint64_t seed)
    : parameters_(p), random_(seed) {
    Fourier2D validate_grid(n);
    pairs_ = forcing_pairs(n, p);
    field_.resize(n * n);
    midpoint_.resize(n * n);
    initialize_ou(field_, pairs_, p, random_);
}
const Field &ShellForcing::midpoint_step(double dt) {
    transition_ou(field_, pairs_, parameters_, random_, dt / 2);
    midpoint_ = field_;
    transition_ou(field_, pairs_, parameters_, random_, dt / 2);
    return midpoint_;
}
Solver::Solver(std::size_t n, Parameters p, std::uint64_t seed)
    : n_(n), parameters_(p), fft_(n), random_(seed), omega_(n * n), forcing_(n * n),
      midpoint_forcing_(n * n), u_(n * n), v_(n * n), wx_(n * n), wy_(n * n), nonlinear_(n * n),
      temporary_(n * n), k1_(n * n), k2_(n * n), k3_(n * n), k4_(n * n), kx_(n * n), ky_(n * n),
      ksq_(n * n), active_(n * n) {
    p.validate(n);
    for (std::size_t y = 0; y < n; ++y)
        for (std::size_t x = 0; x < n; ++x) {
            const auto i = y * n + x;
            const int a = wave_number(x, n), b = wave_number(y, n);
            kx_[i] = a;
            ky_[i] = b;
            ksq_[i] = a * a + b * b;
            active_[i] = 3 * std::abs(a) < static_cast<int>(n) &&
                         3 * std::abs(b) < static_cast<int>(n) && i != 0;
        }
    forced_pairs_ = forcing_pairs(n, p);
    // Start the OU process at stationarity; each complex mode has E|F|^2 = sigma^2 tau/2.
    initialize_ou(forcing_, forced_pairs_, p, random_);
}
void Solver::project(Field &field) const {
    for (std::size_t i = 0; i < field.size(); ++i)
        if (!active_[i])
            field[i] = 0;
}
void Solver::initialize(const std::string &name) {
    if (name != "taylor-green" && name != "multimode" && name != "zero")
        throw std::invalid_argument("unknown initial condition");
    if (time_ != 0)
        throw std::logic_error("cannot reinitialize an evolved solver");
    for (std::size_t y = 0; y < n_; ++y)
        for (std::size_t x = 0; x < n_; ++x) {
            const double a = 2 * pi * static_cast<double>(x) / static_cast<double>(n_),
                         b = 2 * pi * static_cast<double>(y) / static_cast<double>(n_);
            omega_[y * n_ + x] = name == "zero" ? 0.0 : 2 * std::cos(a) * std::cos(b);
            if (name == "multimode")
                omega_[y * n_ + x] += 0.6 * std::sin(2 * a + b) + 0.4 * std::cos(a - 3 * b) +
                                      0.3 * std::sin(3 * a + 2 * b);
        }
    fft_.forward(omega_);
    project(omega_);
}
void Solver::initialize_physical(const Field &vorticity) {
    if (vorticity.size() != omega_.size())
        throw std::invalid_argument("physical initial field size mismatch");
    if (time_ != 0)
        throw std::logic_error("cannot reinitialize an evolved solver");
    omega_ = vorticity;
    for (const auto &value : omega_)
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()) || value.imag() != 0)
            throw std::invalid_argument("physical initial vorticity must be finite and real");
    fft_.forward(omega_);
    project(omega_);
}
void Solver::advance_forcing(double dt) {
    transition_ou(forcing_, forced_pairs_, parameters_, random_, dt);
}
void Solver::velocity(const Field &w) {
    for (std::size_t i = 0; i < w.size(); ++i) {
        const auto psi = ksq_[i] > 0 ? w[i] / ksq_[i] : Complex{};
        u_[i] = imaginary * ky_[i] * psi;
        v_[i] = -imaginary * kx_[i] * psi;
    }
    fft_.inverse(u_);
    fft_.inverse(v_);
}
void Solver::rhs(const Field &state, Field &result) {
    velocity(state);
    for (std::size_t i = 0; i < state.size(); ++i) {
        wx_[i] = imaginary * kx_[i] * state[i];
        wy_[i] = imaginary * ky_[i] * state[i];
    }
    fft_.inverse(wx_);
    fft_.inverse(wy_);
    for (std::size_t i = 0; i < state.size(); ++i)
        nonlinear_[i] = u_[i].real() * wx_[i].real() + v_[i].real() * wy_[i].real();
    fft_.forward(nonlinear_);
    for (std::size_t i = 0; i < state.size(); ++i)
        result[i] = active_[i]
                        ? -nonlinear_[i] -
                              (parameters_.viscosity * ksq_[i] + parameters_.friction) * state[i] +
                              forcing_[i]
                        : Complex{};
}
void Solver::validate_step(double dt) {
    if (!std::isfinite(dt) || dt <= 0 || !std::isfinite(time_ + dt) || time_ + dt == time_)
        throw std::invalid_argument("invalid time step");
    const auto d = diagnostics();
    double max_linear = 0;
    for (std::size_t i = 0; i < omega_.size(); ++i)
        if (active_[i])
            max_linear =
                std::max(max_linear, parameters_.viscosity * ksq_[i] + parameters_.friction);
    if (dt * d.cfl_rate > 0.5 || dt * max_linear > 2.5)
        throw std::runtime_error(
            "time step exceeds conservative explicit stability limits; reduce dt");
}
void Solver::step(double dt, TimeIntegrator integrator) {
    validate_step(dt);
    advance_forcing(dt / 2);
    if (parameters_.forcing_sigma > 0)
        midpoint_forcing_ = forcing_;
    if (integrator == TimeIntegrator::rk4) {
        rhs(omega_, k1_);
        for (std::size_t i = 0; i < omega_.size(); ++i)
            temporary_[i] = omega_[i] + dt / 2 * k1_[i];
        rhs(temporary_, k2_);
        for (std::size_t i = 0; i < omega_.size(); ++i)
            temporary_[i] = omega_[i] + dt / 2 * k2_[i];
        rhs(temporary_, k3_);
        for (std::size_t i = 0; i < omega_.size(); ++i)
            temporary_[i] = omega_[i] + dt * k3_[i];
        rhs(temporary_, k4_);
        for (std::size_t i = 0; i < omega_.size(); ++i)
            omega_[i] += dt / 6 * (k1_[i] + 2.0 * k2_[i] + 2.0 * k3_[i] + k4_[i]);
    } else {
        // Shu--Osher SSPRK(3,3). temporary_ is u1 and k4_ is u2.
        rhs(omega_, k1_);
        for (std::size_t i = 0; i < omega_.size(); ++i)
            temporary_[i] = omega_[i] + dt * k1_[i];
        rhs(temporary_, k2_);
        for (std::size_t i = 0; i < omega_.size(); ++i)
            k4_[i] = 0.75 * omega_[i] + 0.25 * (temporary_[i] + dt * k2_[i]);
        rhs(k4_, k3_);
        for (std::size_t i = 0; i < omega_.size(); ++i)
            omega_[i] = omega_[i] / 3.0 + 2.0 / 3.0 * (k4_[i] + dt * k3_[i]);
    }
    for (std::size_t i = 0; i < omega_.size(); ++i) {
        if (!std::isfinite(omega_[i].real()) || !std::isfinite(omega_[i].imag()))
            throw std::runtime_error("nonfinite flow state");
    }
    project(omega_);
    advance_forcing(dt / 2);
    time_ += dt;
}
Field Solver::drift(const Field &state) {
    if (state.size() != omega_.size())
        throw std::invalid_argument("drift state size mismatch");
    Field result(state.size());
    rhs(state, result);
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] -= forcing_[i];
    return result;
}
Field Solver::linearized_drift(const Field &state, const Field &direction) {
    if (state.size() != omega_.size() || direction.size() != state.size())
        throw std::invalid_argument("tangent state size mismatch");
    velocity(state);
    const auto base_u = u_, base_v = v_;
    for (std::size_t i = 0; i < state.size(); ++i) {
        wx_[i] = imaginary * kx_[i] * state[i];
        wy_[i] = imaginary * ky_[i] * state[i];
    }
    fft_.inverse(wx_);
    fft_.inverse(wy_);
    velocity(direction);
    for (std::size_t i = 0; i < state.size(); ++i)
        nonlinear_[i] = u_[i].real() * wx_[i].real() + v_[i].real() * wy_[i].real();
    for (std::size_t i = 0; i < state.size(); ++i) {
        wx_[i] = imaginary * kx_[i] * direction[i];
        wy_[i] = imaginary * ky_[i] * direction[i];
    }
    fft_.inverse(wx_);
    fft_.inverse(wy_);
    for (std::size_t i = 0; i < state.size(); ++i)
        nonlinear_[i] += base_u[i].real() * wx_[i].real() + base_v[i].real() * wy_[i].real();
    fft_.forward(nonlinear_);
    Field result(state.size());
    for (std::size_t i = 0; i < state.size(); ++i)
        if (active_[i])
            result[i] = -nonlinear_[i] -
                        (parameters_.viscosity * ksq_[i] + parameters_.friction) * direction[i];
    return result;
}
Field Solver::adjoint_drift(const Field &state, const Field &cotangent) {
    if (state.size() != omega_.size() || cotangent.size() != state.size())
        throw std::invalid_argument("adjoint state size mismatch");
    velocity(state);
    Field physical = cotangent, lambda_x(state.size()), lambda_y(state.size());
    for (std::size_t i = 0; i < state.size(); ++i) {
        wx_[i] = imaginary * kx_[i] * state[i];
        wy_[i] = imaginary * ky_[i] * state[i];
        lambda_x[i] = imaginary * kx_[i] * cotangent[i];
        lambda_y[i] = imaginary * ky_[i] * cotangent[i];
    }
    fft_.inverse(wx_);
    fft_.inverse(wy_);
    fft_.inverse(physical);
    fft_.inverse(lambda_x);
    fft_.inverse(lambda_y);
    for (std::size_t i = 0; i < state.size(); ++i) {
        nonlinear_[i] = u_[i].real() * lambda_x[i].real() + v_[i].real() * lambda_y[i].real();
        wx_[i] = physical[i].real() * wx_[i].real();
        wy_[i] = physical[i].real() * wy_[i].real();
    }
    fft_.forward(nonlinear_);
    fft_.forward(wx_);
    fft_.forward(wy_);
    Field result(state.size());
    for (std::size_t i = 0; i < state.size(); ++i)
        if (active_[i])
            result[i] = nonlinear_[i] + imaginary * ky_[i] / ksq_[i] * wx_[i] -
                        imaginary * kx_[i] / ksq_[i] * wy_[i] -
                        (parameters_.viscosity * ksq_[i] + parameters_.friction) * cotangent[i];
    return result;
}
void Solver::step_tangent(double dt, std::array<Field, 2> &sensitivities) {
    if (parameters_.forcing_sigma != 0)
        throw std::invalid_argument("tangent implementation currently requires zero forcing");
    for (const auto &field : sensitivities)
        if (field.size() != omega_.size())
            throw std::invalid_argument("sensitivity size mismatch");
    validate_step(dt);
    using Tangents = std::array<Field, 2>;
    const auto slope = [&](const Field &state, const Tangents &tangent) {
        Tangents result;
        for (std::size_t j = 0; j < 2; ++j) {
            result[j] = linearized_drift(state, tangent[j]);
            for (std::size_t i = 0; i < state.size(); ++i)
                result[j][i] -=
                    (j == 0 ? parameters_.viscosity * ksq_[i] : parameters_.friction) * state[i];
        }
        return result;
    };
    const auto stage_tangents = [&](const Tangents &derivative, double scale) {
        auto result = sensitivities;
        for (std::size_t j = 0; j < 2; ++j)
            for (std::size_t i = 0; i < omega_.size(); ++i)
                result[j][i] += scale * derivative[j][i];
        return result;
    };
    rhs(omega_, k1_);
    const auto t1 = slope(omega_, sensitivities);
    for (std::size_t i = 0; i < omega_.size(); ++i)
        temporary_[i] = omega_[i] + dt / 2 * k1_[i];
    rhs(temporary_, k2_);
    const auto t2 = slope(temporary_, stage_tangents(t1, dt / 2));
    for (std::size_t i = 0; i < omega_.size(); ++i)
        temporary_[i] = omega_[i] + dt / 2 * k2_[i];
    rhs(temporary_, k3_);
    const auto t3 = slope(temporary_, stage_tangents(t2, dt / 2));
    for (std::size_t i = 0; i < omega_.size(); ++i)
        temporary_[i] = omega_[i] + dt * k3_[i];
    rhs(temporary_, k4_);
    const auto t4 = slope(temporary_, stage_tangents(t3, dt));
    for (std::size_t i = 0; i < omega_.size(); ++i) {
        omega_[i] += dt / 6 * (k1_[i] + 2.0 * k2_[i] + 2.0 * k3_[i] + k4_[i]);
        for (std::size_t j = 0; j < 2; ++j) {
            sensitivities[j][i] += dt / 6 * (t1[j][i] + 2.0 * t2[j][i] + 2.0 * t3[j][i] + t4[j][i]);
            if (!std::isfinite(sensitivities[j][i].real()) ||
                !std::isfinite(sensitivities[j][i].imag()))
                throw std::runtime_error("nonfinite sensitivity");
        }
        if (!std::isfinite(omega_[i].real()) || !std::isfinite(omega_[i].imag()))
            throw std::runtime_error("nonfinite tangent flow state");
    }
    project(omega_);
    for (auto &field : sensitivities)
        project(field);
    time_ += dt;
}
Diagnostics Solver::diagnostics() {
    Diagnostics d;
    d.time = time_;
    const double area = 4 * pi * pi;
    for (std::size_t i = 1; i < omega_.size(); ++i) {
        const double norm = std::norm(omega_[i]);
        d.energy += area / 2 * norm / ksq_[i];
        d.enstrophy += area / 2 * norm;
        d.injection += area * std::real(std::conj(omega_[i]) * forcing_[i]) / ksq_[i];
    }
    d.dissipation = 2 * parameters_.viscosity * d.enstrophy;
    velocity(omega_);
    for (std::size_t i = 0; i < omega_.size(); ++i)
        d.cfl_rate = std::max(d.cfl_rate, (std::abs(u_[i].real()) + std::abs(v_[i].real())) *
                                              static_cast<double>(n_) / (2 * pi));
    return d;
}
Field Solver::physical_vorticity() {
    auto field = omega_;
    fft_.inverse(field);
    return field;
}
void Solver::update_latent_state(const Field &vorticity, const Field &forcing) {
    if (vorticity.size() != omega_.size() || forcing.size() != forcing_.size())
        throw std::invalid_argument("latent state size mismatch");
    for (std::size_t i = 0; i < vorticity.size(); ++i)
        if (!std::isfinite(vorticity[i].real()) || !std::isfinite(vorticity[i].imag()) ||
            !std::isfinite(forcing[i].real()) || !std::isfinite(forcing[i].imag()))
            throw std::invalid_argument("nonfinite assimilation state");
    omega_ = vorticity;
    project(omega_);
    std::fill(forcing_.begin(), forcing_.end(), Complex{});
    for (auto [i, j] : forced_pairs_) {
        forcing_[i] = 0.5 * (forcing[i] + std::conj(forcing[j]));
        forcing_[j] = std::conj(forcing_[i]);
    }
    for (std::size_t y = 0; y < n_; ++y)
        for (std::size_t x = 0; x < n_; ++x) {
            const auto i = y * n_ + x, j = ((n_ - y) % n_) * n_ + (n_ - x) % n_;
            if (i < j) {
                omega_[i] = 0.5 * (omega_[i] + std::conj(omega_[j]));
                omega_[j] = std::conj(omega_[i]);
            }
        }
}
std::vector<double> Solver::energy_spectrum() const {
    std::vector<double> result(n_);
    for (std::size_t i = 1; i < omega_.size(); ++i) {
        const auto shell = static_cast<std::size_t>(std::floor(std::sqrt(ksq_[i]) + 0.5));
        result[shell] += 2 * pi * pi * std::norm(omega_[i]) / ksq_[i];
    }
    return result;
}
void Solver::save(const std::filesystem::path &path) const {
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("cannot open checkpoint for writing");
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    const auto &p = parameters_;
    out << "HZL_CHECKPOINT_V1\n"
        << n_ << ' ' << time_ << '\n'
        << p.viscosity << ' ' << p.friction << ' ' << p.forcing_sigma << ' ' << p.forcing_tau << ' '
        << p.forcing_min << ' ' << p.forcing_max << '\n';
    random_.write(out);
    for (std::size_t i = 0; i < omega_.size(); ++i)
        out << omega_[i].real() << ' ' << omega_[i].imag() << ' ' << forcing_[i].real() << ' '
            << forcing_[i].imag() << '\n';
    if (!out)
        throw std::runtime_error("checkpoint write failed");
}
Solver Solver::load(const std::filesystem::path &path) {
    std::ifstream in(path);
    std::string magic;
    std::size_t n = 0;
    double t = 0;
    Parameters p;
    if (!(in >> magic >> n >> t >> p.viscosity >> p.friction >> p.forcing_sigma >> p.forcing_tau >>
          p.forcing_min >> p.forcing_max) ||
        magic != "HZL_CHECKPOINT_V1" || !std::isfinite(t) || t < 0)
        throw std::runtime_error("invalid checkpoint header");
    Solver s(n, p, 0);
    s.time_ = t;
    s.random_.read(in);
    for (std::size_t i = 0; i < s.omega_.size(); ++i) {
        double a = 0, b = 0, c = 0, d = 0;
        if (!(in >> a >> b >> c >> d) || !std::isfinite(a) || !std::isfinite(b) ||
            !std::isfinite(c) || !std::isfinite(d))
            throw std::runtime_error("invalid checkpoint field");
        s.omega_[i] = {a, b};
        s.forcing_[i] = {c, d};
    }
    std::vector<bool> forced(n * n, false);
    for (auto [i, j] : s.forced_pairs_) {
        forced[i] = true;
        forced[j] = true;
    }
    for (std::size_t y = 0; y < n; ++y)
        for (std::size_t x = 0; x < n; ++x) {
            const auto i = y * n + x, partner = ((n - y) % n) * n + (n - x) % n;
            if ((!s.active_[i] && s.omega_[i] != Complex{}) ||
                (!forced[i] && s.forcing_[i] != Complex{}) ||
                std::abs(s.omega_[i] - std::conj(s.omega_[partner])) >
                    1e-10 * (1 + std::abs(s.omega_[i])) ||
                s.forcing_[i] != std::conj(s.forcing_[partner]))
                throw std::runtime_error("checkpoint violates spectral constraints");
        }
    std::string trailing;
    if (in >> trailing)
        throw std::runtime_error("unexpected checkpoint data");
    return s;
}
} // namespace hzl
