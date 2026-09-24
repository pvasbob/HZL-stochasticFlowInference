#include "hzl/inference.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace hzl {
namespace {
struct Stages {
    Field first, second, third, fourth, next;
};
Stages rk4_stages(Solver &model, const Field &initial, double dt) {
    Stages stages;
    stages.first = initial;
    stages.second = initial;
    stages.third = initial;
    stages.fourth = initial;
    stages.next = initial;
    const auto k1 = model.drift(initial);
    for (std::size_t i = 0; i < initial.size(); ++i)
        stages.second[i] += dt / 2 * k1[i];
    const auto k2 = model.drift(stages.second);
    for (std::size_t i = 0; i < initial.size(); ++i)
        stages.third[i] += dt / 2 * k2[i];
    const auto k3 = model.drift(stages.third);
    for (std::size_t i = 0; i < initial.size(); ++i)
        stages.fourth[i] += dt * k3[i];
    const auto k4 = model.drift(stages.fourth);
    for (std::size_t i = 0; i < initial.size(); ++i)
        stages.next[i] += dt / 6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    return stages;
}
void add(Field &destination, const Field &source, double scale = 1) {
    for (std::size_t i = 0; i < destination.size(); ++i)
        destination[i] += scale * source[i];
}
Field scaled(Field field, double scale) {
    for (auto &z : field)
        z *= scale;
    return field;
}
} // namespace
AdjointEvaluation likelihood_adjoint(const Dataset &data, const Vector &q, std::size_t stride) {
    data.validate();
    if (data.stochastic || q.size() != 2 || stride == 0)
        throw std::invalid_argument(
            "adjoint requires deterministic data, two parameters, and positive checkpoint stride");
    AdjointEvaluation result;
    result.objective.gradient.assign(2, 0);
    for (double value : q)
        if (!std::isfinite(value) || value < -12 || value > 0) {
            result.objective.value = std::numeric_limits<double>::infinity();
            return result;
        }
    Parameters p;
    p.viscosity = std::exp(q[0]);
    p.friction = std::exp(q[1]);
    p.forcing_sigma = 0;
    p.forcing_min = data.forcing_min;
    p.forcing_max = data.forcing_max;
    const double largest = static_cast<double>((data.n - 1) / 3);
    if (data.dt * (2 * p.viscosity * largest * largest + p.friction) > 2.5) {
        result.objective.value = std::numeric_limits<double>::infinity();
        return result;
    }
    Solver solver(data.n, p, 0);
    solver.initialize("multimode");
    Fourier2D fft(data.n);
    std::vector<Field> checkpoints{solver.coefficients()};
    std::vector<std::size_t> checkpoint_steps{0};
    Vector residuals(data.observations.size());
    std::size_t observation = 0;
    const auto last = data.observations.back().step;
    for (std::size_t step = 0; step <= last; ++step) {
        if (step > 0) {
            solver.step(data.dt);
            if (step % stride == 0 || step == last) {
                checkpoints.push_back(solver.coefficients());
                checkpoint_steps.push_back(step);
            }
        }
        if (observation < data.observations.size() && data.observations[observation].step == step) {
            const auto field = solver.physical_vorticity();
            while (observation < data.observations.size() &&
                   data.observations[observation].step == step) {
                const auto &o = data.observations[observation];
                const double residual =
                    (field[o.y * data.n + o.x].real() - o.value) / o.standard_deviation;
                result.objective.value += 0.5 * residual * residual;
                residuals[observation] = residual / o.standard_deviation;
                ++observation;
            }
        }
    }
    result.checkpoint_fields = checkpoints.size();
    Field cotangent(data.n * data.n, 0);
    observation = data.observations.size();
    const auto inject = [&](std::size_t step) {
        if (observation == 0 || data.observations[observation - 1].step != step)
            return;
        Field physical(data.n * data.n, 0);
        while (observation > 0 && data.observations[observation - 1].step == step) {
            --observation;
            const auto &o = data.observations[observation];
            physical[o.y * data.n + o.x] +=
                residuals[observation] * static_cast<double>(data.n * data.n);
        }
        fft.forward(physical);
        for (std::size_t y = 0; y < data.n; ++y)
            for (std::size_t x = 0; x < data.n; ++x) {
                const auto i = y * data.n + x;
                const int a = wave_number(x, data.n), b = wave_number(y, data.n);
                if (i != 0 && 3 * std::abs(a) < static_cast<int>(data.n) &&
                    3 * std::abs(b) < static_cast<int>(data.n))
                    cotangent[i] += physical[i];
            }
    };
    const auto parameter_contribution = [&](const Field &state, const Field &weight) {
        for (std::size_t i = 0; i < state.size(); ++i) {
            const double a = wave_number(i % data.n, data.n), b = wave_number(i / data.n, data.n),
                         inner = std::real(std::conj(weight[i]) * state[i]);
            result.objective.gradient[0] -= p.viscosity * (a * a + b * b) * inner;
            result.objective.gradient[1] -= p.friction * inner;
        }
    };
    for (std::size_t block = checkpoints.size(); block > 1; --block) {
        const auto begin = checkpoint_steps[block - 2], end = checkpoint_steps[block - 1];
        std::vector<Field> states;
        states.reserve(end - begin + 1);
        states.push_back(checkpoints[block - 2]);
        for (std::size_t step = begin; step < end; ++step) {
            states.push_back(rk4_stages(solver, states.back(), data.dt).next);
            ++result.replayed_steps;
        }
        result.peak_block_fields = std::max(result.peak_block_fields, states.size());
        for (std::size_t step = end; step > begin; --step) {
            inject(step);
            const auto stages = rk4_stages(solver, states[step - begin - 1], data.dt);
            ++result.replayed_steps;
            auto previous = cotangent, k1 = scaled(cotangent, data.dt / 6),
                 k2 = scaled(cotangent, data.dt / 3), k3 = scaled(cotangent, data.dt / 3),
                 k4 = scaled(cotangent, data.dt / 6);
            parameter_contribution(stages.fourth, k4);
            const auto b4 = solver.adjoint_drift(stages.fourth, k4);
            add(previous, b4);
            add(k3, b4, data.dt);
            parameter_contribution(stages.third, k3);
            const auto b3 = solver.adjoint_drift(stages.third, k3);
            add(previous, b3);
            add(k2, b3, data.dt / 2);
            parameter_contribution(stages.second, k2);
            const auto b2 = solver.adjoint_drift(stages.second, k2);
            add(previous, b2);
            add(k1, b2, data.dt / 2);
            parameter_contribution(stages.first, k1);
            add(previous, solver.adjoint_drift(stages.first, k1));
            cotangent = std::move(previous);
        }
    }
    return result;
}
} // namespace hzl
