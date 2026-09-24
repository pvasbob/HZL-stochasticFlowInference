#include "hzl/cuda_solver.hpp"
#include "hzl/inference.hpp"
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;
template <class T> T parse(const char *text) {
    T value{};
    const std::string input = text;
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
    if (error != std::errc{} || end != input.data() + input.size())
        throw std::invalid_argument("invalid ensemble benchmark argument");
    return value;
}
double energy(const hzl::Field &vorticity, std::size_t n) {
    double result = 0;
    for (std::size_t y = 0; y < n; ++y)
        for (std::size_t x = 0; x < n; ++x) {
            const int kx = hzl::wave_number(x, n), ky = hzl::wave_number(y, n),
                      k2 = kx * kx + ky * ky;
            if (k2 > 0)
                result +=
                    2 * std::numbers::pi * std::numbers::pi * std::norm(vorticity[y * n + x]) / k2;
        }
    return result;
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc != 6) {
            std::cerr << "usage: hzl-cuda-ensemble-benchmark GRID STEPS DT MEMBERS CPU_WORKERS\n";
            return 1;
        }
        const auto n = parse<std::size_t>(argv[1]), steps = parse<std::size_t>(argv[2]),
                   count = parse<std::size_t>(argv[4]), workers = parse<std::size_t>(argv[5]);
        const double dt = parse<double>(argv[3]);
        if (steps == 0 || count == 0 || workers == 0 || dt <= 0)
            throw std::invalid_argument("positive steps, dt, members, and workers required");
        hzl::Parameters p;
        constexpr std::uint64_t seed = 42;
        const hzl::Field zero(n * n);
        std::vector<std::unique_ptr<hzl::CudaSolver>> solvers;
        std::vector<std::unique_ptr<hzl::ShellForcing>> forcing;
        solvers.reserve(count);
        forcing.reserve(count);
        const auto setup_start = Clock::now();
        for (std::size_t member = 0; member < count; ++member) {
            const auto member_seed = hzl::stream_seed(seed, member);
            solvers.push_back(std::make_unique<hzl::CudaSolver>(n, p, zero));
            forcing.push_back(std::make_unique<hzl::ShellForcing>(n, p, member_seed));
        }
        for (auto &solver : solvers)
            solver->synchronize();
        const double setup_seconds =
            std::chrono::duration<double>(Clock::now() - setup_start).count();
        const auto evolution_start = Clock::now();
        for (std::size_t step = 0; step < steps; ++step)
            for (std::size_t member = 0; member < count; ++member)
                solvers[member]->step(dt, forcing[member]->midpoint_step(dt));
        for (auto &solver : solvers)
            solver->synchronize();
        const double evolution_seconds =
            std::chrono::duration<double>(Clock::now() - evolution_start).count();
        const auto download_start = Clock::now();
        std::vector<double> gpu_energy(count);
        for (std::size_t member = 0; member < count; ++member)
            gpu_energy[member] = energy(solvers[member]->coefficients(), n);
        const double download_seconds =
            std::chrono::duration<double>(Clock::now() - download_start).count();
        const auto cpu_start = Clock::now();
        const auto cpu = hzl::ensemble(n, p, dt, steps, count, workers, seed);
        const double cpu_seconds = std::chrono::duration<double>(Clock::now() - cpu_start).count();
        double maximum_relative_error = 0;
        for (std::size_t member = 0; member < count; ++member)
            maximum_relative_error = std::max(maximum_relative_error,
                                              std::abs(gpu_energy[member] - cpu[member].energy) /
                                                  std::max(std::abs(cpu[member].energy), 1e-300));
        const double gpu_total = setup_seconds + evolution_seconds + download_seconds;
        std::cout << std::setprecision(17) << "{\"n\":" << n << ",\"steps\":" << steps
                  << ",\"dt\":" << dt << ",\"members\":" << count << ",\"cpu_workers\":" << workers
                  << ",\"gpu_setup_seconds\":" << setup_seconds
                  << ",\"gpu_evolution_seconds\":" << evolution_seconds
                  << ",\"gpu_download_and_diagnostics_seconds\":" << download_seconds
                  << ",\"gpu_total_seconds\":" << gpu_total << ",\"cpu_seconds\":" << cpu_seconds
                  << ",\"cpu_over_gpu_total\":" << cpu_seconds / gpu_total
                  << ",\"gpu_member_steps_per_second\":"
                  << static_cast<double>(count * steps) / evolution_seconds
                  << ",\"maximum_relative_energy_error\":" << maximum_relative_error
                  << ",\"gpu_execution\":\"multiple resident solver instances, sequential launch "
                     "per member; not batched kernels\"}\n";
        if (!std::isfinite(maximum_relative_error) || maximum_relative_error > 1e-9)
            throw std::runtime_error("CPU/GPU ensemble agreement tolerance exceeded");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
