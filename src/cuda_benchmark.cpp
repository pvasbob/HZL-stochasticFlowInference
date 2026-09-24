#include "hzl/cuda_solver.hpp"
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
template <class T> T parse(const char *text) {
    T value{};
    const std::string s = text;
    const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
    if (ec != std::errc{} || end != s.data() + s.size())
        throw std::invalid_argument("invalid benchmark argument");
    return value;
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc != 4 && argc != 5) {
            std::cerr << "usage: hzl-cuda-benchmark GRID STEPS DT [FORCING_SIGMA]\n";
            return 1;
        }
        const auto n = parse<std::size_t>(argv[1]), steps = parse<std::size_t>(argv[2]);
        const double dt = parse<double>(argv[3]);
        if (steps == 0)
            throw std::invalid_argument("steps must be positive");
        hzl::Parameters p;
        p.forcing_sigma = argc == 5 ? parse<double>(argv[4]) : 0;
        p.viscosity = 0.001;
        hzl::Solver cpu(n, p, 0);
        cpu.initialize("multimode");
        const auto initial = cpu.coefficients();
        const auto gpu_total_start = Clock::now();
        hzl::CudaSolver gpu(n, p, initial);
        hzl::ShellForcing forcing(n, p, 0);
        gpu.synchronize();
        const double setup = elapsed(gpu_total_start);
        const auto gpu_start = Clock::now();
        for (std::size_t i = 0; i < steps; ++i) {
            if (p.forcing_sigma > 0)
                gpu.step(dt, forcing.midpoint_step(dt));
            else
                gpu.step(dt);
        }
        gpu.synchronize();
        const double gpu_seconds = elapsed(gpu_start);
        const auto download_start = Clock::now();
        const auto result = gpu.coefficients();
        const double download = elapsed(download_start), gpu_total = elapsed(gpu_total_start);
        const auto cpu_start = Clock::now();
        for (std::size_t i = 0; i < steps; ++i)
            cpu.step(dt);
        const double cpu_seconds = elapsed(cpu_start);
        double numerator = 0, denominator = 0;
        for (std::size_t i = 0; i < result.size(); ++i) {
            numerator += std::norm(result[i] - cpu.coefficients()[i]);
            denominator += std::norm(cpu.coefficients()[i]);
        }
        const double relative_error = std::sqrt(numerator / denominator);
        std::cout << std::setprecision(17) << "{\n\"n\":" << n << ",\n\"steps\":" << steps
                  << ",\n\"dt\":" << dt
                  << ",\n\"nu\":0.001,\n\"alpha\":0.1,\n\"sigma\":" << p.forcing_sigma
                  << ",\n\"tau\":0.5,\n"
                  << "\"cpu_evolution_seconds\":" << cpu_seconds
                  << ",\n\"gpu_setup_and_upload_seconds\":" << setup
                  << ",\n\"gpu_evolution_including_host_forcing_and_transfers_seconds\":"
                  << gpu_seconds << ",\n\"gpu_final_download_seconds\":" << download
                  << ",\n\"gpu_total_seconds\":" << gpu_total
                  << ",\n\"cpu_evolution_over_gpu_total\":" << cpu_seconds / gpu_total
                  << ",\n\"evolution_ratio\":" << cpu_seconds / gpu_seconds
                  << ",\n\"relative_l2_error\":" << relative_error << "\n}\n";
        if (!std::isfinite(relative_error) || relative_error > 1e-9)
            throw std::runtime_error("CPU/GPU agreement tolerance exceeded");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
