#include "hzl/solver.hpp"
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
template <class T> T parse(const char *text) {
    T value{};
    const std::string input = text;
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
    if (error != std::errc{} || end != input.data() + input.size())
        throw std::invalid_argument("invalid MPI ensemble argument");
    return value;
}
} // namespace
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        if (argc != 5)
            throw std::invalid_argument("usage: hzl-mpi-ensemble GRID STEPS DT MEMBERS");
        const auto n = parse<std::size_t>(argv[1]), steps = parse<std::size_t>(argv[2]),
                   count = parse<std::size_t>(argv[4]);
        const double dt = parse<double>(argv[3]);
        if (steps == 0 || count == 0 || dt <= 0)
            throw std::invalid_argument("positive steps, dt, and members required");
        hzl::Parameters parameters;
        constexpr std::uint64_t seed = 42;
        std::vector<double> local_energy(count), local_enstrophy(count), local_dissipation(count);
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t member = static_cast<std::size_t>(rank); member < count;
             member += static_cast<std::size_t>(ranks)) {
            hzl::Solver solver(n, parameters, hzl::stream_seed(seed, member));
            solver.initialize("zero");
            for (std::size_t step = 0; step < steps; ++step)
                solver.step(dt);
            const auto diagnostics = solver.diagnostics();
            local_energy[member] = diagnostics.energy;
            local_enstrophy[member] = diagnostics.enstrophy;
            local_dissipation[member] = diagnostics.dissipation;
        }
        const double local_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::vector<double> energy(count), enstrophy(count), dissipation(count);
        MPI_Reduce(local_energy.data(), energy.data(), static_cast<int>(count), MPI_DOUBLE, MPI_SUM,
                   0, MPI_COMM_WORLD);
        MPI_Reduce(local_enstrophy.data(), enstrophy.data(), static_cast<int>(count), MPI_DOUBLE,
                   MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_dissipation.data(), dissipation.data(), static_cast<int>(count),
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        double wall_seconds = 0;
        MPI_Reduce(&local_seconds, &wall_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            const double mean =
                std::accumulate(energy.begin(), energy.end(), 0.0) / static_cast<double>(count);
            std::cout << std::setprecision(17) << "{\"ranks\":" << ranks << ",\"members\":" << count
                      << ",\"n\":" << n << ",\"steps\":" << steps << ",\"dt\":" << dt
                      << ",\"mean_energy\":" << mean << ",\"wall_seconds\":" << wall_seconds
                      << ",\"seed_policy\":\"logical member seeds independent of MPI rank\"}\n";
        }
    } catch (const std::exception &error) {
        std::cerr << "rank " << rank << " error: " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return 0;
}
