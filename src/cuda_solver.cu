#include "hzl/cuda_solver.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace hzl {
namespace {
void check(cudaError_t result, const char *operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
void fft_check(cufftResult result, const char *operation) {
    if (result != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuFFT error " +
                                 std::to_string(result));
}
__device__ int wave(int i, int n) {
    return i < n / 2 ? i : i - n;
}
__device__ bool active(int x, int y, int n) {
    return 3 * abs(x) < n && 3 * abs(y) < n && (x != 0 || y != 0);
}
__global__ void derivatives(const cufftDoubleComplex *state, cufftDoubleComplex *fields, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, count = n * n;
    if (i >= count)
        return;
    const int kx = wave(i % n, n), ky = wave(i / n, n);
    const double k2 = kx * kx + ky * ky;
    const auto z = state[i];
    const double inv = k2 > 0 ? 1 / k2 : 0;
    fields[i] = {-ky * z.y * inv, ky * z.x * inv};
    fields[count + i] = {kx * z.y * inv, -kx * z.x * inv};
    fields[2 * count + i] = {-kx * z.y, kx * z.x};
    fields[3 * count + i] = {-ky * z.y, ky * z.x};
}
__global__ void advection(const cufftDoubleComplex *fields, cufftDoubleComplex *nonlinear,
                          int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    nonlinear[i] = {
        fields[i].x * fields[2 * count + i].x + fields[count + i].x * fields[3 * count + i].x, 0};
}
__global__ void cfl_max(const cufftDoubleComplex *fields, unsigned long long *maximum, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * n)
        return;
    const double rate =
        (fabs(fields[i].x) + fabs(fields[n * n + i].x)) * n / (2 * 3.14159265358979323846);
    atomicMax(maximum, static_cast<unsigned long long>(__double_as_longlong(rate)));
}
__global__ void assemble_rhs(const cufftDoubleComplex *state, const cufftDoubleComplex *nonlinear,
                             const cufftDoubleComplex *forcing, cufftDoubleComplex *result, int n,
                             double nu, double alpha) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * n)
        return;
    const int x = wave(i % n, n), y = wave(i / n, n);
    const double rate = nu * (x * x + y * y) + alpha;
    if (active(x, y, n))
        result[i] = {-nonlinear[i].x / (n * n) - rate * state[i].x + forcing[i].x,
                     -nonlinear[i].y / (n * n) - rate * state[i].y + forcing[i].y};
    else
        result[i] = {0, 0};
}
__global__ void stage(const cufftDoubleComplex *state, const cufftDoubleComplex *slope,
                      cufftDoubleComplex *temporary, double dt, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    temporary[i] = {state[i].x + dt * slope[i].x, state[i].y + dt * slope[i].y};
}
__global__ void finish(cufftDoubleComplex *state, const cufftDoubleComplex *k1,
                       const cufftDoubleComplex *k2, const cufftDoubleComplex *k3,
                       const cufftDoubleComplex *k4, double dt, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    state[i].x += dt / 6 * (k1[i].x + 2 * k2[i].x + 2 * k3[i].x + k4[i].x);
    state[i].y += dt / 6 * (k1[i].y + 2 * k2[i].y + 2 * k3[i].y + k4[i].y);
}
} // namespace
struct CudaSolver::Implementation {
    int n = 0, count = 0, blocks = 0;
    Parameters p;
    cufftHandle inverse = 0, forward = 0;
    cufftDoubleComplex *storage = nullptr, *state = nullptr, *fields = nullptr,
                       *nonlinear = nullptr, *temporary = nullptr, *k1 = nullptr, *k2 = nullptr,
                       *k3 = nullptr, *k4 = nullptr, *forcing = nullptr;
    std::vector<cufftDoubleComplex> upload;
    bool external_forcing = false;
    unsigned long long *maximum = nullptr;
    ~Implementation() {
        if (inverse)
            cufftDestroy(inverse);
        if (forward)
            cufftDestroy(forward);
        if (storage)
            cudaFree(storage);
        if (maximum)
            cudaFree(maximum);
    }
    void rhs(const cufftDoubleComplex *source, cufftDoubleComplex *result, bool stability,
             double dt) {
        derivatives<<<blocks, 256>>>(source, fields, n);
        fft_check(cufftExecZ2Z(inverse, fields, fields, CUFFT_INVERSE), "inverse FFT");
        if (stability) {
            check(cudaMemset(maximum, 0, sizeof(unsigned long long)), "reset CFL reduction");
            cfl_max<<<blocks, 256>>>(fields, maximum, n);
            double rate = 0;
            check(cudaMemcpy(&rate, maximum, sizeof(double), cudaMemcpyDeviceToHost),
                  "CFL scalar transfer");
            if (!std::isfinite(rate) || dt * rate > 0.5)
                throw std::runtime_error("GPU CFL limit exceeded");
        }
        advection<<<blocks, 256>>>(fields, nonlinear, count);
        fft_check(cufftExecZ2Z(forward, nonlinear, nonlinear, CUFFT_FORWARD), "forward FFT");
        assemble_rhs<<<blocks, 256>>>(source, nonlinear, forcing, result, n, p.viscosity,
                                      p.friction);
        check(cudaGetLastError(), "RHS kernels");
    }
};
CudaSolver::CudaSolver(std::size_t n, Parameters p, const Field &initial)
    : implementation_(std::make_unique<Implementation>()) {
    Fourier2D validate_grid(n);
    p.validate(n);
    if (initial.size() != n * n)
        throw std::invalid_argument("CUDA initial coefficient size mismatch");
    auto &s = *implementation_;
    s.n = static_cast<int>(n);
    s.count = s.n * s.n;
    s.blocks = (s.count + 255) / 256;
    s.p = p;
    check(
        cudaMalloc(reinterpret_cast<void **>(&s.storage), 12 * n * n * sizeof(cufftDoubleComplex)),
        "allocate GPU state and workspaces");
    check(cudaMalloc(reinterpret_cast<void **>(&s.maximum), sizeof(unsigned long long)),
          "allocate CFL reduction");
    s.state = s.storage;
    s.fields = s.state + s.count;
    s.nonlinear = s.fields + 4 * s.count;
    s.temporary = s.nonlinear + s.count;
    s.k1 = s.temporary + s.count;
    s.k2 = s.k1 + s.count;
    s.k3 = s.k2 + s.count;
    s.k4 = s.k3 + s.count;
    s.forcing = s.k4 + s.count;
    s.upload.resize(n * n);
    check(cudaMemset(s.forcing, 0, n * n * sizeof(cufftDoubleComplex)), "initialize GPU forcing");
    int dimensions[2] = {s.n, s.n};
    fft_check(cufftPlanMany(&s.inverse, 2, dimensions, nullptr, 1, s.count, nullptr, 1, s.count,
                            CUFFT_Z2Z, 4),
              "plan batched inverse FFT");
    fft_check(cufftPlan2d(&s.forward, s.n, s.n, CUFFT_Z2Z), "plan forward FFT");
    std::vector<cufftDoubleComplex> values(n * n);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const int x = wave_number(i % n, n), y = wave_number(i / n, n);
        if (!std::isfinite(initial[i].real()) || !std::isfinite(initial[i].imag()))
            throw std::invalid_argument("nonfinite initial GPU state");
        const bool keep = 3 * std::abs(x) < static_cast<int>(n) &&
                          3 * std::abs(y) < static_cast<int>(n) && i != 0;
        values[i] = keep ? cufftDoubleComplex{initial[i].real(), initial[i].imag()}
                         : cufftDoubleComplex{0, 0};
    }
    check(cudaMemcpy(s.state, values.data(), values.size() * sizeof(cufftDoubleComplex),
                     cudaMemcpyHostToDevice),
          "initial state upload");
}
CudaSolver::~CudaSolver() = default;
void CudaSolver::step(double dt) {
    auto &s = *implementation_;
    if (s.p.forcing_sigma != 0)
        throw std::invalid_argument("stochastic GPU step requires midpoint forcing coefficients");
    if (s.external_forcing) {
        check(cudaMemset(s.forcing, 0, s.upload.size() * sizeof(cufftDoubleComplex)),
              "clear GPU forcing");
        s.external_forcing = false;
    }
    advance(dt);
}
void CudaSolver::step(double dt, const Field &forcing) {
    auto &s = *implementation_;
    if (forcing.size() != s.upload.size())
        throw std::invalid_argument("GPU forcing size mismatch");
    for (std::size_t i = 0; i < forcing.size(); ++i) {
        if (!std::isfinite(forcing[i].real()) || !std::isfinite(forcing[i].imag()))
            throw std::invalid_argument("nonfinite GPU forcing");
        const auto n = static_cast<std::size_t>(s.n), x = i % n, y = i / n,
                   partner = ((n - y) % n) * n + (n - x) % n;
        if (std::abs(forcing[i] - std::conj(forcing[partner])) > 1e-12 * (1 + std::abs(forcing[i])))
            throw std::invalid_argument("GPU forcing must preserve conjugate symmetry");
        s.upload[i] = {forcing[i].real(), forcing[i].imag()};
    }
    check(cudaMemcpy(s.forcing, s.upload.data(), s.upload.size() * sizeof(cufftDoubleComplex),
                     cudaMemcpyHostToDevice),
          "midpoint forcing upload");
    s.external_forcing = true;
    advance(dt);
}
void CudaSolver::advance(double dt) {
    auto &s = *implementation_;
    const int largest = (s.n - 1) / 3;
    if (!std::isfinite(dt) || dt <= 0 ||
        dt * (2 * s.p.viscosity * largest * largest + s.p.friction) > 2.5)
        throw std::invalid_argument("invalid GPU time step or diffusion stability limit exceeded");
    s.rhs(s.state, s.k1, true, dt);
    stage<<<s.blocks, 256>>>(s.state, s.k1, s.temporary, dt / 2, s.count);
    s.rhs(s.temporary, s.k2, false, dt);
    stage<<<s.blocks, 256>>>(s.state, s.k2, s.temporary, dt / 2, s.count);
    s.rhs(s.temporary, s.k3, false, dt);
    stage<<<s.blocks, 256>>>(s.state, s.k3, s.temporary, dt, s.count);
    s.rhs(s.temporary, s.k4, false, dt);
    finish<<<s.blocks, 256>>>(s.state, s.k1, s.k2, s.k3, s.k4, dt, s.count);
    check(cudaGetLastError(), "RK4 kernels");
}
void CudaSolver::synchronize() {
    check(cudaDeviceSynchronize(), "synchronize GPU");
}
Field CudaSolver::coefficients() const {
    const auto &s = *implementation_;
    std::vector<cufftDoubleComplex> values(static_cast<std::size_t>(s.count));
    check(cudaMemcpy(values.data(), s.state, values.size() * sizeof(cufftDoubleComplex),
                     cudaMemcpyDeviceToHost),
          "final state download");
    Field result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i].x) || !std::isfinite(values[i].y))
            throw std::runtime_error("nonfinite GPU result");
        result[i] = {values[i].x, values[i].y};
    }
    return result;
}
} // namespace hzl
