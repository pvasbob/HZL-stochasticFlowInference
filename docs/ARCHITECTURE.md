# Architecture

The primary numerical implementation is C++20. CUDA code is compiled as C++17
with CUDA 12.1 for compatibility with the installed toolkit. Python launchers use the standard library; optional statistical analysis, plotting,
and archival scripts use NumPy, Matplotlib, and h5py.

| Component | Responsibility |
| --- | --- |
| `include/hzl/fft.hpp`, `src/fft.cpp` | Fourier normalization, wave-number mapping, reusable radix-2 plans/workspace |
| `include/hzl/solver.hpp`, `src/solver.cpp` | Model parameters, stochastic streams, spectral dynamics, diagnostics, restart |
| `include/hzl/inference.hpp`, `src/inference.cpp` | Observations, optimization, likelihood, sampling, online moments, CPU ensembles |
| `src/adjoint.cpp` | Checkpointed reverse differentiation of the discrete likelihood |
| `include/hzl/surrogate.hpp`, `src/surrogate.cpp` | QR response surfaces and corrected delayed acceptance |
| `include/hzl/filter.hpp`, `src/filter.cpp` | Particle likelihood, pseudo-marginal sampling, joint-state EnKF |
| `include/hzl/reduced.hpp`, `src/reduced.cpp` | Snapshot POD and projected quadratic dynamics |
| `include/hzl/monte_carlo.hpp`, `src/monte_carlo.cpp` | Coupled Stokes control and independent-pilot variance reduction |
| `src/main.cpp` | Validated CLI options and experiment files |
| `include/hzl/cuda_solver.hpp`, `src/cuda_solver.cu` | GPU-resident evolution with shared host-generated OU forcing with cuFFT and CUDA kernels |
| `src/cuda_benchmark.cpp` | CPU–GPU numerical comparison and timing boundaries |
| `src/cuda_ensemble_benchmark.cpp` | Resident multi-instance GPU ensemble throughput and CPU agreement |
| `src/mpi_ensemble.cpp` | Optional rank-partitioned ensemble with logical member seeds |
| `tests/verification.cpp` | Milestone numerical/statistical verification |
| `scripts/` | JSON launch configuration, reproduction, and posterior reporting |

The CPU solver owns all scratch arrays and reuses them through integration.
Its FFT object is not shared across threads. Ensemble workers own independent
solvers; results are collected in logical member order. Parameters and diagnostics
are value types. CUDA resources are owned through a private RAII implementation.

The current full-complex layout favors transparent Fourier conventions and
verification. Real-to-complex layouts, FFTW, and reusable/batched inference
workspaces are future measured optimizations. Avoid general backend abstractions
until shared requirements are established by the CPU and GPU implementations.

The inference module can be separated further into observation, optimization,
and sampling modules as those capabilities grow. The current implementation keeps
the dependency graph small and the complete inverse workflow runnable.

All reported features must be linked to runnable commands and evidence. Output
directories are never silently reused. Checkpoints and datasets have explicit
schema versions. Unknown CLI options and invalid physical parameters fail with
an error instead of being silently accepted.

The EnKF inflates joint spectral state/forcing anomalies at each observation
time. Localization transforms cross-covariance fields to physical space, applies
a periodic Gaspari--Cohn taper centered on the sensor, then transforms them back.
Forecast records continue the analyzed ensemble with independent future OU
innovations.
