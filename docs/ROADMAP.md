# Implementation status and remaining scope

The complete target remains the `description` specification. This table tracks
working code rather than treating planned interfaces as completed capabilities.
Correctness, mathematical understanding, and measured evidence govern promotion
of a capability to validated status.

| Area | Current implementation | Remaining work |
| --- | --- | --- |
| Build/platform | C++20, CMake presets, CPU and CUDA builds, optional Python dependency lock | Execute the added hosted CI workflow, numerical-library dependency integration, static analysis |
| PDE solver | Pseudo-spectral vorticity solver, strict dealiasing, RK4 and SSPRK3, CFL checks, exact temporal-order and smooth spatial self-convergence studies | Independent manufactured solutions, semi-implicit/ETD comparison, broader flow-regime convergence |
| Stochastic model | Stationary shell OU forcing, exact OU transitions, deterministic streams | Coupled stochastic convergence, correlated modes, additional physically justified forcing models |
| Diagnostics | Energy, enstrophy, dissipation, injection, spectrum, fields, time-series PSD/ACF and sampled budget | Structure functions, longer stationarity and balance studies |
| Data | Versioned sensor CSV, checkpoints, metadata, JSON launcher, typed HDF5 archive export | Native streaming HDF5, richer schemas, irregular timestamps, missing observations, correlated noise |
| Calibration | Deterministic recovery with internal Nelder–Mead/BFGS; multi-start CRN particle MAP with likelihood-scale replication, untouched-seed checks, and conditional identifiability slices | Reliable four-parameter recovery under an informative experiment design, nuisance-optimized profiles, additional optimizers |
| Derivatives | Analytic tangent and checkpointed discrete adjoint through RK4; full PDE gradient comparisons | AD, stochastic sensitivities, broader memory/scaling tradeoffs |
| Bayesian inference | Random-walk/adaptive Metropolis, HMC, pseudo-marginal particle MCMC, rank-normalized folded split R-hat, bulk/tail ESS, deterministic posterior prediction, repeated-data Laplace/Wald coverage | Longer HMC/particle studies, MCMC credible-interval coverage, joint-state stochastic forecasts |
| Hidden states | Bootstrap particle filter; serial joint flow/forcing EnKF with inflation, periodic Gaspari--Cohn localization, reconstruction and free forecasts | Linear/extended Kalman filters, smoothing, out-of-sample localization tuning, broader forecast coverage |
| CPU parallelism | Scheduling-independent parallel flow ensembles | Parallel objectives/chains, reusable worker pools, cancellation, scaling studies |
| CUDA | GPU-resident deterministic/stochastic solver, shared host OU paths, resident multi-instance ensemble benchmark and CPU agreement | Device-side forcing, likelihood/filter integration, truly batched kernels, broader precision/scaling studies |
| Reduced models | Snapshot POD, projection/reconstruction, quadratic Galerkin dynamics, temporal holdout and viscosity/friction parameter-jump sweep | Large-scale SVD backend, closure, stochastic factors, broader parameter-region training |
| Monte Carlo | Ensembles, online moments, standard errors, deterministic posterior propagation | Coverage studies, stochastic posterior propagation, broader convergence reports |
| Variance reduction | Shared-forcing Stokes control with exact mean/independent pilot; equal-cost antithetic log-parameter propagation | Stratification, Latin hypercube, randomized QMC, MLMC across grid/time levels |
| Surrogates | Quadratic QR and regularized Gaussian RBF models, shared interpolation/extrapolation validation, corrected delayed acceptance | GP uncertainty, adaptive parameter-region design, ESS-per-cost studies |
| Rare events | Direct final/path-maximum estimates, Wilson intervals, validated fixed-level trajectory splitting with replicate SE | Importance sampling, adaptive level placement, splitting bias studies, more extreme tails |
| Stress testing | Five parameter/forcing scenarios with common random streams | Observation failures, model mismatch, stationary scenarios and longer horizons |
| MPI | Optional logical-member distributed ensemble target with rank-independent seeds; uncompiled because MPI is absent locally | Install/runtime validation, restart, strong/weak multi-rank and multi-node scaling |
| Evidence | Numerical/statistical milestones, recovery, gradients, posterior coverage, forecast/ROM/surrogate/control/rare-event studies, GPU timings, quick extended workflow | Full extended workflow archive, broader scientific validation, external CI/MPI execution |

Implementation order follows the 15 agreed milestones, with small useful pieces
of later stages brought forward when they exercise already working components.
No fixed-path stochastic fit will be described as inference over unknown forcing.
Single-machine MPI checks will not be described as multi-node scaling evidence.
