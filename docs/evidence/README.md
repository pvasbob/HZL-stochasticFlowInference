# Initial measured evidence

These are measurements from the first implementation, not performance promises.
Machine: Intel i7-8700K (6 cores / 12 logical CPUs), approximately 32 GiB RAM,
NVIDIA RTX 3080 10 GiB. Compiler: GCC 11.4; CUDA toolkit 12.1;
driver reported by `nvidia-smi`: 580.178.04.

## CPU verification

Release milestone verification passed. The address/undefined-behavior sanitizer
build also passed outside the sandbox; inside the traced sandbox LeakSanitizer
could not initialize its exit-time check.

Taylor–Green relative errors at `dt = 0.04, 0.02, 0.01` were approximately
`2.9191e-7, 1.76454e-8, 1.0846e-9`, demonstrating the expected deterministic
fourth-order temporal convergence in that experiment. Inviscid multimode energy
and enstrophy drift over 100 steps at `dt = 0.005` were approximately
`1.90e-13` and `1.20e-13`. OU transition correlation across 2,048 independent
replicas was `0.851354`, compared with the analytical value `exp(-0.05/0.3)`.

Reproduce using `build/release/hzl-verify` after building.

## Deterministic inverse experiment

Reference viscosity and friction: `0.035`, `0.12`. With the default observation
seed and noise standard deviation `0.02`, recovered values were
`0.034043300061258573`, `0.12286731391956116`.

The training objective decreased from `4132.213550859043` to
`41.698276191098444` in 107 optimizer evaluations. Held-out objective:
`24.996649` for 40 validation observations. These are results for one controlled
two-parameter experiment, not broad identifiability or uncertainty-coverage evidence.
Use `scripts/reproduce.py` for the complete data-generation and fitting commands.

## CUDA comparison

Run `build/cuda/hzl-cuda-benchmark 256 100 0.005`.
Raw measured output is in [`gpu-256.json`](gpu-256.json).

The CPU evolution took about 8.00 seconds. GPU setup, upload, synchronized
evolution, and final download together took about 0.425 seconds. The final
relative Fourier-state difference was `1.40e-16`.

The ratio is about 18.8 for **CPU evolution / total GPU time**, with the timing
asymmetry explicitly named. The native CPU FFT is not an optimized FFTW baseline.
This is a single run with a low-mode deterministic initial condition, not a
scaling study or stochastic turbulence performance claim. GPU context startup
can dominate small problems: in a 64² / 100-step run, both CPU evolution and
total GPU time were about 0.41 seconds despite faster GPU evolution.

## Extended implementation measurements

Four adaptive chains (3,000 iterations each, first 1,000 discarded) gave classic
split R-hat about 1.006 and summed autocorrelation ESS about 1,199 for viscosity
and 1,136 for friction. These are not rank-normalized diagnostics. A 200-draw
posterior predictive experiment covered 36 of 40 held-out observations with its
nominal 95% intervals; repeated-dataset coverage remains unverified.

The EnKF reconstructed unobserved grid points with RMSE 0.1128 versus 0.1496
for the zero-forcing baseline in one controlled experiment. The four-parameter
particle calibration did **not** converge: independent log-likelihood estimates
at the fitted point had standard deviation about 8.76. The 100-step particle
chain accepted only 8% of proposals and is not posterior convergence evidence.

The rank-eight POD model captured essentially all training snapshot variance,
yet its maximum relative error in the later temporal holdout reached 2.90%.
The independent-pilot Stokes control reduced the short-horizon energy standard
error from 0.01026 to 0.00005573 with 128 production paths. This is a weakly
nonlinear case; long-horizon effectiveness and equal-cost savings remain open.

The checkpointed adjoint and tangent gradients pass comparisons at several
checkpoint strides. The measured stride-20 case used six checkpoint fields and
at most 21 replay fields, excluding solver/stage workspaces. It took 0.086 s
versus 0.080 s for the tangent calculation; two parameters are too few to expect
an adjoint speed advantage. See [storage/timing](adjoint-storage.json).

The quadratic response surface used 87 centering, 25 design, and 24 validation
full-model evaluations. Interpolation NLL RMSE was 0.85, extrapolation RMSE 22.10.
Delayed acceptance used 235 full evaluations for 1,000 proposals, accepting 18.5%.
This is an evaluation-count observation, not an ESS-per-second claim.
See [raw summary](surrogate.json).

A stochastic CUDA comparison at 128², 200 steps of 0.005, sigma 0.2 took 3.702 s
CPU evolution and 0.553 s GPU total (0.382 setup, 0.172 evolution including host
forcing and transfers, 0.000119 download). Relative state difference was
2.54e-16. This single measurement is not a scaling study. Compute Sanitizer
reported zero errors for a separate 32², ten-step stochastic run.

The 64² stochastic simulation ran to time 200 in 79.61 s. After discarding time
0–50, its 1,501 saved records gave mean energy 2.1864 and an estimated effective
sample count of 35.4. The sampled energy-budget residual was 0.00101. These
finite-record estimates do not establish stationarity; diagnostic sampling and
endpoint forcing contribute to the budget residual. See
[raw statistics](long-statistics.json) and the reproduction command in the README.

## Integrator and spatial convergence

The Taylor--Green study measured SSPRK3 orders 3.003, 3.001, and 2.993 as the
step decreased from 0.04 to 0.005. RK4 measured orders 3.980 and 4.258 before
the error reached roughly 7e-16 and roundoff dominated the final refinement.
For the smooth nonlinear spatial-operator comparison, relative drift errors at
16², 32², and 64² were 2.70e-2, 2.43e-5, and 3.94e-13 against the restricted
128² reference. Raw measurements are in
[temporal-convergence.csv](temporal-convergence.csv) and
[spatial-convergence.csv](spatial-convergence.csv). This is finite-reference
self-convergence, not an independent manufactured-solution proof.

## Multi-start stochastic calibration

The revised particle objective averages two independent likelihood estimates on
the likelihood scale and uses common random numbers across parameter evaluations.
Three 24-particle starts were run for 80 Nelder--Mead iterations on the existing
sparse stochastic training data. The best point was `(nu, alpha, sigma, tau) =`
`(0.04698, 0.03498, 0.27704, 0.27020)` versus generating values
`(0.03, 0.1, 0.2, 0.5)`. On eight untouched seeds it improved log likelihood over
the deliberately poor initial point by `29.78 ± 6.41` (paired mean ± standard
error), but the best optimizer hit its iteration limit and the three starts found
widely separated basins. Independent fitted log-likelihood SD remained 4.31.

This is evidence that replication and multi-start validation detect improvement;
it is also evidence that this short, noisy sensor dataset does not reliably
identify all four parameters. The project therefore does not claim successful
four-parameter recovery. See [stochastic calibration summary](stochastic-calibration-v2.json).

## Rank diagnostics and repeated-data coverage

For four adaptive chains after 1,000 warmup draws each, rank-normalized folded
split R-hat was 1.0056 for viscosity and 1.0060 for friction. Bulk ESS values
were 1,100 and 1,140; tail ESS values were 1,171 and 1,487. Estimated posterior
mean MCSEs were `5.16e-5` and `2.10e-4`. See
[rank diagnostics](adaptive-rank-diagnostics.json).

A 40-dataset deterministic experiment tested local observed-Hessian 95%
intervals. Thirty-eight fits converged with positive-definite Hessians. Conditional
on those fits, viscosity coverage was 38/38, friction coverage 37/38, and joint
ellipse coverage 37/38. Wilson intervals were `[0.908, 1.000]`, `[0.865, 0.995]`,
and `[0.865, 0.995]`. The two optimizer failures remain part of the result. This
supports the controlled local approximation only; it is not stochastic-model or
MCMC interval coverage. See [coverage summary](coverage-v1.json).

## Forecast, ROM, surrogate, GPU ensemble, and rare events

On a separate stochastic trajectory, the baseline 128-member EnKF forecast over
five future output times had grid RMSE 0.24346. Inflation 1.02 reduced it slightly
to 0.24239; radius-4 and radius-8 localization increased error. These settings
were inspected on the same trajectory, so this is exploratory tuning.

The fixed rank-eight POD basis had maximum relative errors 2.90% at its training
parameters, 2.75% at half viscosity, 7.17% at double viscosity, 4.51% at half
friction, and 12.39% at double friction.

The Gaussian RBF reduced interpolation NLL RMSE from the quadratic model's 0.850
to 0.397, while extrapolation RMSE worsened from 22.1 to 115.5. Corrected delayed
acceptance used 235 exact evaluations for 1,000 proposals.

At 64² and 100 steps, resident GPU ensemble totals for 1, 4, and 16 members were
0.238, 0.420, and 0.726 seconds. Six-worker CPU times were 0.435, 0.410, and
1.240 seconds. Maximum member energy disagreement was 3.20e-16. Members use
separate resident solvers with sequential launches, not batched kernels.

For path-maximum dissipation threshold 0.14 over time 2, eight independent
128-particle splitting runs estimated `0.0485 ± 0.0103`. Direct Monte Carlo
observed 41/1,024, or 0.0400 with Wilson interval `[0.0297, 0.0539]`.
Antithetic log-parameter sampling reduced final-energy estimator SE from 0.02626
to 0.00612 at equal cost, an estimated variance ratio of 18.4.

The MPI source target exists, but no MPI implementation is installed locally, so
it remains compile/runtime unverified. The extended quick CPU workflow completed.
Consolidated measurements are in [extended evidence](extended-v1.json).
