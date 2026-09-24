# Numerical conventions and current assumptions

## Flow equations and Fourier representation

The domain is `[0, 2π)²`. The state is vorticity, satisfying

`∂t ω = -u·∇ω + ν Δω - αω + f`, with `Δψ = -ω`,
`u = ∂yψ`, and `v = -∂xψ`.

Fourier coefficients are normalized by `1/N²` on the forward transform. The
inverse is an unnormalized Fourier sum. Thus `ψ̂ = ω̂ / |k|²`,
`û = i ky ψ̂`, and `v̂ = -i kx ψ̂`. The zero vorticity mode is constrained to zero;
mean velocity is also zero in this formulation.

Nonlinear advection is evaluated in physical space. Modes are retained only when
`3|kx| < N` and `3|ky| < N`; strict truncation avoids ambiguous cutoff modes.
This is the two-thirds truncation rule for quadratic products. Both the state
and the nonlinear RHS are projected to this space.

Integrated diagnostics use the full domain area `A = 4π²`:

- `E = A/2 Σ |ω̂k|² / |k|²`;
- `Z = A/2 Σ |ω̂k|²`;
- viscous energy dissipation `ε = 2νZ`;
- forcing energy injection `I = A Σ Re(conj(ω̂k) f̂k) / |k|²`.

The energy balance is `dE/dt = I - 2νZ - 2αE`. The spectrum bins by the nearest
integer `|k|`; summing bins gives integrated kinetic energy.

## Time stepping

The inference, tangent, adjoint, reduced-order, and CUDA implementations use
classical explicit RK4. CPU forward simulation also offers the three-stage,
third-order Shu--Osher SSPRK3 method. SSPRK3 needs three drift evaluations per
step instead of RK4's four and preserves strong-stability properties under an
appropriate forward-Euler bound; it has lower asymptotic order. Neither method
is asserted to be optimal for stiff viscous flow. The current stability guard is
deliberately conservative for both.

The convergence command checks both methods against exact Taylor--Green decay.
It also samples a smooth non-bandlimited vorticity field at several grids and
compares its instantaneous dealiased drift with modes restricted from a finer
reference. This tests the complete spatial operator and demonstrates spectral
self-convergence. Because the reference is numerical rather than an independent
forced manufactured solution, this does not prove correctness by itself.
The implementation rejects steps when either:

- `dt max_grid (|u| + |v|)/dx > 0.5`, or
- `dt max_retained (ν|k|² + α) > 2.5`.

These conservative checks do not replace convergence studies. The CPU backend
checks finite states after each step. GPU results are checked on download.
Integrating-factor or exponential methods remain candidates for larger or stiffer
problems, after measured comparisons.

## OU forcing

Only modes in the specified radial shell are forced. One complex coefficient is
sampled per conjugate pair; its partner is the complex conjugate. The convention is
`dFk = -Fk/τ dt + σ (dB1 + i dB2)/√2`. Consequently each real component has
stationary variance `σ²τ/4`, and `E|Fk|² = σ²τ/2`.

Exact OU transitions over `h` use decay `exp(-h/τ)` and real-component innovation
standard deviation `σ sqrt(τ(1-exp(-2h/τ))/4)`. `expm1` avoids cancellation for
small `h/τ`. Initialization samples the stationary forcing distribution.

For each flow step, forcing advances by half a step, remains at that midpoint
value through the four RK stages, then advances by another half step. The OU
transition is exact; the coupled stochastic PDE discretization is not. The
deterministic RK4 order must not be advertised as the stochastic convergence order.
Coupled stochastic refinement studies with shared paths remain to be implemented.

`σ` is a per-mode amplitude. Changing the shell changes total forcing power.
Stationary forcing does not imply the initial flow has reached stationarity.

## Reproducibility

Each ensemble member gets a seed from its logical member index via a SplitMix64
mixing function. Workers own separate solver and RNG instances. Aggregation uses
member order. This provides scheduling-independent results in the same build;
hash-derived streams are not a mathematical proof of independence.

The RNG uses `mt19937_64` and an explicitly implemented Box–Muller normal
transform. Checkpoints serialize the engine, cached normal draw, Fourier state,
forcing, parameters, and time at round-trip precision. Cross-platform bitwise
identity is not promised because floating-point transcendental libraries differ.

## Initial deterministic inverse problem

The initial state is known and contains several wave numbers. Sensors measure
vorticity with known, independent Gaussian standard deviations. The negative
log likelihood is `0.5 Σ ((prediction-observation)/sd)²`; parameter-independent
normalizing constants are omitted.

Optimization operates in `q = (log ν, log α)` with numerical bounds `[-12, 0]`.
The stochastic forcing amplitude is zero. This provides a controlled first
inverse problem and must not be presented as four-parameter stochastic recovery.
Nelder–Mead convergence alone is not evidence of parameter identifiability.

Metropolis sampling uses independent Gaussian priors on the log parameters,
centered at `log(0.03)` and `log(0.1)` with standard deviation 1.5, truncated to
the same numerical domain. Since the prior and proposal are specified in `q`,
no additional log-transform Jacobian belongs in this target. A prior specified
on physical parameters would require its appropriate transformation.

For a perturbation `s`, the tangent operator is
`DF(ω)s = -u(s)·∇ω - u(ω)·∇s + νΔs - αs`.
Log-viscosity sensitivity adds `νΔω`; log-friction sensitivity adds `-αω`.
The coupled state/tangent equations differentiate every RK4 stage. The PDE
likelihood gradient is compared with finite differences across several step
sizes before BFGS or HMC uses it. This implementation is deterministic;
stochastic, automatic, and adjoint differentiation remain future work.

## Particle likelihood and state estimation

The latent Markov state contains both Fourier vorticity and OU forcing.
Each bootstrap particle evolves from its own stationary forcing draw and the
known initial vorticity. Gaussian observation weights include their normalizing
constants and are normalized using log-sum-exp. Systematic resampling occurs
below `N/2` effective particles. A copied ancestor retains its current forcing
but receives an independent future innovation stream.

The product of sequential normalizing-factor estimates is the particle likelihood
estimate for the discretized model. Pseudo-marginal Metropolis proposes new
parameters and an independent likelihood seed; rejection retains the previous
parameters **and likelihood estimate**. This follows the construction described
by [Andrieu and Roberts](https://arxiv.org/abs/0903.5480). It does not remove
time-discretization bias or poor mixing from high likelihood variance.

The four-parameter prior is Gaussian in log coordinates, standard deviation 1.5,
with centers `log(0.03), log(0.1), log(0.2), log(0.5)`. Bounds are `[-10,0]`
for log viscosity/friction and `[-10,2]` for log amplitude/correlation time.
Particle-MAP optimization holds randomness fixed for repeatable evaluations,
then checks the fitted parameters with independent seeds. The improved objective
computes several independent unbiased likelihood estimates and takes the log of
their arithmetic mean. Averaging must occur before the logarithm: averaging log
likelihoods would target a different quantity. Fixed common random numbers make
parameter comparisons repeatable, while multiple starts expose local basins.
Untouched seeds provide paired initial-versus-fitted checks. Conditional slices
vary one log parameter while holding the others fixed; they are sensitivity
diagnostics rather than nuisance-optimized profile likelihoods. This remains a
noisy approximation, not an exact optimizer for the marginal likelihood.

The serial EnKF uses independent perturbed observations and sample cross
covariances to update both vorticity and forcing. Scalar observations are
assimilated sequentially, appropriate for the current diagonal observation
covariance. Fourier projection and conjugate symmetry are preserved. EnKF
approximates the filtering distribution; its output is not an unbiased likelihood.
Multiplicative anomaly inflation and periodic physical-space Gaspari--Cohn
localization are configurable. Localization applies to vorticity and forcing
cross-covariances before each serial update. These controls require out-of-sample
tuning. Forecasts freely propagate the final analyzed ensemble and include future
forcing innovations. Smoothing is not implemented.

## Bayesian computation and prediction

Adaptive Metropolis estimates proposal covariance during warmup with online
second moments and a small positive diagonal regularizer. It uses a
`2.38² / dimension` covariance scaling and freezes the proposal after warmup.
This is a finite-adaptation variant of covariance-based adaptive proposals;
see [Haario, Saksman, and Tamminen](https://researchportal.helsinki.fi/en/publications/an-adaptive-metropolis-algorithm/).
Discard adaptation samples when reporting posterior summaries.

Multi-chain reporting splits every retained chain, rank-normalizes pooled draws,
and reports the larger of rank and folded-rank split R-hat. Bulk ESS uses a
multi-chain variogram and an initial positive, monotone paired autocorrelation
sequence. Tail ESS is the smaller ESS of indicators below the pooled 5% quantile
and above the 95% quantile. These remain diagnostics rather than proofs.

The repeated-data coverage command fits independent synthetic datasets, forms a
symmetrized observed Hessian by finite differencing exact tangent gradients, and
inverts the two-by-two Hessian in log-parameter coordinates. It tests marginal
normal intervals and a chi-square joint ellipse. Wilson intervals quantify
binomial uncertainty. Failed optimizations and non-positive Hessians are counted
and excluded explicitly from the conditional coverage denominator. This validates
a local Laplace/Wald approximation, not MCMC or stochastic-model intervals.

HMC uses identity mass, Gaussian momenta, a reversible leapfrog proposal, and
a Metropolis energy correction. Invalid-domain trajectories are rejected.
Energy errors above magnitude 1,000 are flagged as divergences. The fixed step
size and trajectory length are configurable; NUTS and automatic tuning are not
implemented. A short accepted chain is not evidence of convergence.

The deterministic posterior-predictive command propagates selected post-warmup
parameter draws through the full solver. It reports state uncertainty separately
from intervals that also include measurement noise. Finite empirical coverage
in one held-out dataset is not a repeated-experiment coverage study.

## POD and Galerkin reduction

Snapshots are centered and use the real Fourier inner product
`Re Σ conj(a_k)b_k`, equivalent to domain-averaged physical vorticity L2.
Consequently the retained variance is vorticity/enstrophy-related variance,
not kinetic-energy variance. The snapshot Gram matrix is diagonalized with a
cyclic symmetric Jacobi solver, then dominant modes are reorthogonalized.
The Gram approach squares singular-value conditioning; tiny eigenvalues are
truncated and the implementation limits training to 512 snapshots. A production
large-scale SVD backend remains planned.

Galerkin projection produces `da/dt = c + A a + Q(a,a)`. Coefficients are
precomputed from the full deterministic drift and its analytic linearization.
Runtime evolution is RK4 in reduced coordinates, without full-grid FFTs.
Validation separates projection error from autonomous trajectory error on a
later time window. High captured training variance alone does not ensure
out-of-training accuracy, stability, or parameter-region validity. Closure and
stochastic reduced factors are not yet implemented.

## Linear Stokes control variate

The control drops nonlinear advection but shares the full model's midpoint
forcing samples. For a mode with `λ = ν|k|²+α` and `z = λdt`, RK4 gives
`w_{n+1} = a w_n + b F_{n+1/2}`, where
`a = 1-z+z²/2-z³/6+z⁴/24` and `b = dt(1-z/2+z²/6-z³/24)`.

Let `S = σ²τ/2` and `ρ = exp(-dt/τ)`. Starting at zero vorticity, the exact
moments of this discrete control satisfy
`V_next = a²V + b²S + 2abρC` and `C_next = aρC + bS`,
where `C` is covariance with the most recent midpoint forcing. Summing
`A V/(2|k|²)` gives its known mean energy. This matches the implemented discrete
control rather than borrowing a continuum expectation with discretization bias.

A separate pilot estimates `β = Cov(Q,C)/Var(C)`. Production samples use
`Q - β(C-E[C])`, with standard errors calculated from independent production
realizations. Timings include pilot and production separately. A variance ratio
is not automatically an equal-cost speedup. The short measured experiment is
weakly nonlinear; variance reduction at long turbulent horizons remains unverified.

## GPU benchmark scope

The GPU state remains resident during evolution. Each step transfers one CFL
scalar to the host. The benchmark reports setup/context/plans/upload, synchronized
evolution including those scalar transfers, host OU generation and forcing uploads
when enabled, and final download separately. CPU and GPU use identical forcing
samples from the shared `ShellForcing` implementation.
CPU initialization is excluded from the CPU evolution time, while the reported
GPU total includes its setup. The explicit ratio names reflect that asymmetry.
The baseline CPU FFT is a reference implementation, not an optimized FFTW baseline.
No stationary turbulence, stochastic inference, or universal speedup claim follows
from these finite-time benchmarks. GPU inference integration remains planned.


## Checkpointed discrete adjoint

The reverse pass differentiates all four RK4 stages and injects observation
cotangents at their discrete observation steps. Its spatial transpose uses the
real Fourier inner product and the same projection as the forward solver.
Parameter contributions are accumulated in log viscosity/friction coordinates.
Every configurable number of steps, a state checkpoint is saved; reverse blocks
replay forward states and reconstruct stage values. Reported checkpoint and replay
field counts exclude solver and RK4 workspaces. Verification compares multiple
checkpoint strides against tangent derivatives and central finite differences.
This implementation currently differentiates deterministic observations only.

## Response surface and delayed acceptance

A quadratic model fits scaled log parameters with overdetermined Householder QR.
Training, interpolation, and extrapolation errors are reported independently.
For a symmetric proposal from x to y, stage one accepts with probability
`min(1, exp(s(y)-s(x)))`; stage two uses
`min(1, exp(l(y)-l(x)-s(y)+s(x)))`, where l is the full log target and s the
approximation. This retains the full target under the usual MCMC conditions;
it does not guarantee efficient mixing. See
[Christen and Fox (2005)](https://www.tandfonline.com/doi/abs/10.1198/106186005X76983).
The model is frozen during sampling; centering, design, and validation evaluations
are counted separately from sampling calls. A local quadratic can extrapolate
poorly, and a low full-evaluation count alone is not an equal-cost ESS improvement.

The Gaussian RBF alternative solves a ridge-regularized kernel system with partial
pivoting in normalized coordinates. It can improve local interpolation while
extrapolating much worse. The exact delayed-acceptance correction means surrogate
error changes efficiency rather than the invariant target.

## Fixed-level splitting and antithetic parameters

Path-maximum dissipation levels partition zero to the requested threshold. Each
trajectory advances until first hitting the next level or the common final time.
Survivors are cloned from their hitting states and receive independent future
innovations. The estimate is the product of conditional survival fractions.
Independent splitting runs provide an empirical standard error. Finite-population
cloning bias is not corrected, so direct Monte Carlo remains part of validation.

For parameter uncertainty, antithetic pairs evaluate Gaussian log-parameter
perturbations `z` and `-z`. Pair-average variance determines the antithetic
standard error; ordinary sampling uses the same number of deterministic full
solves.

## Finite-record statistics

The optional time-series report removes a specified initial interval and requires
regularly sampled diagnostics. The PSD is a mean-subtracted rectangular-window
one-sided periodogram normalized so frequency-bin integration recovers the sample
population variance. FFT autocovariance uses a fixed record-length denominator;
initial positive paired autocorrelation sums estimate statistical inefficiency.
These estimates assume approximate stationarity and may be unreliable for short
records. Complete nonoverlapping block means are exported for inspection.
The energy budget compares endpoint energy change with trapezoidal integration
of `injection - dissipation - 2 alpha energy`. Saved endpoint forcing and coarse
output cadence add quadrature error beyond the solver's time-discretization error.
