# Changelog

All notable changes to `QuantumFCS.jl` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-07-24

The first release since 1.0.0, and a large one. Version 1.0.0 exported a *single*
function, `fcscumulants_recursive`, which took no options at all: one direct
sparse-LU path, one supported quantum framework. This release turns that into a
configurable API — a second solver backend for large sparse problems, problem and
context types, trace-constrained steady states, and a second framework backend —
while leaving the original call path numerically untouched.

|                        | 1.0.0 | 1.1.0 |
|------------------------|-------|-------|
| Exported symbols       | 1     | 14    |
| Package extensions     | 1     | 3     |
| Drazin backends        | 1     | 2     |
| Keyword options        | 0     | 8     |
| `src/` + `ext/` lines  | 430   | 1691  |
| Test files             | 3     | 10    |

**Everything here is additive.** Code written against 1.0.0 keeps working: the
positional call `fcscumulants_recursive(L, mJ, nC, rho_ss, nu)` on 1.1.0 was
checked against the registered 1.0.0 source and returns **bit-identical** results
through the fourth cumulant, as does the explicit `method = :lu`.

### Added

#### Iterative Drazin backend for large sparse Liouvillians

`method = :iterative` solves each Drazin system with **incomplete-LU-preconditioned
GMRES** instead of a direct factorization, so memory stays bounded where sparse-LU
fill-in would not fit. It never assembles the gauge-fixed operator, reuses one
preconditioner and one Krylov workspace across all cumulant orders, and keeps the
iteration count roughly flat as the Hilbert space grows. This makes counting
statistics of Liouvillians with vectorised dimension beyond `10^6` tractable on a
workstation.

It lives in the `QuantumFCSIterativeExt` extension, enabled by
`using Krylov, IncompleteLU`, so it costs nothing when unused. `:lu` remains the
default and the right choice unless LU has been *measured* to be the bottleneck.
See the new [Drazin solvers](https://marcelojbp.github.io/QuantumFCS.jl/dev/solvers/)
guide.

#### Preconditioner reuse (`Pl`)

Building the ILU dominates the cost of an iterative point, and it is usually
redundant — the steady state is normally obtained from a preconditioned solve of
the *same* Liouvillian. The new `Pl` keyword on `fcscumulants_recursive`,
`LindbladFCS` and `prepare_fcs_context` accepts an externally built preconditioner
instead of building one internally, so a parameter point can cost a single
factorization. An injected `Pl` is applied on the **right**, so GMRES still stops
on the true residual: a poor preconditioner costs iterations, never correctness.

#### Trace-constrained steady states

`trace_constrained_system` and `trace_constrained_steadystate` replace the singular
condition `L ρ = 0` with a non-singular system by imposing `tr ρ = 1`, solve it
(`:lu` or `:iterative`), and **return the preconditioner they built** alongside
`rho_ss`. That result feeds straight into the FCS solve, which is what makes the
one-factorization-per-point pattern above practical. Supporting types
`TraceConstrainedSystem` / `TraceConstrainedSteadyState` and the helper
`shifted_ilu_preconditioner` are exported. Passing a prebuilt system also enables
warm-started continuation (`Pl`, `u0`) across a parameter sweep.

#### Problem types and prepared contexts

- `LindbladFCS` (and its supertype `FCSProblem`) bundles a model, a current and
  solver options into one object, solved with `fcscumulants_recursive(problem)`.
  Fields accept plain arrays or backend operators interchangeably.
- `prepare_fcs_context` / `PreparedLindbladFCS` prepare the Drazin solver **once**
  and evaluate any number of observables against it. The solver depends only on
  the Liouvillian and steady state — never on the monitored jumps or weights — so
  hot and cold heat currents, or several output channels, share one factorization.
  The result is numerically identical to preparing each separately; only the cost
  changes.
- `prepare_drazin_solver`, `drazin_solve` and the `DrazinSolver` supertype expose
  that machinery directly for callers who want it.

#### QuantumToolbox.jl support

A second framework backend, `QuantumFCSQuantumToolboxExt`, alongside the existing
QuantumOptics one. `QuantumObject`s can be passed anywhere operators are accepted.
Both backends also gained a two-positional convenience constructor
`LindbladFCS(H, J; mJ, rho_ss, nu, nC)`.

#### Factorial cumulants

`factorial_cumulants` converts ordinary cumulants via the signed Stirling numbers
of the first kind, also reachable as `cumulant_type = "factorial"` on any
`fcscumulants_recursive` call.

#### Options on `fcscumulants_recursive`

1.0.0 accepted none. 1.1.0 accepts `method`, `σ`, `τ`, `Pl`, `rtol`, `itmax`,
`memory` and `cumulant_type`, with defaults that reproduce 1.0.0 behaviour.

#### Documentation

A [Drazin solvers](https://marcelojbp.github.io/QuantumFCS.jl/dev/solvers/) guide
covering backend choice, tuning and the reuse patterns; a mathematical-background
page aligned with the companion manuscript's notation; and two worked applications
from that manuscript — the
[driven-dissipative Jaynes–Cummings model](https://marcelojbp.github.io/QuantumFCS.jl/dev/examples/jaynes_cummings/)
(photon-blockade breakdown, three cumulants on Liouvillians up to `~10^6 × 10^6`,
including how the cumulants rather than the boundary tail gauge truncation) and a
[circuit-QED heat engine](https://marcelojbp.github.io/QuantumFCS.jl/dev/examples/circuit_qed_heat_engine/)
(two heat currents on one prepared context, ending in a thermodynamic-uncertainty
analysis).

### Changed

- Minimum Julia version is now **1.10**; CI covers 1.10, 1.11 and 1.12. (1.0.0
  declared `1.9, 1.10, 1.11`, so support for 1.9 is dropped and 1.12 is now
  explicitly tested.)
- `QuantumToolbox` compat is `0.28 - 0.47`. The package reads only `.data` off
  backend operators and never constructs a `QuantumObject`, so it is unaffected by
  the constructor changes across that range; verified on both endpoints. Code that
  *builds* `QuantumObject`s should follow its own QuantumToolbox version.
- Added the previously missing `QuantumOptics` compat bound (`1.2`).
- `TraceConstrainedSystem.dimensions` is documented as reserved and is `nothing`
  unless supplied through the new `dimensions` keyword of
  `trace_constrained_system`. No backend populates it automatically: a
  `QuantumToolbox` super-operator reports Liouville-space dimensions, not the
  Hilbert-space ones a caller needs to rewrap `rho_ss`.
- Corrected docstrings, most consequentially `mJ`, which was described as being in
  "vectorized representation". Monitored jumps are ordinary `n×n` operators.

### Removed

- `Test` is no longer a hard dependency. It was declared in `[deps]` but used
  nowhere in `src/` or `ext/`; it remains a test-only dependency.

## [1.0.0] - 2025-09-12

Initial registered release: the recursive zero-frequency cumulant scheme
(`fcscumulants_recursive`), a cached sparse-LU Drazin solve, and `QuantumOptics.jl`
integration.

[1.1.0]: https://github.com/marcelojbp/QuantumFCS.jl/releases/tag/v1.1.0
[1.0.0]: https://github.com/marcelojbp/QuantumFCS.jl/releases/tag/v1.0.0
