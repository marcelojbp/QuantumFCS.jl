# Changelog

All notable changes to `QuantumFCS.jl` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-07-24

A large, entirely additive release: everything below is new API or new
documentation. Existing 1.0 code continues to work unchanged.

### Added

- **Iterative Drazin backend** (`method = :iterative`) — matrix-free,
  incomplete-LU-preconditioned GMRES for large sparse Liouvillians where direct-LU
  fill-in is prohibitive. Ships as the `QuantumFCSIterativeExt` extension, enabled
  by `using Krylov, IncompleteLU`. See the [Drazin solvers](https://marcelojbp.github.io/QuantumFCS.jl/dev/solvers/) guide.
- **`QuantumToolbox.jl` support** via the `QuantumFCSQuantumToolboxExt` extension,
  alongside the existing `QuantumOptics.jl` backend.
- **Prepared FCS contexts** — `prepare_fcs_context` and `PreparedLindbladFCS`
  prepare the Drazin solver once and evaluate any number of observables that share
  a Liouvillian and steady state, so hot/cold currents (or several output channels)
  cost one factorization in total.
- **Trace-constrained steady states** — `trace_constrained_system`,
  `trace_constrained_steadystate`, `TraceConstrainedSystem`,
  `TraceConstrainedSteadyState`, and `shifted_ilu_preconditioner`. The steady-state
  solve returns the preconditioner it built, which can be handed straight to the
  FCS cumulants (`Pl`) or to `prepare_fcs_context`.
- **Preconditioner injection** — the `Pl` keyword on `fcscumulants_recursive`,
  `LindbladFCS`, and `prepare_fcs_context` reuses an externally built
  preconditioner instead of building an ILU internally. Applied on the right, so a
  poor `Pl` costs iterations rather than correctness.
- **Reusable Drazin solvers** — `prepare_drazin_solver`, `drazin_solve`, and the
  `DrazinSolver` abstract type.
- **`factorial_cumulants`** — convert ordinary cumulants to factorial cumulants via
  the signed Stirling numbers of the first kind, also reachable through the
  `cumulant_type = "factorial"` keyword.
- **Documentation**: two worked applications from the companion manuscript — the
  [driven-dissipative Jaynes–Cummings model](https://marcelojbp.github.io/QuantumFCS.jl/dev/examples/jaynes_cummings/)
  and a [circuit-QED heat engine](https://marcelojbp.github.io/QuantumFCS.jl/dev/examples/circuit_qed_heat_engine/) —
  plus a mathematical-background page aligned with the manuscript's notation.

### Changed

- Minimum Julia version is now **1.10** (was 1.9). CI tests 1.10, 1.11 and 1.12.
- `QuantumToolbox` compat widened to `0.28 - 0.47`. The package reads only `.data`
  off backend operators and never constructs a `QuantumObject`, so it is unaffected
  by the constructor changes across that range; user code that builds
  `QuantumObject`s should follow its own QuantumToolbox version.
- Added the previously missing `QuantumOptics` compat bound (`1.2`).
- `TraceConstrainedSystem.dimensions` is documented as reserved and is `nothing`
  unless supplied via the new `dimensions` keyword of `trace_constrained_system`.
  No backend populates it automatically: a `QuantumToolbox` super-operator reports
  Liouville-space dimensions rather than the Hilbert-space ones a caller needs.

### Removed

- `Test` is no longer a hard dependency. It was listed in `[deps]` but used nowhere
  in `src/` or `ext/`; it remains a test-only dependency.

## [1.0.0] - 2025-09-12

Initial registered release: the recursive zero-frequency cumulant scheme
(`fcscumulants_recursive`), the `LindbladFCS` problem type, the direct sparse-LU
Drazin backend, and `QuantumOptics.jl` integration.

[1.1.0]: https://github.com/marcelojbp/QuantumFCS.jl/releases/tag/v1.1.0
[1.0.0]: https://github.com/marcelojbp/QuantumFCS.jl/releases/tag/v1.0.0
