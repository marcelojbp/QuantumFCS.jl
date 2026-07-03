# Repository Guidelines

## Project Structure & Module Organization

`QuantumFCS.jl` is a Julia package for full counting statistics of open quantum systems. The main module lives in `src/QuantumFCS.jl` and includes core implementations from `src/FCS_functions.jl`. Optional `QuantumOptics.jl` integration is isolated in `ext/QuantumFCSQuantumOpticsExt.jl` and `ext/FCS_QuantumOptics_functions.jl`, so core functionality should not require that weak dependency. Tests live in `test/`, with `test/runtests.jl` including focused files such as `drazin_inverse.jl` and `qd_heat_engine.jl`. Documentation sources are in `docs/src/`; generated HTML is under `docs/build/`. `scripts/demo.jl` is available for local experimentation and profiling.

## Build, Test, and Development Commands

- `julia --project=. -e 'using Pkg; Pkg.instantiate()'`: install package dependencies for the active Julia environment.
- `julia --project=. -e 'using Pkg; Pkg.test()'`: run the full test suite, including `QuantumOptics` extension tests.
- `julia --project=docs docs/make.jl`: build and doctest the Documenter.jl documentation.
- `julia --project=. scripts/demo.jl`: run the demo problem used for quick local checks.

CI currently tests Julia `1.9`, `1.10`, and `1.11` on Ubuntu, so keep changes compatible with those versions.

## Coding Style & Naming Conventions

Follow standard Julia style with 4-space indentation, explicit type annotations where they clarify dispatch, and descriptive lowercase function names with underscores, for example `fcscumulants_recursive` and `drazin_apply`. Prefer `LinearAlgebra` and `SparseArrays` APIs over ad hoc implementations. Keep exported API additions in `src/QuantumFCS.jl`; keep weak-dependency methods in `ext/`. No formatter configuration is committed, so preserve nearby style and avoid broad formatting-only diffs.

## Testing Guidelines

Use Julia's `Test` standard library. Add new tests as small files in `test/` and include them from `test/runtests.jl` inside the top-level `@testset "QuantumFCS.jl"`. Name files after the behavior or model under test, such as `drazin_comparison.jl`. For numerical checks, prefer `isapprox` or equivalent approximate comparisons with explicit tolerances, following existing `1e-10` style where appropriate.

## Commit & Pull Request Guidelines

Recent commits use short, plain-language summaries such as `Fix installation instructions...` and `Add 'Factorial moments'...`; keep messages concise and action-oriented. Pull requests should describe the motivation, list user-visible API or docs changes, and mention test commands run locally. Link related issues when available. Include screenshots only for documentation or visual asset changes.

## Documentation Notes

Update `docs/src/` when public behavior, examples, or signatures change. Do not hand-edit `docs/build/` unless the project intentionally tracks regenerated documentation for the change.
