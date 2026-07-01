# QuantumFCS.jl

[![CI](https://github.com/marcelojbp/QuantumFCS/actions/workflows/CI.yml/badge.svg)](https://github.com/marcelojbp/QuantumFCS/actions/workflows/CI.yml)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://marcelojbp.github.io/QuantumFCS.jl)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Cite](https://img.shields.io/badge/cite-CITATION.bib-informational.svg)](CITATION.bib)

*A Julia package for computing Full Counting Statistics (FCS) of open quantum systems.*

## Description

**QuantumFCS.jl** provides tools to study **Full Counting Statistics** (FCS) of quantum transport and quantum optics models based on Lindblad master equations. 
It implements a recursive method in which the $n+1$-th cumulant is computed using the $n$-th cumulant and the application of the Drazin inverse of the Liouvillian.

- The package follows the approach introduced in [Flindt et al., Phys. Rev. B 82, 155407 (2010)](https://arxiv.org/abs/1002.4506), focusing on Markovian dynamics.  
- It is designed for efficient numerical calculations using sparse or dense linear algebra.

## Features

- **Current cumulants to arbitrary order** for Lindblad master equations, via the
  Flindt et al. recursive scheme.
- **Multiple monitored jump channels** with signed / unitful weights (`nu`) — count
  particles ($\pm 1$), charge ($e$), heat/work (energy), etc.
- **Two Drazin-inverse backends behind one interface**: a direct sparse **LU** backend
  (`:lu`, default) and a matrix-free, preconditioned-GMRES **iterative** backend
  (`:iterative`) for large sparse Liouvillians — see the
  [Drazin solvers](https://marcelojbp.github.io/QuantumFCS.jl/dev/solvers/) guide.
- **Works with plain arrays, [`QuantumOptics.jl`](https://qojulia.org), and
  [`QuantumToolbox.jl`](https://qutip.org/QuantumToolbox.jl/)** — the relevant extension
  loads automatically. No framework is required; build your Liouvillian however you like.
- A **`LindbladFCS`** problem type to bundle a problem together with its solver options.

*(See the [API docs](https://marcelojbp.github.io/QuantumFCS.jl) for the full list.)*

## Installation
To install the package, in the Julia REPL, 
```julia
using Pkg
Pkg.add("QuantumFCS")
```

## Quickstart example

A self-contained example: a single quantum dot coupled to a hot and a cold reservoir,
built with `QuantumOptics.jl`. We monitor electrons entering the cold reservoir and
compute the first two cumulants of the current.

```julia
using QuantumOptics
using QuantumFCS

b = FockBasis(1)                 # single fermionic mode
d = destroy(b); d_dag = create(b)

ϵd = 1.0                         # dot energy level
κc = 0.1; κh = 0.5              # cold / hot coupling strengths

H = ϵd * d_dag * d              # Hamiltonian
Jcloss = sqrt(κc) * d           # jump into the cold reservoir
Jhgain = sqrt(κh) * d_dag       # jump from the hot reservoir
J = [Jcloss, Jhgain]

ρss = steadystate.iterative(H, J)

mJ = [Jcloss]                   # monitored jump(s)
nu = [1]                        # weight(s): +1 per electron into the cold bath

c1, c2 = fcscumulants_recursive(H, J, mJ, 2, ρss, nu)
```

You can also pass a vectorised Liouvillian `L` directly —
`fcscumulants_recursive(L, mJ, nC, ρss, nu)` — with no quantum framework loaded. See the
[Quickstart](https://marcelojbp.github.io/QuantumFCS.jl/dev/quickstart/) and
[Examples](https://marcelojbp.github.io/QuantumFCS.jl/dev/examples/) for more.

## Documentation

- ⚡ [Quickstart](https://marcelojbp.github.io/QuantumFCS.jl/dev/quickstart/) — install and first calculation
- 📘 [Mathematical Background](https://marcelojbp.github.io/QuantumFCS.jl/dev/math/) — the recursive scheme and Drazin inverse
- ⚙️ [Drazin solvers](https://marcelojbp.github.io/QuantumFCS.jl/dev/solvers/) — choosing the `:lu` vs `:iterative` backend
- 📝 [Examples](https://marcelojbp.github.io/QuantumFCS.jl/dev/examples/) — worked models
- 🧭 [API](https://marcelojbp.github.io/QuantumFCS.jl/dev/api/) — full reference

## Roadmap

Planned extensions:

- Time-dependent systems
- Non-Markovian dynamics
- Computing the FCS distribution numerically
- Factorial moments

## Citation

If you use QuantumFCS.jl in your research, please cite it via
[`CITATION.bib`](CITATION.bib) / [`CITATION.yaml`](CITATION.yaml). A companion paper with
applications is in preparation — its reference will be added here once available.
