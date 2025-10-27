# QuantumFCS.jl

[![CI](https://github.com/marcelojbp/QuantumFCS/actions/workflows/CI.yml/badge.svg)](https://github.com/marcelojbp/QuantumFCS/actions/workflows/CI.yml)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://marcelojbp.github.io/QuantumFCS)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Cite](https://img.shields.io/badge/cite-CITATION.bib-informational.svg)](CITATION.bib)

*A Julia package for computing Full Counting Statistics (FCS) of open quantum systems.*

## Description

**QuantumFCS.jl** provides tools to study **Full Counting Statistics** (FCS) of quantum transport and quantum optics models based on Lindblad master equations. 
It implements a recursive method in which the $n+1$-th cumulant is computed using the $n$-th cumulant and the application of the Drazin inverse of the Liouvillian.

- The package follows the approach introduced in [Flindt et al., Phys. Rev. B 82, 155407 (2010)](https://arxiv.org/abs/1002.4506), focusing on Markovian dynamics.  
- It is designed for efficient numerical calculations using sparse or dense linear algebra.

## Features

- Compute current cumulants to arbitrary order
- Support for **sparse** and **dense** matrix backends  
- Methods for integration with `QuantumOptics.jl` operators, states.
- Does not need `QuantumOptics.jl`; you can build your Liouvillian however you like and use `QuantumFCS.jl`.


### Main function
- `fcscumulants_recursive(L, mJ, nC, rho_ss, nu)` – compute FCS cumulants up to order `nC`, taking as input 
a (vectorised) Liouvillian, `L`, the steady-state `rho_ss`, a vector of monitored jumps `mJ`, and a vector `nu` to attribute sign 
and unitful weighs to the monitored jumps.
- `fcscumulants_recursive(H,J, mJ, nC, rho_ss, nu)` – Method for quantum optics---you can pass, instead of the Liouvillian,
the Hamiltonian and the vector of all jump operators.

*(See the [API docs](https://marcelojbp.github.io/QuantumFCS) for the full list.)*

## Installation
To install the package, in the Julia REPL, 
```julia
] add https://github.com/marcelojbp/QuantumFCS
```

## Quickstart example

This example requires you to provide the vectorised Liouvillian, its steady-state and specify other inputs as described below.

```julia
using QuantumFCS

# Build your Liouvillian L and monitored jumps mJ
# L     : Complex sparse/dense matrix   (vectorized Liouvillian)
# mJ    : Vector of sparse jump super-operators you want to monitor
# nC    : Number of cumulats to be computed
# rho_ss : Steady-state density matrix (matrix, not vectorized)
# nu     : Vector of weighs (same length as mJ) for the monitored jumps

mJ = [sqrt(kappa) * a, sqrt(kappa) * a_dagger] #Monitoring loss and injection of photons
nu = [-1, 1] #We count -1 if we anihilate and +1 if we create
# compute first 3 cumulants
c1, c2, c3 = fcscumulants_recursive(L, mJ, 3, rho_ss, nu)

```

## Extensions
For the future,

- Non-Markovian dynamics
- Time-dependent systems
- Computing the FCS distribution numerically
- Integration with other frameworks, such as `QuantumToolbox.jl`
