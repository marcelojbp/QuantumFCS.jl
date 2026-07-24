# QuantumFCS.jl

`QuantumFCS.jl` computes the Full Counting Statistics (FCS) of open quantum systems —
current cumulants to arbitrary order for Lindblad master equations — using a recursive
scheme built on the Drazin inverse of the Liouvillian. Currents are specified by
choosing monitored jump operators (`mJ`) and their weights (`nu`), so the same
Liouvillian can be reused for particle, photon, charge, or heat currents.

The package works with plain arrays, `QuantumOptics.jl`, and `QuantumToolbox.jl`.
It ships two Drazin backends (direct LU and a matrix-free iterative solver), plus
prepared contexts for reusing solver setup across several observables.

- ⚡ Start with the [Quickstart](@ref quickstart)
- 📘 Read up on the [Mathematical Background](@ref math)
- ⚙️ Pick a backend for large systems in [Drazin solvers](@ref solvers)
- 📝 Study the [Examples](@ref examples), including two applications from the
  companion paper: the [driven-dissipative Jaynes–Cummings model](@ref jc-example)
  and a [circuit-QED heat engine](@ref qhe-example)
- 🧭 Browse the [API](@ref api)




