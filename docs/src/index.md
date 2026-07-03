# QuantumFCS.jl

`QuantumFCS.jl` computes the Full Counting Statistics (FCS) of open quantum systems —
current cumulants to arbitrary order for Lindblad master equations — using a recursive
scheme built on the Drazin inverse of the Liouvillian. It works with plain arrays,
`QuantumOptics.jl`, and `QuantumToolbox.jl`, and ships two Drazin backends (direct LU and
a matrix-free iterative solver) for scaling to large systems.

- ⚡ Start with the [Quickstart](@ref quickstart)
- 📘 Read up on the [Mathematical Background](@ref math)
- ⚙️ Pick a backend for large systems in [Drazin solvers](@ref solvers)
- 📝 Study the [Examples](@ref examples)
- 🧭 Browse the [API](@ref api)





