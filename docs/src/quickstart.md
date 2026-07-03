# [Quickstart](@id quickstart)

This package provides tools to compute full-counting-statistics cumulants from a Liouvillian.

## Installation
To install the package, in the Julia REPL, 
```julia
using Pkg 
Pkg.add("QuantumFCS")
```

## Quickstart example

We model a quantum dot heat engine as a single mode coupled to two reservoirs (hot and cold) in the large bias limit ($n_c = 0,~ n_h = 1$).

```julia
using QuantumOptics 
using QuantumFCS
# Basis for the single quantum dot
b = FockBasis(1)
# Operators 
d = destroy(b)
d_dag = create(b)
# Parameters
ϵd = 1.0;                            # Energy level of the quantum dot
κc = 0.1;                            # Coupling strength to cold reservoir
κh = 0.5;                            # Coupling strength to hot reservoir
# Hamiltonian and jump operators
H = ϵd * d_dag * d;                  
Jcloss = sqrt(κc) * d;               # Jumps into the cold reservoir 
Jhgain = sqrt(κh) * d_dag;           # Jumps from the hot reservoir
J = [Jcloss, Jhgain];
# Steady state
ρss = steadystate.iterative(H, J)
# Weight vector 
nu = [1];
# Monitored jump operator (particles entering the cold reservoir)
mJ = [Jcloss];
# Calculating the first two cumulants
c1, c2 = fcscumulants_recursive(H, J, mJ, 2, ρss, nu);
println("\nFull Counting Statistics:")
println("First cumulant : $c1")
println("Second cumulant : $c2") 
```

## Without a quantum framework

You do not need `QuantumOptics.jl` (or `QuantumToolbox.jl`). If you already have a
vectorised Liouvillian `L`, its steady state `ρss`, and your monitored jump operators
`mJ`, call the core method directly:

```julia
using QuantumFCS

# L    : ComplexF64 sparse/dense matrix — the vectorized Liouvillian
# mJ   : Vector of monitored jump operators (matrices, not super-operators)
# ρss  : steady-state density matrix
# nu   : weights, one per entry of mJ
c1, c2, c3 = fcscumulants_recursive(L, mJ, 3, ρss, nu)
```

See the [Examples](@ref examples) for a full manual Liouvillian construction.

## Next steps

- For large sparse Liouvillians, switch to the matrix-free iterative backend with
  `fcscumulants_recursive(...; method = :iterative)` — see [Drazin solvers](@ref solvers).
- To bundle a problem with its solver options, use the [`LindbladFCS`](@ref) problem type
  and call `fcscumulants_recursive(problem)`.

