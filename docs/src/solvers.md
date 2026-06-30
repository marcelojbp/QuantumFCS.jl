# [Drazin solvers](@id solvers)

Computing the cumulants of the counting statistics requires applying the
(projected) **Drazin inverse** ``\mathcal{L}^{+}`` of the Liouvillian to a sequence
of right-hand sides — one per cumulant order (see the
[Mathematical Background](@ref math) for where ``\mathcal{L}^{+}`` enters the
recursion). `QuantumFCS` never forms ``\mathcal{L}^{+}`` explicitly; instead it
prepares a solver once and *applies* it to each right-hand side.

Two backends share a single interface and are selected with the `method` keyword of
[`fcscumulants_recursive`](@ref) (or the `method` field of a
[`LindbladFCS`](@ref) problem):

| Backend | `method` | What it does | Best for |
|---|---|---|---|
| Direct LU | `:lu` (default) | Caches one sparse LU factorization of ``\mathcal{L}`` and reuses it for every order | Small to medium systems; dense Liouvillians |
| Iterative | `:iterative` | Matrix-free preconditioned GMRES (no factorization) | Large, sparse Liouvillians |

## The LU backend (`:lu`)

The default. It factorizes the Liouvillian once with a sparse LU and reuses that
factorization across all cumulant orders. Each order then costs only a triangular
solve, so it is very fast and accurate whenever the factorization fits in memory.

This is the right choice for the vast majority of problems. **Use it unless you
have measured LU factorization (or its memory) to be the bottleneck.** A dense
Liouvillian always uses this path.

!!! note "Why LU eventually fails to scale"
    The Liouvillian acts on a vectorized density matrix, so for a Hilbert space of
    dimension ``d`` it is a ``d^2 \times d^2`` matrix. Even when ``\mathcal{L}`` is
    sparse, its LU factors suffer **fill-in** — they become far denser than
    ``\mathcal{L}`` itself — and both runtime and memory grow steeply with ``d``.
    Past a few hundred Hilbert dimensions this becomes the limiting factor.

## The iterative backend (`:iterative`)

For large sparse Liouvillians where LU fill-in is prohibitive, the iterative
backend solves each system with **preconditioned GMRES** and never factorizes
``\mathcal{L}``. It combines three ingredients:

1. **Matrix-free gauge fixing.** ``\mathcal{L}`` is singular (its kernel is the
   steady state). We solve with the operator ``A x = \mathcal{L} x + \rho\,(v_\mathbb{1}\cdot x)``,
   a rank-1 update that lifts the null mode so ``A`` is nonsingular. `A` is applied
   as an operator only — it is never assembled, so no fill-in is created.
2. **Shifted incomplete-LU preconditioner.** An incomplete LU of ``\mathcal{L} - \sigma I``
   is built once and reused for every order. The shift ``\sigma`` keeps the
   incomplete factorization well-behaved so the GMRES iteration count is governed
   by the preconditioner rather than the (possibly tiny) physical Liouvillian gap.
3. **Reused Krylov workspace.** The GMRES basis is allocated once and reused for
   every cumulant order.

The result keeps a roughly constant iteration count and bounded memory as ``d``
grows, so its advantage over LU widens with system size.

### Enabling it

The iterative backend lives in a package extension and loads only when its
dependencies are present:

```julia
using Krylov, IncompleteLU   # activates QuantumFCSIterativeExt
using QuantumFCS

cumulants = fcscumulants_recursive(L, mJ, nC, ρss, nu; method = :iterative)
```

Without `Krylov` and `IncompleteLU` loaded, `method = :iterative` raises an
informative error. Both are [weak dependencies](https://pkgdocs.julialang.org/v1/creating-packages/#Weak-dependencies),
so they add no load-time cost when you stick to the default backend.

### Tuning and caveats

The iterative backend exposes a few knobs (keywords of
[`fcscumulants_recursive`](@ref) / fields of [`LindbladFCS`](@ref)). The defaults
are sensible; reach for these only when convergence or performance needs help:

| Option | Meaning | Guidance |
|---|---|---|
| `τ` | ILU drop tolerance | Smaller ⇒ denser preconditioner (fewer GMRES iterations, more memory); larger ⇒ sparser/cheaper but may need more iterations. |
| `σ` | Diagonal shift for the preconditioner only | `nothing` auto-scales from ``\mathcal{L}``. Increase if the ILU is unstable. Does **not** change the solution, only the preconditioner. |
| `rtol` | GMRES relative tolerance | Loosen for speed, tighten for accuracy. |
| `itmax` | GMRES iteration cap | Raise if you see a non-convergence warning. |
| `memory` | GMRES restart (Krylov basis) size | Larger can improve convergence at higher memory cost. |

!!! tip "Steady-state accuracy dominates the high cumulants"
    Both backends inherit the accuracy of the steady state `ρss` you pass in. High
    cumulants are sensitive to it: an approximate steady state limits agreement
    between `:lu` and `:iterative` at the same order. For tight comparisons use an
    exact steady state (e.g. `steadystate.eigenvector`).

!!! warning "Match the backend to the problem"
    `:iterative` carries preconditioner-setup and per-solve iteration overhead that
    direct LU does not. On small or dense systems it is **slower** than `:lu`. Only
    switch when LU fill-in is the measured bottleneck.

See [Krylov.jl](https://jso.dev/Krylov.jl/stable/) and
[IncompleteLU.jl](https://github.com/haampie/IncompleteLU.jl) for details on the
underlying GMRES and ILU implementations.

## API

```@docs
QuantumFCS.prepare_drazin_solver
QuantumFCS.drazin_solve
QuantumFCS.DrazinSolver
```
