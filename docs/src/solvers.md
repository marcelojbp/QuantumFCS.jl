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
| `τ` | ILU drop tolerance | Smaller ⇒ denser preconditioner (fewer GMRES iterations, more memory); larger ⇒ sparser/cheaper but may need more iterations. Ignored when `Pl` is supplied. |
| `σ` | Diagonal shift for the preconditioner only | `nothing` auto-scales from ``\mathcal{L}``. Increase if the ILU is unstable. Does **not** change the solution, only the preconditioner. Ignored when `Pl` is supplied. |
| `Pl` | Externally built preconditioner to reuse | Skips the internal ILU build; see [Reusing a preconditioner](@ref reuse-precond) below. |
| `rtol` | GMRES relative tolerance | Loosen for speed, tighten for accuracy. |
| `itmax` | GMRES iteration cap | Raise if you see a non-convergence warning. |
| `memory` | GMRES restart (Krylov basis) size | Larger can improve convergence at higher memory cost. |

### [Reusing a preconditioner](@id reuse-precond)

Building the ILU is the dominant cost of the iterative backend, and it is often
redundant: the steady state `ρss` is usually obtained by a preconditioned linear
solve of the *same* Liouvillian, so a suitable ILU already exists by the time the
cumulants are computed. Pass it as `Pl` to skip the internal build:

```julia
using Krylov, IncompleteLU

# `Pss` is an ILU of the (shifted) Liouvillian, e.g. the one built for the
# steady-state solve at this or a neighbouring parameter point.
cumulants = fcscumulants_recursive(L, mJ, nC, ρss, nu;
                                   method = :iterative, Pl = Pss)
```

`Pl` must support `LinearAlgebra.ldiv!(y, Pl, x)` and `ldiv!(Pl, x)` — an
`IncompleteLU.ILUFactorization` does — and need only approximate ``\mathcal{L}^{-1}``
up to a diagonal shift, the rank-1 gauge term, and mild parameter drift. Because an
injected preconditioner may have been built for a *neighbouring* operator, it is
applied on the **right**, so GMRES converges on the true residual rather than a
preconditioned surrogate: a genuinely poor `Pl` shows up as extra iterations or a
non-convergence warning, never as a silently wrong result. When `Pl` is supplied,
`σ` and `τ` are ignored.

This is the recommended pattern for parameter sweeps — build (or reuse) one ILU per
neighbourhood of points and feed it to both the steady-state solve and the FCS
cumulants.

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

## [Preparing the steady state for iterative FCS](@id steady-state-prep)

The [Reusing a preconditioner](@ref reuse-precond) pattern above assumes you already
have a suitable ILU in hand. [`trace_constrained_steadystate`](@ref) is the
package-level helper that produces one: it solves for the steady state *and* returns
the preconditioner it built, so the two feed straight into the FCS cumulants without
a second ILU build.

The Liouvillian ``\mathcal{L}`` is singular (its kernel is the steady state), so the
steady state is found from the **trace-constrained** system ``A\,\vec\rho = b``, where
the redundant first equation is replaced by the normalization ``\mathrm{tr}\,\rho = 1``.
`trace_constrained_steadystate` builds that system, builds a shifted-ILU
preconditioner, solves it with GMRES, and returns a lean
[`TraceConstrainedSteadyState`](@ref):

```julia
using Krylov, IncompleteLU   # for method = :iterative
using QuantumFCS

ss = trace_constrained_steadystate(L; method = :iterative)   # solves for ρss and Pl

ctx = prepare_fcs_context(ss; method = :iterative)           # reuses ss.Pl — no rebuild
hot  = fcscumulants_recursive(ctx; mJ = mJ_hot,  nu = nu_hot,  nC = 2)
cold = fcscumulants_recursive(ctx; mJ = mJ_cold, nu = nu_cold, nC = 2)
```

`ss.rho_ss` is a hermitianized, trace-normalized sparse matrix; `ss.Pl` is the
steady-state preconditioner; and `ss.stats` carries scalar diagnostics (convergence,
iterations, residuals, trace/hermiticity errors, ILU/GMRES timings). The steady-state
`Pl` was built for the trace-constrained matrix rather than ``\mathcal{L}`` itself, so
the FCS backend applies it on the **right** (as with any injected `Pl`), keeping the
GMRES stopping test on the true residual.

`method = :lu` gives a direct-solve baseline (`A \ b`, `Pl = nothing`) for small and
medium systems; `method = :iterative` requires the `QuantumFCSIterativeExt` extension.
Both accept a prebuilt Liouvillian `L`, a Hamiltonian/jumps pair `H, J`, or a
prebuilt [`TraceConstrainedSystem`](@ref) — the last enables warm-started continuation
across a parameter sweep by reusing one system and preconditioner (`Pl`, `u0`) over
neighbouring points.

## [Reusing a prepared solver across observables](@id prepared-context)

The Drazin solver depends only on the Liouvillian ``\mathcal{L}`` and the steady
state ``\rho_\text{ss}`` — never on the monitored jumps or their weights, which
enter only the counting-field super-operators and the recursion's right-hand sides.
So when several observables share the *same* ``\mathcal{L}`` and ``\rho_\text{ss}``
— hot and cold heat currents, left and right particle currents, several output
channels of a multi-terminal device, or different weight choices for one jump set —
the expensive solver preparation (the sparse LU factorization, or the shifted-ILU
build) need only be done **once**.

A plain [`LindbladFCS`](@ref) problem prepares its solver internally for a single
observable, so evaluating ``k`` observables that way repeats that preparation ``k``
times. [`prepare_fcs_context`](@ref) instead builds the invariant data once, into a
[`PreparedLindbladFCS`](@ref), and reuses it for every observable:

```julia
ctx = prepare_fcs_context(; L = L, rho_ss = ρss, method = :lu)   # prepares the solver once

hot  = fcscumulants_recursive(ctx; mJ = mJ_hot,  nu = nu_hot,  nC = 2)
cold = fcscumulants_recursive(ctx; mJ = mJ_cold, nu = nu_cold, nC = 2)
```

The context accepts the same inputs as `LindbladFCS`: pass a prebuilt Liouvillian
`L`, or a Hamiltonian `H` and jump operators `J`; `L`/`H`/`J`/`rho_ss` may be plain
arrays or backend operators (`QuantumOptics`, `QuantumToolbox`). It is
**backend-generic** — the same `method`, `σ`, `τ`, `Pl`, `rtol`, `itmax`, and
`memory` keywords select and tune the `:lu` or `:iterative` solve, and configure the
one-time preparation:

```julia
using Krylov, IncompleteLU   # for method = :iterative

ctx = prepare_fcs_context(; H = H, J = J, rho_ss = ρss, method = :iterative, Pl = Pss)
```

The result is numerically identical to preparing the solver anew for each observable
(the reuse changes only the cost, never the answer). This is the recommended setup
whenever many monitored observables are computed on one Liouvillian and steady state.

## API

```@docs
QuantumFCS.prepare_drazin_solver
QuantumFCS.drazin_solve
QuantumFCS.DrazinSolver
```

The context object and its constructor for the workflow above,
[`prepare_fcs_context`](@ref) and [`PreparedLindbladFCS`](@ref), are documented with
the other problem types on the [API](@ref api) page.
