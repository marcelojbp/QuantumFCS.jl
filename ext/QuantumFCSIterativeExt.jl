module QuantumFCSIterativeExt

# Iterative Drazin backend for QuantumFCS.
#
# Loaded automatically when both `Krylov` and `IncompleteLU` are present. It adds
# a `method = :iterative` backend to `prepare_drazin_solver` that scales to large
# sparse Liouvillians where direct sparse-LU fill-in is prohibitive.
#
# Method (general, for any singular sparse Liouvillian L with right null vector ρ
# and left null vector vId):
#   * Gauge-fixed operator, matrix-free:  A·x = L·x + ρ·(vId·x).  A is nonsingular
#     and never assembled — only its action is needed, so no dense fill is created.
#   * Preconditioner:  P = ilu(L − σI; τ).  The diagonal shift σ makes the operator
#     comfortably nonsingular so the incomplete factorization stays well-behaved
#     and the Krylov iteration count is governed by the preconditioner, not by the
#     (possibly tiny) physical Liouvillian gap. Built once, reused across orders.
#   * Solve:  project the RHS onto range(L), run preconditioned GMRES, re-impose the
#     trace-zero gauge, sparsify — same pre/post-processing as `drazin_apply`.

using QuantumFCS
using QuantumFCS: DrazinSolver, drazin_solve,
    _drazin_project, _drazin_gauge!, _drazin_sparsify
using LinearAlgebra
using SparseArrays
using SparseArrays: rowvals, nonzeros, nnz
using Krylov
using IncompleteLU

"""
    GaugeOp(L, ρ, vId)

Matrix-free gauge-fixed Liouvillian `A·x = L·x + ρ·(vId·x)`.

`L` is singular (it has the steady state `ρ` as a right null vector and the trace
functional `vId` as a left null vector). Adding the rank-1 term `ρ·(vId·x)` lifts
that null mode, making `A` nonsingular while never assembling a dense matrix — only
its action via `mul!` is needed, so no fill-in is created. Used as the operator
passed to GMRES.
"""
struct GaugeOp{TL, Tρ, TV}
    L::TL
    ρ::Tρ
    vId::TV
end

Base.size(G::GaugeOp) = size(G.L)
Base.size(G::GaugeOp, d::Integer) = size(G.L, d)
Base.eltype(::GaugeOp) = ComplexF64

function LinearAlgebra.mul!(y::AbstractVector, G::GaugeOp, x::AbstractVector)
    mul!(y, G.L, x)                    # y = L·x
    s = dot(G.vId, x)                  # vId·x  (conjugates vId; vId is real here)
    ρ = G.ρ
    @inbounds for k in 1:nnz(ρ)
        y[rowvals(ρ)[k]] += s * nonzeros(ρ)[k]
    end
    return y
end

Base.:*(G::GaugeOp, x::AbstractVector) = mul!(similar(x, ComplexF64, size(G, 1)), G, x)

"""
    IterativeDrazinSolver <: DrazinSolver

Prepared iterative Drazin solver: a matrix-free [`GaugeOp`](@ref), a reusable
preconditioner, and a single preallocated GMRES workspace.

The recursion applies the Drazin inverse to `nC-1` right-hand sides with the *same*
operator and preconditioner, so the Krylov basis is allocated once (`ws`) and
reused via the in-place `gmres!` rather than reallocated on every solve. Build it
with `prepare_drazin_solver(L, ρ, vId; method=:iterative, ...)` and apply it with
[`drazin_solve`](@ref).

The preconditioner `P` is either built internally as a shifted ILU of `L − σI`
(`side = :left`, the tested default) or supplied by the caller through the `Pl`
keyword (`side = :right`). An injected preconditioner is applied on the right so
GMRES converges on the *true* residual rather than a preconditioned surrogate —
this keeps the stopping criterion honest when `Pl` only approximates `L` up to a
shift, a low-rank gauge term, or mild parameter drift (see `prepare_drazin_solver`).
"""
struct IterativeDrazinSolver{TA, TP, Tρ, TV, TW} <: DrazinSolver
    A::TA              # matrix-free GaugeOp
    P::TP              # preconditioner: internal ILU of (L − σI), or an injected Pl
    ρ::Tρ
    vId::TV
    ws::TW             # preallocated Krylov.GmresWorkspace, reused across RHS
    rtol::Float64      # Krylov relative tolerance
    atol::Float64      # Krylov absolute tolerance
    itmax::Int
    memory::Int        # GMRES basis size (baked into `ws` at allocation)
    sparsify_rtol::Float64
    side::Symbol       # preconditioning side: :left (internal ILU) or :right (injected Pl)
end

"""
    _prepare_iterative_drazin_solver(L, ρ, vId; σ=nothing, τ=0.05, Pl=nothing,
                                     rtol=1e-8, atol=1e-12, itmax=200, memory=30,
                                     sparsify_rtol=1e-12)

Build an [`IterativeDrazinSolver`](@ref) for the sparse Liouvillian `L`. Backs the
`method=:iterative` branch of `prepare_drazin_solver`; more specific than the core
catch-all, so it takes over once this extension loads.

Keyword arguments:
* `Pl`  — an externally built preconditioner to reuse instead of building an ILU
  here. When supplied, `σ` and `τ` are ignored and the preconditioner is applied
  on the *right* (see [`IterativeDrazinSolver`](@ref)). `Pl` must support
  `LinearAlgebra.ldiv!(y, Pl, x)` and `ldiv!(Pl, x)` and approximate `L⁻¹` up to a
  shift, a low-rank gauge term, and mild parameter drift — e.g. an
  `IncompleteLU.ILUFactorization` of a nearby Liouvillian. `nothing` (default)
  builds the internal shifted ILU.
* `σ`   — diagonal shift used only to build the internal ILU preconditioner of
  `L − σI`. `nothing` auto-scales it to `0.01·maximum(abs, nonzeros(L))`. Ignored
  when `Pl` is supplied.
* `τ`   — internal ILU drop tolerance: smaller keeps more fill (stronger
  preconditioner, more memory), larger is sparser/cheaper but may need more GMRES
  iterations. Ignored when `Pl` is supplied.
* `rtol`/`atol`/`itmax` — GMRES convergence tolerances and iteration cap.
* `memory` — GMRES Krylov basis size (restart length), fixed at allocation.
* `sparsify_rtol` — relative threshold for sparsifying each solution.
"""
function QuantumFCS._prepare_iterative_drazin_solver(
        L::SparseMatrixCSC{ComplexF64, Int},
        ρ::SparseVector{ComplexF64, Int},
        vId::AbstractVector{ComplexF64};
        σ = nothing,
        τ::Float64 = 0.05,
        Pl = nothing,
        rtol::Float64 = 1.0e-8,
        atol::Float64 = 1.0e-12,
        itmax::Int = 200,
        memory::Int = 30,
        sparsify_rtol::Float64 = 1.0e-12
    )

    if Pl === nothing
        # Auto-scale the shift from the operator magnitude when not supplied. The shift
        # only needs to lift the near-zero mode enough for a stable incomplete LU; a
        # small fraction of the largest entry generalizes across systems.
        σeff = σ === nothing ? 0.01 * maximum(abs, nonzeros(L)) : Float64(σ)

        Ls = L - σeff * I              # sparse, same sparsity pattern as L
        P = IncompleteLU.ilu(Ls; τ = τ)
        side = :left                   # internal ILU closely matches L: left-precondition
    else
        # Reuse the caller's preconditioner. It was built for a nearby operator
        # (different shift / gauge term / parameter point), so precondition on the
        # right to keep the GMRES stopping test on the true residual.
        P = Pl
        side = :right
    end
    A = GaugeOp(L, ρ, vId)

    # Allocate the GMRES basis once and reuse it for every cumulant-order RHS.
    # `memory` is a workspace-construction parameter in Krylov (the Krylov basis
    # size), not a per-solve `gmres!` keyword, so it is fixed here.
    n = size(L, 1)
    ws = Krylov.GmresWorkspace(n, n, Vector{ComplexF64}; memory = memory)

    return IterativeDrazinSolver(
        A, P, ρ, vId, ws, rtol, atol, itmax, memory,
        sparsify_rtol, side
    )
end

function QuantumFCS.drazin_solve(s::IterativeDrazinSolver, α::AbstractVector)
    # Project RHS onto range(L): α' = α - ρ (vId·α).
    αp = _drazin_project(α, s.ρ, s.vId)

    # Preconditioned GMRES on the matrix-free gauge-fixed operator, reusing the
    # preallocated workspace `s.ws`. `gmres!` zeros the initial guess on entry and
    # leaves `warm_start = false`, so each RHS is solved cleanly with no leftover
    # state — only the Krylov basis is shared, not the solution.
    #
    # Preconditioning side (fixed at preparation): an internal ILU closely matches
    # `L` and is applied on the left (`M`); an injected `Pl` may only approximate
    # `L` and is applied on the right (`N`), so GMRES stops on the true residual.
    if s.side === :right
        Krylov.gmres!(
            s.ws, s.A, αp;
            N = s.P, ldiv = true,
            rtol = s.rtol, atol = s.atol,
            itmax = s.itmax
        )
    else
        Krylov.gmres!(
            s.ws, s.A, αp;
            M = s.P, ldiv = true,
            rtol = s.rtol, atol = s.atol,
            itmax = s.itmax
        )
    end

    stats = Krylov.statistics(s.ws)
    stats.solved || @warn "Iterative Drazin solve did not converge" niter = stats.niter rtol = s.rtol

    # Re-impose trace-zero gauge on the solution, then sparsify into a fresh
    # SparseVector (the next solve overwrites `s.ws.x`, so we copy out here).
    y = Krylov.solution(s.ws)
    _drazin_gauge!(y, s.ρ, s.vId)
    return _drazin_sparsify(y; rtol = s.sparsify_rtol)
end

# ============================================================================
#  Trace-constrained steady state (iterative backend)
# ============================================================================
#
# Provides the iterative half of the package-level steady-state API. The numerics
# reproduce the application helpers: the shifted-ILU preconditioner of the
# trace-constrained matrix, and the GMRES call (left-preconditioned, `ldiv=true`,
# memory 60, rtol 1e-10, atol 1e-14). The resulting preconditioner is returned so it
# can be reused by the FCS Drazin backend (its main purpose).

"""
    shifted_ilu_preconditioner(A::SparseMatrixCSC{ComplexF64,Int};
                               τ = 1e-3, shift_factor = 1e-6, shift = nothing)

Incomplete-LU preconditioner of the shifted trace-constrained matrix `A + shift·I`.
See the core [`shifted_ilu_preconditioner`](@ref) for the shift convention.
"""
function QuantumFCS.shifted_ilu_preconditioner(
        A::SparseMatrixCSC{ComplexF64, Int};
        τ::Float64 = 1.0e-3, shift_factor::Float64 = 1.0e-6, shift = nothing
    )
    σ = shift === nothing ?
        shift_factor * max(real(norm(A, 1) / size(A, 1)), eps(Float64)) :
        Float64(shift)
    return IncompleteLU.ilu(A + σ * I; τ = τ)
end

function QuantumFCS._trace_constrained_steadystate_iterative(
        sys::QuantumFCS.TraceConstrainedSystem;
        Pl = nothing, u0 = nothing,
        τ::Float64 = 1.0e-3, shift_factor::Float64 = 1.0e-6, shift = nothing,
        rtol::Float64 = 1.0e-10, atol::Float64 = 1.0e-14,
        itmax::Int = 200, memory::Int = 60
    )
    A = sys.A
    b = sys.b

    # Build the preconditioner unless the caller supplies one to reuse.
    ilu_seconds = 0.0
    if Pl === nothing
        t0 = time_ns()
        Pl = QuantumFCS.shifted_ilu_preconditioner(
            A;
            τ = τ, shift_factor = shift_factor, shift = shift
        )
        ilu_seconds = (time_ns() - t0) / 1.0e9
    end

    # Left-preconditioned GMRES, optionally warm-started from `u0` (continuation).
    t1 = time_ns()
    x, stats = u0 === nothing ?
        Krylov.gmres(
            A, b; M = Pl, ldiv = true, memory = memory,
            rtol = rtol, atol = atol, itmax = itmax
        ) :
        Krylov.gmres(
            A, b, u0; M = Pl, ldiv = true, memory = memory,
            rtol = rtol, atol = atol, itmax = itmax
        )
    gmres_seconds = (time_ns() - t1) / 1.0e9

    stats.solved || @warn "Trace-constrained steady-state GMRES did not converge" niter = stats.niter rtol = rtol

    residual = norm(A * x - b)
    relative_residual = residual / norm(b)
    return QuantumFCS._finalize_steadystate(
        sys.L, x; Pl = Pl,
        converged = stats.solved, iterations = stats.niter,
        residual = residual, relative_residual = relative_residual,
        ilu_seconds = ilu_seconds, gmres_seconds = gmres_seconds
    )
end

end # module
