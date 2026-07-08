# ============================================================================
#  Trace-constrained steady state — the upstream half of the iterative workflow
# ============================================================================
#
# The iterative FCS backend's main advantage is *reusing* the preconditioner built
# for the steady-state solve as `Pl` (see `prepare_fcs_context` / the `:iterative`
# Drazin backend). This file provides that upstream step as package-level API:
# build the trace-constrained linear system `A ρ = b`, build a stable shifted-ILU
# preconditioner, solve for the steady state, and return both `rho_ss` and the
# reusable `Pl`.
#
# The numerics reproduce the application helpers (`jc_*` / `qhe_*` in the FCS paper
# code) exactly: the constraint weight, the trace row, the ILU shift, and the GMRES
# call all match, so results are identical to the current hand-rolled workflow.

"""
    TraceConstrainedSystem

The trace-constrained steady-state linear system `A ρ⃗ = b` for a Liouvillian `L`.

`L` is singular (the physical steady state is its null vector), so the redundant
first equation is replaced by a trace constraint: `A = L + |row 1⟩⟨diag|·w` and
`b = w·e₁`, where the constraint hits the vectorized-identity (diagonal) indices so
that solving imposes `tr(ρ) = 1`.

This is a low-level object for constructing, testing, and warm-started continuation
solves. Prefer the lean [`TraceConstrainedSteadyState`](@ref) as the user-facing
result. Build one with [`trace_constrained_system`](@ref).

# Fields
* `L`: the sparse `ComplexF64` Liouvillian (unmodified; this is what FCS uses).
* `A`: the trace-constrained system matrix `L + trace_constraint`.
* `b`: the right-hand side (`w` in the first entry, zeros elsewhere).
* `vId`: the vectorized identity / trace functional (diagonal indices).
* `dimensions`: backend Hilbert-space dimensions when available, else `nothing`.
"""
struct TraceConstrainedSystem{TL, TA, Tb, TV, TD}
    L::TL
    A::TA
    b::Tb
    vId::TV
    dimensions::TD
end

"""
    TraceConstrainedSteadyState

The lean, user-facing result of [`trace_constrained_steadystate`](@ref): a steady
state together with the preconditioner built to find it, ready to feed into the
iterative FCS machinery via [`prepare_fcs_context`](@ref).

It intentionally does **not** retain the large trace-constrained matrix `A` or the
right-hand side `b` — the target problems are memory critical.

# Fields
* `L`: the sparse `ComplexF64` Liouvillian used for FCS.
* `rho_ss`: the steady state as a sparse `ComplexF64` matrix, hermitianized and
  trace-normalized.
* `Pl`: the preconditioner built for the steady-state solve (`nothing` for the
  direct `:lu` method). Reused by the `:iterative` FCS backend — this reuse is the
  whole point of the API.
* `stats`: a `NamedTuple` of scalar diagnostics only (convergence, iterations,
  residuals, trace/hermiticity errors, ILU/GMRES timings) — no large arrays.
"""
struct TraceConstrainedSteadyState{TL, Tρ, TP, TS}
    L::TL
    rho_ss::Tρ
    Pl::TP
    stats::TS
end

# Backend hook: Hilbert-space dimensions for reconstructing a backend object later.
# Core knows none; extensions (QuantumToolbox) override for their operator types.
_backend_dimensions(::Any) = nothing

"""
    trace_constrained_system(L; weight = nothing) -> TraceConstrainedSystem
    trace_constrained_system(H, J; weight = nothing) -> TraceConstrainedSystem

Build the trace-constrained steady-state linear system for a Liouvillian.

Supply a (vectorized) Liouvillian `L`, or a Hamiltonian `H` and jump operators `J`
(built into `L` by the active backend). Inputs may be plain dense/sparse
`ComplexF64` arrays or backend operators (`QuantumOptics`, `QuantumToolbox`); they
are normalized to `SparseMatrixCSC{ComplexF64, Int}`.

The default constraint `weight` is `norm(L, 1) / length(L)` (the entrywise 1-norm of
`L` divided by its total number of entries), matching the application helpers; pass
`weight` to override it.
"""
function trace_constrained_system(L; weight = nothing)
    Ldata = SparseMatrixCSC{ComplexF64, Int}(_operator_data(L))

    l = size(Ldata, 1)
    size(Ldata, 2) == l || throw(
        DimensionMismatch("Liouvillian must be square; got $(size(Ldata))."))
    n = isqrt(l)
    n * n == l || throw(
        DimensionMismatch("Liouvillian side $l is not a perfect square (n²)."))

    w = weight === nothing ? norm(Ldata, 1) / length(Ldata) : Float64(weight)
    wc = ComplexF64(w)

    # RHS: only the (redundant) first equation carries the constraint value.
    b = zeros(ComplexF64, l)
    b[1] = wc

    # Trace constraint in row 1 at the vectorized-identity (diagonal) columns
    # 1:(n+1):l — solving A ρ⃗ = b then imposes w·tr(ρ) = w, i.e. tr(ρ) = 1.
    diag_idx = collect(1:(n + 1):l)
    trace_constraint = sparse(ones(Int, n), diag_idx, fill(wc, n), l, l)
    A = Ldata + trace_constraint

    vId = SparseVector{ComplexF64, Int}(l, diag_idx, fill(1.0 + 0.0im, n))

    return TraceConstrainedSystem(Ldata, A, b, vId, _backend_dimensions(L))
end

function trace_constrained_system(H, J; weight = nothing)
    L = _build_liouvillian(H, J)
    return trace_constrained_system(L; weight = weight)
end

# --- Preconditioner: generic stub; the iterative extension provides the real one --
"""
    shifted_ilu_preconditioner(A; τ = 1e-3, shift_factor = 1e-6, shift = nothing)

Build a shifted incomplete-LU preconditioner for the trace-constrained matrix `A`.

A small diagonal shift keeps the incomplete factorization well-behaved:
`ilu(A + shift·I; τ)`. If `shift` is `nothing` it is scaled from the operator as
`shift_factor · max(norm(A, 1) / size(A, 1), eps(Float64))`, matching the
application helpers. The shift affects **only** the preconditioner, never the solved
system.

Requires the `QuantumFCSIterativeExt` extension (`using Krylov, IncompleteLU`).
"""
function shifted_ilu_preconditioner(A; kwargs...)
    throw(
        ArgumentError(
            "shifted_ilu_preconditioner requires the Krylov and IncompleteLU " *
                "packages. Run `using Krylov, IncompleteLU` to enable it."
        )
    )
end

# Extension hook for the iterative solve; the core catch-all errors informatively.
# Keep this signature a catch-all (`args...`) so the extension's specific
# `TraceConstrainedSystem` method *adds* a method rather than overwriting this one
# (method overwriting is forbidden during precompilation).
function _trace_constrained_steadystate_iterative(args...; kwargs...)
    throw(
        ArgumentError(
            "method=:iterative for trace_constrained_steadystate requires the Krylov " *
                "and IncompleteLU packages. Run `using Krylov, IncompleteLU`."
        )
    )
end

"""
    trace_constrained_steadystate(L; method = :iterative, kwargs...)
    trace_constrained_steadystate(H, J; method = :iterative, kwargs...)
    trace_constrained_steadystate(sys::TraceConstrainedSystem; method = :iterative, kwargs...)

Solve for the trace-constrained steady state, returning a lean
[`TraceConstrainedSteadyState`](@ref) that carries `rho_ss` and the reusable
preconditioner `Pl`.

`method = :lu` solves the sparse system directly (`A \\ b`) and returns `Pl = nothing`
— a reliable baseline for small/medium systems. `method = :iterative` builds a
shifted-ILU preconditioner and solves with GMRES (requires the
`QuantumFCSIterativeExt` extension); the resulting `Pl` is meant to be reused by the
iterative FCS backend.

Iterative keyword arguments (ignored by `:lu`):
* `Pl` — reuse an externally built preconditioner instead of building one.
* `u0` — initial guess for GMRES (warm start, e.g. a neighbouring solution).
* `τ`, `shift_factor`, `shift` — [`shifted_ilu_preconditioner`](@ref) options.
* `rtol` (`1e-10`), `atol` (`1e-14`), `itmax` (`200`), `memory` (`60`) — GMRES
  tolerances, iteration cap, and Krylov basis size (defaults match the applications).

Pass the low-level `sys::TraceConstrainedSystem` form to reuse a prebuilt system and
preconditioner across a continuation sweep without rebuilding either.
"""
function trace_constrained_steadystate(sys::TraceConstrainedSystem;
        method::Symbol = :iterative, kwargs...)
    if method === :lu
        return _trace_constrained_steadystate_lu(sys)
    elseif method === :iterative
        return _trace_constrained_steadystate_iterative(sys; kwargs...)
    else
        throw(ArgumentError(
            "Unknown steady-state method :$(method) (expected :lu or :iterative)."))
    end
end

function trace_constrained_steadystate(L; weight = nothing, kwargs...)
    return trace_constrained_steadystate(
        trace_constrained_system(L; weight = weight); kwargs...)
end

function trace_constrained_steadystate(H, J; weight = nothing, kwargs...)
    return trace_constrained_steadystate(
        trace_constrained_system(H, J; weight = weight); kwargs...)
end

# Direct sparse solve (no weak deps needed). Fast, exact baseline.
function _trace_constrained_steadystate_lu(sys::TraceConstrainedSystem)
    t0 = time_ns()
    ρvec = sys.A \ sys.b
    solve_seconds = (time_ns() - t0) / 1e9

    residual = norm(sys.A * ρvec - sys.b)
    relative_residual = residual / norm(sys.b)
    return _finalize_steadystate(sys.L, ρvec; Pl = nothing,
        converged = relative_residual < 1e-8, iterations = 0,
        residual = residual, relative_residual = relative_residual,
        ilu_seconds = 0.0, gmres_seconds = solve_seconds)
end

# Shared post-processing: reshape (column-major), hermitianize, trace-normalize,
# sparsify, and assemble the scalar diagnostics. Used by both backends.
function _finalize_steadystate(L, ρvec::AbstractVector; Pl, converged::Bool,
        iterations::Integer, residual::Real, relative_residual::Real,
        ilu_seconds::Real, gmres_seconds::Real)
    n = isqrt(length(ρvec))
    ρ = reshape(collect(ρvec), n, n)
    trace_error = abs(tr(ρ) - 1)
    hermiticity_error = norm(ρ - ρ')

    ρh = (ρ + ρ') / 2
    ρh ./= tr(ρh)
    rho_ss = SparseMatrixCSC{ComplexF64, Int}(sparse(ρh))

    stats = (; converged, iterations, residual, relative_residual,
        trace_error, hermiticity_error, ilu_seconds, gmres_seconds)
    return TraceConstrainedSteadyState(L, rho_ss, Pl, stats)
end

# --- Bridge to the existing FCS machinery ----------------------------------
"""
    prepare_fcs_context(ss::TraceConstrainedSteadyState; method = :iterative,
                        σ = nothing, τ = 0.05, rtol = 1e-8, itmax = 200, memory = 30)

Bridge a solved [`TraceConstrainedSteadyState`](@ref) into a reusable
[`PreparedLindbladFCS`](@ref) context, forwarding the steady-state preconditioner so
the iterative FCS backend does **not** rebuild an ILU.

With `method = :iterative` the steady-state `ss.Pl` is passed through as the FCS
preconditioner; with `method = :lu` the direct backend is used and `Pl` is ignored.
The steady state is not recomputed. The `σ`/`τ`/`rtol`/`itmax`/`memory` keywords tune
the FCS Drazin solve exactly as in the keyword [`prepare_fcs_context`](@ref).
"""
function prepare_fcs_context(ss::TraceConstrainedSteadyState;
        method::Symbol = :iterative,
        σ::Union{Nothing, Float64} = nothing,
        τ::Float64 = 0.05,
        rtol::Float64 = 1e-8,
        itmax::Int = 200,
        memory::Int = 30,
    )
    Pl = method === :iterative ? ss.Pl : nothing
    return prepare_fcs_context(; L = ss.L, rho_ss = ss.rho_ss,
        method = method, σ = σ, τ = τ, Pl = Pl,
        rtol = rtol, itmax = itmax, memory = memory)
end
