"""
    factorial_cumulants(cumulants::AbstractVector{<:Number})

Convert ordinary cumulants `[c1, c2, ..., cn]` to factorial cumulants
`[f1, f2, ..., fn]`.

The conversion uses the signed Stirling numbers of the first kind,

```
f_m = sum(s(m, j) * c_j for j in 1:m)
```

with `s(1, 1) = 1` and the recurrence
`s(m, j) = s(m - 1, j - 1) - (m - 1) * s(m - 1, j)`.

The physical factorial-cumulant interpretation assumes dimensionless count
cumulants. Dimensionful heat-current cumulants should be rescaled first.
"""
function factorial_cumulants(cumulants::AbstractVector{<:Number})
    n = length(cumulants)
    # Typed empty return keeps `_postprocess_cumulants` (and thus
    # `fcscumulants_recursive`) inferable to a concrete vector type.
    n == 0 && return empty(cumulants)

    stirling = zeros(Int, n, n)
    stirling[1, 1] = 1

    for m in 2:n
        for j in 1:m
            previous_diagonal = j == 1 ? 0 : stirling[m - 1, j - 1]
            previous_column = j == m ? 0 : stirling[m - 1, j]
            stirling[m, j] = previous_diagonal - (m - 1) * previous_column
        end
    end

    return [sum(stirling[m, j] * cumulants[j] for j in 1:m) for m in 1:n]
end

"""
    _postprocess_cumulants(cumulants, cumulant_type)

Internal helper used by `fcscumulants_recursive` to apply the requested output
type after the ordinary cumulants have been computed.

Use `cumulant_type == ""` to return `cumulants` unchanged, or
`cumulant_type == "factorial"` to convert them with [`factorial_cumulants`](@ref).
Unsupported values throw an `ArgumentError` with the accepted options.
"""
function _postprocess_cumulants(cumulants, cumulant_type::AbstractString)
    if cumulant_type == ""
        return cumulants
    elseif cumulant_type == "factorial"
        return factorial_cumulants(cumulants)
    else
        throw(ArgumentError("Unsupported cumulant_type=$(repr(cumulant_type)). Use \"\" or \"factorial\"."))
    end
end

"""
    fcscumulants_recursive(L, mJ, nC, rho_ss, nu; cumulant_type="")
    fcscumulants_recursive(H, J, mJ, nC, rho_ss, nu; cumulant_type="")

Calculate n-th zero-frequency cumulant of full counting statistics using a recursive scheme.

# Arguments
* `L`: Vectorized Liouvillian matrix (sparse or dense, ComplexF64)
Alternatively, one can provide the Hamiltonian and jump operators instead of `L`
* `H`: Hamiltonian operator (sparse or dense, Operator from QuantumOptics.jl)
* `J`: Vector of jump operators (sparse or dense, Operator from QuantumOptics.jl)
* `mJ`: Vector containing the monitored jump matrices (sparse operators in vectorized representation).
* `nC`: Number of cumulants to be calculated.
* `rho_ss`: Steady-state density matrix (sparse or dense, ComplexF64)
* `nu`: Vector of length `length(mJ)` with weights for each jump.

# Keyword arguments
* `method`: Drazin-solve backend, `:lu` (default) or `:iterative`.
* `σ`, `τ`, `rtol`, `itmax`, `memory`: options for the `:iterative` backend
  (see [`prepare_drazin_solver`](@ref)); ignored by `:lu`.
* `Pl`: an externally built preconditioner reused by the `:iterative` backend
  instead of building an ILU internally (see [`prepare_drazin_solver`](@ref)).
  `nothing` (default) builds the internal preconditioner; ignored by `:lu` and by
  the dense-Liouvillian method.
* `cumulant_type`: Optional keyword. Use `""` (default) for ordinary cumulants, or
  `"factorial"` to convert the result with [`factorial_cumulants`](@ref).
"""
function fcscumulants_recursive(
        L::SparseMatrixCSC{ComplexF64, Int},
        mJ::AbstractVector{<:SparseMatrixCSC{ComplexF64, Int}},
        nC::Integer,
        rho_ss::SparseMatrixCSC{ComplexF64, Int},
        nu::AbstractVector{<:Real};
        method::Symbol = :lu,
        σ = nothing,
        τ::Float64 = 0.05,
        Pl = nothing,
        rtol::Float64 = 1.0e-8,
        itmax::Int = 200,
        memory::Int = 30,
        cumulant_type::AbstractString = "",
    )
    if length(mJ) != length(nu)
        throw(ArgumentError("Length of mJ ($(length(mJ))) must match length of nu ($(length(nu)))."))
    end
    # Dimensions
    n = size(rho_ss, 1)           # matrix side
    l = n * n                     # vectorized length

    # Vectorized identity (diagonal entries of an n×n identity under vec)
    # Indices: 1:(n+1):l in column-major vectorization
    diag_idx = collect(1:(n + 1):l)
    vId = SparseVector{ComplexF64, Int}(l, diag_idx, fill(1.0 + 0.0im, n))

    # Vectorized steady state, normalized
    vrho_ss = SparseVector(vec(rho_ss ./ tr(rho_ss)))

    # Prepared Drazin solver, reused for every cumulant order. The :lu backend
    # caches a sparse LU (default); :iterative builds a matrix-free preconditioned
    # Krylov solver for large sparse Liouvillians (extension required). LU keeps the
    # established 1e-12 sparsify threshold; iterative uses `rtol` as the Krylov tol.
    # A supplied `Pl` lets :iterative reuse an externally built preconditioner
    # (e.g. the ILU from the steady-state solve) instead of building its own.
    #
    # To reuse this preparation across *several* observables that share `L` and
    # `rho_ss`, build a context once with [`prepare_fcs_context`](@ref) instead.
    solver = prepare_drazin_solver(
        L, vrho_ss, vId;
        method = method,
        rtol = (method === :lu ? 1.0e-12 : rtol),
        σ = σ, τ = τ, Pl = Pl, itmax = itmax, memory = memory
    )

    # Run the shared recursion with the freshly prepared solver.
    return _fcscumulants_recursive_prepared(
        solver, mJ, nC, vrho_ss, vId, nu;
        cumulant_type = cumulant_type
    )
end
# Dense method
function fcscumulants_recursive(
        L::Matrix{ComplexF64},
        mJ::AbstractVector{<:SparseMatrixCSC{ComplexF64, Int}},
        nC::Integer,
        rho_ss::Union{SparseMatrixCSC{ComplexF64, Int}, Matrix{ComplexF64}},
        nu::AbstractVector{<:Real};
        method::Symbol = :lu,
        cumulant_type::AbstractString = "",
        kwargs...,
    )
    # The iterative backend targets large *sparse* Liouvillians; a dense L is by
    # definition small, so the dense path always uses the cached direct solve.
    method === :lu || throw(
        ArgumentError(
            "fcscumulants_recursive with a dense Liouvillian supports only method=:lu " *
                "(got :$(method)); the :iterative backend requires a sparse L."
        )
    )
    # Dimensions
    n = size(rho_ss, 1)
    l = n * n

    # Cached solve operator: try LU, fall back to dense pseudoinverse (computed once).
    F = try
        lu(L)
    catch e
        e isa SingularException ? pinv(L) : rethrow()
    end

    # Vectorized identity as a sparse vector: indices 1:(n+1):l (column-major)
    diag_idx = collect(1:(n + 1):l)
    vId = SparseVector{ComplexF64, Int}(l, diag_idx, fill(1.0 + 0.0im, n))

    # ℒ(n) derivative matrices (still sparse is fine)
    Ln = [m_jumps(mJ; n = k, nu = nu) for k in 1:nC]

    # Vectorized steady-state, normalized, **dense** state vector
    trρ = tr(rho_ss)
    vrho1_dense = vec(Matrix(rho_ss) ./ trρ)  # Vector{ComplexF64}

    # Output cumulants
    vI = Vector{Float64}(undef, nC)

    # I₁ = Re( vId ⋅ (ℒ(1) * ρ_ss) )
    tmp = Vector{ComplexF64}(undef, l)             # dense work buffer length l
    mul!(tmp, Ln[1], vrho1_dense)                  # SparseMatrix * dense -> dense
    vI[1] = real(dot(vId, tmp))

    # States used in recursion, all **dense** for type stability
    vρ = Vector{Vector{ComplexF64}}(undef, nC)
    vρ[1] = vrho1_dense

    # Another dense buffer for α accumulation
    αbuf = zeros(ComplexF64, l)

    # Pre-computed once; passed to drazin_apply on every cumulant step
    vrho_ss_sparse = SparseVector(vrho1_dense)

    for ncur in 2:nC
        # valpha = Σ_{m=1}^{n-1} C(n-1,m) * ( vI[m]*vρ[n-m] - Ln[m]*vρ[n-m] )
        fill!(αbuf, 0)
        for m in 1:(ncur - 1)
            c = binomial(ncur - 1, m)

            # αbuf += c * vI[m] * vρ[n-m]
            vnm = vρ[ncur - m]
            @inbounds @simd for i in eachindex(vnm)
                αbuf[i] += c * vI[m] * vnm[i]
            end

            # tmp = Ln[m] * vρ[n-m]; αbuf -= c * tmp
            mul!(tmp, Ln[m], vnm)
            @inbounds @simd for i in eachindex(tmp)
                αbuf[i] -= c * tmp[i]
            end
        end

        # y_n = L^D * valpha  (project+gauge inside drazin_apply)
        vρ[ncur] = drazin_apply(L, αbuf, vrho_ss_sparse, vId; F = F)

        # I_n = Re( Σ_{m=1}^n C(n,m) * vId⋅(Ln[m] * vρ[n+1-m]) )
        acc = 0.0
        for m in 1:ncur
            mul!(tmp, Ln[m], vρ[ncur + 1 - m])
            acc += binomial(ncur, m) * real(dot(vId, tmp))
        end
        vI[ncur] = acc
    end

    return _postprocess_cumulants(vI, cumulant_type)
end

# ============================================================================
#  Prepared Drazin solvers
# ============================================================================
#
# The recursion applies the (projected) Drazin inverse of the Liouvillian `L`
# to `nC-1` right-hand sides. The work splits into a one-time *preparation*
# (factorization or preconditioner setup, reusable across cumulant orders) and
# a per-RHS *apply*. `prepare_drazin_solver` builds a reusable solver object and
# `drazin_solve` applies it.
#
# Two backends share one interface:
#   * `LUDrazinSolver`        — cached sparse LU (default, `method = :lu`).
#   * `IterativeDrazinSolver` — matrix-free, preconditioned Krylov solve
#                               (opt-in, `method = :iterative`). Defined in the
#                               `QuantumFCSIterativeExt` extension, which loads
#                               only when `Krylov` and `IncompleteLU` are present.

"""
    DrazinSolver

Abstract supertype for prepared Drazin-inverse solvers. A concrete solver caches
whatever per-Liouvillian work (factorization, preconditioner, matrix-free
operator) is reused across cumulant orders; apply it with [`drazin_solve`](@ref).
"""
abstract type DrazinSolver end

"""
    drazin_solve(solver::DrazinSolver, α) -> SparseVector

Apply the prepared (projected) Drazin inverse held by `solver` to the RHS `α`.
Projects `α` onto range(L), solves, re-imposes the trace-zero gauge, and returns
a sparsified result — identical pre/post-processing to [`drazin_apply`](@ref).
"""
function drazin_solve end

"""
    _fcscumulants_recursive_prepared(solver, mJ, nC, vrho_ss, vId, nu; cumulant_type="")

Core zero-frequency cumulant recursion, driven by an **already prepared** Drazin
`solver` (see [`prepare_drazin_solver`](@ref)).

This is the shared engine behind both the ordinary [`fcscumulants_recursive`](@ref)
path — which prepares a solver internally for one observable — and the "prepare
once, evaluate many observables" path exposed through `prepare_fcs_context`. Only
the observable-dependent data enters here: the monitored jumps `mJ`, their weights
`nu`, and the requested number of cumulants `nC`. The observable-independent inputs
`solver`, `vrho_ss` (normalized, vectorized steady state) and `vId` (vectorized
identity / trace functional) are built once and may be reused across many calls.

Internal helper; not exported.
"""
function _fcscumulants_recursive_prepared(
        solver::DrazinSolver,
        mJ::AbstractVector{<:SparseMatrixCSC{ComplexF64, Int}},
        nC::Integer,
        vrho_ss::SparseVector{ComplexF64, Int},
        vId::AbstractVector{ComplexF64},
        nu::AbstractVector{<:Real};
        cumulant_type::AbstractString = "",
    )
    l = length(vrho_ss)                       # vectorized length (n²)

    # d/dχ n-derivatives ℒ(n) — the only observable-dependent super-operators.
    Ln = [m_jumps(mJ; n = k, nu = nu) for k in 1:nC]

    # Outputs
    vI = Vector{Float64}(undef, nC)

    # First cumulant: I₁ = Re( vId⋅(ℒ(1)*ρ_ss) )
    vI[1] = real(dot(vId, Ln[1] * vrho_ss))

    # States used in recursion
    vrho = Vector{SparseVector{ComplexF64, Int}}(undef, nC)
    vrho[1] = vrho_ss

    # --- Work buffers (reused) ---
    # dense buffers of length l to avoid repeated allocations
    tmp = Vector{ComplexF64}(undef, l)        # for Ln[m] * vrho[·]
    αbuf = zeros(ComplexF64, l)               # accumulates valpha densely

    # main recursion
    for ncur in 2:nC
        # Build valpha = Σ_{m=1}^{n-1} binom(n-1,m) * ( vI[m]*vrho[n-m] - Ln[m]*vrho[n-m] )
        fill!(αbuf, 0)
        for m in 1:(ncur - 1)
            c = binomial(ncur - 1, m)
            # αbuf += c * vI[m] * vrho[n-m]
            sv = vrho[ncur - m]
            @inbounds for k in 1:nnz(sv)
                i = rowvals(sv)[k]
                αbuf[i] += c * vI[m] * nonzeros(sv)[k]
            end
            # tmp = Ln[m] * vrho[n-m]; αbuf -= c * tmp
            mul!(tmp, Ln[m], sv)             # SparseMatrix * SparseVector -> dense tmp
            @inbounds @simd for i in eachindex(tmp)
                αbuf[i] -= c * tmp[i]
            end
        end

        # y_n = L^D * valpha  (project+gauge inside the prepared solver), sparse result
        vrho[ncur] = drazin_solve(solver, αbuf)

        # I_n = Re( Σ_{m=1}^n binom(n,m) * vId⋅(Ln[m] * vrho[n+1-m]) )
        acc = 0.0
        for m in 1:ncur
            mul!(tmp, Ln[m], vrho[ncur + 1 - m])
            acc += binomial(ncur, m) * real(dot(vId, tmp))
        end
        vI[ncur] = acc
    end

    return _postprocess_cumulants(vI, cumulant_type)
end

"""
    prepare_drazin_solver(L, ρ, vId; method=:lu, rtol=1e-12, σ=nothing, τ=0.05,
                          Pl=nothing, itmax=200, memory=30) -> DrazinSolver

Build a reusable solver for the (projected) Drazin inverse of the singular
Liouvillian `L`, given its vectorized steady state `ρ` (right null vector) and
the vectorized identity / trace functional `vId` (left null vector).

`method = :lu` (default) caches a sparse LU and reproduces the existing behavior.
`method = :iterative` builds a matrix-free, preconditioned Krylov solver suited
to large sparse Liouvillians where direct-LU fill-in is prohibitive; it requires
the `QuantumFCSIterativeExt` extension (`using Krylov, IncompleteLU`). Keyword
arguments `σ`/`τ`/`Pl`/`itmax`/`memory`/`rtol` configure the iterative backend and
are ignored by the LU backend.

`Pl` supplies an externally built preconditioner for the `:iterative` backend to
reuse instead of building its own shifted ILU — for example the ILU already
computed for the steady-state solve at the same or a neighbouring parameter point.
It must support `LinearAlgebra.ldiv!(y, Pl, x)` and `ldiv!(Pl, x)` and approximate
`L⁻¹` up to a diagonal shift, a low-rank gauge term, and mild parameter drift; it
is applied on the right so GMRES converges on the true residual. When `Pl` is
supplied, `σ` and `τ` are ignored.
"""
function prepare_drazin_solver(
        L::SparseMatrixCSC{ComplexF64, Int},
        ρ::SparseVector{ComplexF64, Int},
        vId::AbstractVector{ComplexF64};
        method::Symbol = :lu,
        rtol::Float64 = 1.0e-12,
        kwargs...
    )
    if method === :lu
        # Cached solve operator: LU, or dense pseudoinverse if exactly singular.
        F = try
            lu(L)
        catch e
            e isa SingularException ? pinv(Matrix(L)) : rethrow()
        end
        return LUDrazinSolver(L, F, ρ, vId, rtol)
    elseif method === :iterative
        return _prepare_iterative_drazin_solver(L, ρ, vId; rtol = rtol, kwargs...)
    else
        throw(ArgumentError("Unknown Drazin solver method :$(method) (expected :lu or :iterative)."))
    end
end

# Cached-LU backend. `F` is an LU factorization or a dense pseudoinverse.
struct LUDrazinSolver{TL, TF, Tρ, TI} <: DrazinSolver
    L::TL
    F::TF
    ρ::Tρ
    vId::TI
    rtol::Float64
end

# Apply: delegate to drazin_apply, which already does project → solve → gauge →
# sparsify with the cached factorization.
drazin_solve(s::LUDrazinSolver, α::AbstractVector) =
    drazin_apply(s.L, α, s.ρ, s.vId; F = s.F, rtol = s.rtol)

# Extension hook. The catch-all errors with a clear message; the
# `QuantumFCSIterativeExt` extension adds a more specific (and therefore
# preferred) method once Krylov and IncompleteLU are loaded.
function _prepare_iterative_drazin_solver(args...; kwargs...)
    throw(
        ArgumentError(
            "The :iterative Drazin backend requires the Krylov and IncompleteLU " *
                "packages. Run `using Krylov, IncompleteLU` to enable it."
        )
    )
end

# --- Shared building blocks for Drazin backends ----------------------------
# These reproduce the project / gauge / sparsify steps of `drazin_apply` so the
# iterative extension can reuse them without densifying ρ.

# Project the RHS onto range(L): α' = α - ρ (vId⋅α). Returns a dense copy.
function _drazin_project(
        α::AbstractVector, ρ::SparseVector{ComplexF64, Int},
        vId::AbstractVector
    )
    sα = dot(vId, α)
    y = Vector{ComplexF64}(undef, length(α))
    copyto!(y, α)
    @inbounds for k in 1:nnz(ρ)
        i = rowvals(ρ)[k]
        y[i] -= sα * nonzeros(ρ)[k]
    end
    return y
end

# Enforce the trace-zero gauge in place: y ← y - ρ (vId⋅y).
function _drazin_gauge!(
        y::AbstractVector, ρ::SparseVector{ComplexF64, Int},
        vId::AbstractVector
    )
    sy = dot(vId, y)
    @inbounds for k in 1:nnz(ρ)
        i = rowvals(ρ)[k]
        y[i] -= sy * nonzeros(ρ)[k]
    end
    return y
end

# One-pass sparsification with absolute/relative threshold.
function _drazin_sparsify(
        y::AbstractVector; rtol::Float64 = 1.0e-12,
        atol::Float64 = 0.0
    )
    thr = max(atol, rtol * norm(y, Inf))
    nzI = Int[];        sizehint!(nzI, length(y))
    nzV = ComplexF64[]; sizehint!(nzV, length(y))
    @inbounds for i in eachindex(y)
        yi = y[i]
        if abs(yi) > thr
            push!(nzI, i); push!(nzV, yi)
        end
    end
    return sparsevec(nzI, nzV, length(y))
end

# Apply the cached solve operator to a (dense) RHS. `F` is one of:
#   nothing       → factorize L on the fly (may throw SingularException)
#   Factorization → reuse a caller-supplied LU/etc (fast path)
#   AbstractMatrix → cached dense pseudoinverse: F * y (avoids refactorization)
# Falls back to a dense pseudoinverse if L turns out to be exactly singular.
function _drazin_linear_solve(F, L::AbstractMatrix, y::AbstractVector)
    F isa AbstractMatrix && return F * y
    try
        return F === nothing ? (L \ y) : (F \ y)
    catch e
        e isa SingularException ? (pinv(Matrix(L)) * y) : rethrow()
    end
end

# ============================================================================
#  Low-level Drazin building blocks
# ============================================================================
#
# `drazin` builds the full (dense) Drazin inverse; `drazin_apply` applies the
# projected inverse to a single RHS by solving a linear system; `m_jumps` builds
# the counting-field derivative super-operators ℒ(n). These underpin the LU
# backend and the public recursion above.

"""
    drazin(L, vrho_ss, vId, IdL)

Calculate the Drazin inverse of a Liouvillian defined by the Hamiltonian H and jump operators J.
 
# Arguments
* `L` : Liouvillian matrix
* `vrho_ss`: vectorised density matrix specifying the steady-state of the Liouvillian.
* `vId`: vectorised identity matrix (1×N row or vector)
* `IdL`: Identity matrix in Liouville space (N×N)

# Returns
A dense matrix holding the (projected) Drazin inverse `Lᴰ` of `L`.

!!! note
    This builds the full Drazin inverse via a dense pseudoinverse and is intended
    for small systems or reference checks. For computing cumulants prefer
    [`prepare_drazin_solver`](@ref) / [`drazin_solve`](@ref), which only apply the
    inverse to the right-hand sides actually needed.
 """
function drazin(L, vrho_ss, vId, IdL)
    # Ensure vId is a 1×N row for outer product
    vId_row = isa(vId, AbstractVector) ? (vId') : vId
    # Projector onto range(L) assuming vrho_ss spans kernel(L)
    Q = IdL - vrho_ss * vId_row
    # The Drazin inverse is computed by projecting the Moore-Penrose pseudo-inverse, computed using pinv.
    LD = Q * pinv(Matrix(L)) * Q
    return LD
end

"""
    m_jumps(mJ; n=1, nu=vcat(fill(+1, length(mJ)÷2), fill(-1, length(mJ)÷2)))

Calculate the vectorized super-operator ℒ(n) = ∑ₖ (νₖ)ⁿ (Lₖ*)⊗Lₖ.
# Arguments
* `mJ`: List of monitored jumps
* `n` : Power of the weights νₖ. By default set to 1, since this case appears more often.
* `nu`: vector of length length(mJ) with weights for each jump operator.
"""
function m_jumps(mJ::AbstractVector{<:SparseMatrixCSC{ComplexF64, Int}}; n::Integer = 1, nu = vcat(fill(+1, Int(length(mJ) ÷ 2)), fill(-1, Int(length(mJ) ÷ 2))))
    # Sum of sparse Kronecker products stays sparse; element types remain ComplexF64
    return sum(nu[k]^n * kron(conj(mJ[k]), mJ[k]) for k in 1:length(mJ))
end

"""
    drazin_apply(L, α, ρ, vId; F=nothing, rtol=1e-12, atol=0.0)

Apply the (projected) Drazin inverse of the Liouvillean `L` to the vector `α` by solving a linear system.

# Arguments
- `L`: Liouvillean operator (matrix).
- `α`: Right-hand side vector.
- `ρ`: Steady-state vector.
- `vId`: Vectorized identity vector.
- `F`: Optional factorization of `L` to reuse (default: `nothing`).
- `rtol`: Relative tolerance for the solver (default: `1e-12`).
- `atol`: Absolute tolerance for the solver (default: `0.0`).

# Returns
A (sparse) vector representing the result of applying the projected Drazin inverse.

"""
function drazin_apply(
        L::SparseMatrixCSC{ComplexF64, Int},
        α::SparseVector{ComplexF64, Int},
        ρ::SparseVector{ComplexF64, Int},
        vId::AbstractVector{ComplexF64};
        F::Union{Nothing, Factorization, AbstractMatrix} = nothing,
        rtol::Float64 = 1.0e-12,
        atol::Float64 = 0.0
    )
    y = _drazin_project(α, ρ, vId)          # α' = α - ρ (vId⋅α), dense
    y = _drazin_linear_solve(F, L, y)       # L⁺ α'
    _drazin_gauge!(y, ρ, vId)               # re-impose trace-zero gauge in place
    return _drazin_sparsify(y; rtol = rtol, atol = atol)
end

# Dense-RHS variant: same pipeline, only the input vector is dense.
function drazin_apply(
        L::SparseMatrixCSC{ComplexF64, Int},
        α::AbstractVector{ComplexF64},       # DENSE RHS here
        ρ::SparseVector{ComplexF64, Int},
        vId::SparseVector{ComplexF64, Int};
        F::Union{Nothing, Factorization, AbstractMatrix} = nothing,
        rtol::Float64 = 1.0e-12,
        atol::Float64 = 0.0
    )
    y = _drazin_project(α, ρ, vId)
    y = _drazin_linear_solve(F, L, y)
    _drazin_gauge!(y, ρ, vId)
    return _drazin_sparsify(y; rtol = rtol, atol = atol)
end

# vId provided as a 1×N row (e.g., SparseMatrixCSC): flatten and forward.
function drazin_apply(
        L::SparseMatrixCSC{T, Int},
        x::AbstractVector{T},
        vrho_ss::AbstractVector{T},
        vId_row::AbstractMatrix{T}
    ) where {T <: Number}
    size(vId_row, 1) == 1 ||
        throw(DimensionMismatch("vId_row must be a 1×N row (got size $(size(vId_row)))."))
    vId_vec = vec(permutedims(vId_row))::Vector{T}
    return drazin_apply(L, x, vrho_ss, vId_vec)
end

# Dense-Liouvillian variant: returns a dense vector (no sparsification).
function drazin_apply(
        L::Matrix{ComplexF64},
        α::AbstractVector{ComplexF64},
        ρ::SparseVector{ComplexF64, Int},
        vId::SparseVector{ComplexF64, Int};
        F::Union{Nothing, Factorization, AbstractMatrix} = nothing
    )
    size(L, 1) == size(L, 2) == length(α) == length(vId) ||
        throw(
        DimensionMismatch(
            "Incompatible sizes: L is $(size(L)), length(α)=$(length(α)), length(vId)=$(length(vId))."
        )
    )
    y = _drazin_project(α, ρ, vId)
    y = _drazin_linear_solve(F, L, y)
    _drazin_gauge!(y, ρ, vId)
    return y  # Vector{ComplexF64}
end
