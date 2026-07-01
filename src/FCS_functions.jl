"""
    fcscumulants_recursive(L, mJ, nC, rho_ss, nu)
    fcscumulants_recursive(H, J, mJ, nC, rho_ss, nu)

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
    rtol::Float64 = 1e-8,
    itmax::Int = 200,
    memory::Int = 30,
)
    if length(mJ) != length(nu)
        throw(ArgumentError("Length of mJ ($(length(mJ))) must match length of nu ($(length(nu)))."))
    end
    # Dimensions
    n = size(rho_ss, 1)           # matrix side
    l = n * n                     # vectorized length

    # Vectorized identity (diagonal entries of an n×n identity under vec)
    # Indices: 1:(n+1):l in column-major vectorization
    diag_idx = collect(1:(n+1):l)
    vId = SparseVector{ComplexF64,Int}(l, diag_idx, fill(1.0 + 0.0im, n))

    # d/dχ n-derivatives ℒ(n)
    Ln = [m_jumps(mJ; n = k, nu = nu) for k = 1:nC]

    # Vectorized steady state, normalized
    vrho_ss = SparseVector(vec(rho_ss ./ tr(rho_ss)))

    # Prepared Drazin solver, reused for every cumulant order. The :lu backend
    # caches a sparse LU (default); :iterative builds a matrix-free preconditioned
    # Krylov solver for large sparse Liouvillians (extension required). LU keeps the
    # established 1e-12 sparsify threshold; iterative uses `rtol` as the Krylov tol.
    solver = prepare_drazin_solver(L, vrho_ss, vId;
        method = method,
        rtol = (method === :lu ? 1e-12 : rtol),
        σ = σ, τ = τ, itmax = itmax, memory = memory)

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
    for ncur = 2:nC
        # Build valpha = Σ_{m=1}^{n-1} binom(n-1,m) * ( vI[m]*vrho[n-m] - Ln[m]*vrho[n-m] )
        fill!(αbuf, 0)
        for m = 1:(ncur-1)
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
        for m = 1:ncur
            mul!(tmp, Ln[m], vrho[ncur + 1 - m])
            acc += binomial(ncur, m) * real(dot(vId, tmp))
        end
        vI[ncur] = acc
    end

    return vI
end
# Dense method
function fcscumulants_recursive(
    L::Matrix{ComplexF64},
    mJ::AbstractVector{<:SparseMatrixCSC{ComplexF64,Int}},
    nC::Integer,
    rho_ss::Union{SparseMatrixCSC{ComplexF64,Int}, Matrix{ComplexF64}},
    nu::AbstractVector{<:Real};
    method::Symbol = :lu,
    kwargs...,
)
    # The iterative backend targets large *sparse* Liouvillians; a dense L is by
    # definition small, so the dense path always uses the cached direct solve.
    method === :lu || throw(ArgumentError(
        "fcscumulants_recursive with a dense Liouvillian supports only method=:lu " *
        "(got :$(method)); the :iterative backend requires a sparse L."))
    # Dimensions
    n = size(rho_ss, 1)
    l = n*n

    # Cached solve operator: try LU, fall back to dense pseudoinverse (computed once).
    F = try
        lu(L)
    catch e
        e isa SingularException ? pinv(L) : rethrow()
    end
    
    # Vectorized identity as a sparse vector: indices 1:(n+1):l (column-major)
    diag_idx = collect(1:(n+1):l)
    vId = SparseVector{ComplexF64,Int}(l, diag_idx, fill(1.0 + 0.0im, n))

    # ℒ(n) derivative matrices (still sparse is fine)
    Ln = [m_jumps(mJ; n = k, nu = nu) for k = 1:nC]

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

    for ncur = 2:nC
        # valpha = Σ_{m=1}^{n-1} C(n-1,m) * ( vI[m]*vρ[n-m] - Ln[m]*vρ[n-m] )
        fill!(αbuf, 0)
        for m = 1:(ncur-1)
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
        for m = 1:ncur
            mul!(tmp, Ln[m], vρ[ncur + 1 - m])
            acc += binomial(ncur, m) * real(dot(vId, tmp))
        end
        vI[ncur] = acc
    end

    return vI
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
    prepare_drazin_solver(L, ρ, vId; method=:lu, rtol=1e-12, σ=nothing, τ=0.05,
                          itmax=200, memory=30) -> DrazinSolver

Build a reusable solver for the (projected) Drazin inverse of the singular
Liouvillian `L`, given its vectorized steady state `ρ` (right null vector) and
the vectorized identity / trace functional `vId` (left null vector).

`method = :lu` (default) caches a sparse LU and reproduces the existing behavior.
`method = :iterative` builds a matrix-free, preconditioned Krylov solver suited
to large sparse Liouvillians where direct-LU fill-in is prohibitive; it requires
the `QuantumFCSIterativeExt` extension (`using Krylov, IncompleteLU`). Keyword
arguments `σ`/`τ`/`itmax`/`memory`/`rtol` configure the iterative backend and are
ignored by the LU backend.
"""
function prepare_drazin_solver(L::SparseMatrixCSC{ComplexF64,Int},
                               ρ::SparseVector{ComplexF64,Int},
                               vId::AbstractVector{ComplexF64};
                               method::Symbol = :lu,
                               rtol::Float64 = 1e-12,
                               kwargs...)
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
struct LUDrazinSolver{TL,TF,Tρ,TI} <: DrazinSolver
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
    error("The :iterative Drazin backend requires the Krylov and IncompleteLU " *
          "packages. Run `using Krylov, IncompleteLU` to enable it.")
end

# --- Shared building blocks for Drazin backends ----------------------------
# These reproduce the project / gauge / sparsify steps of `drazin_apply` so the
# iterative extension can reuse them without densifying ρ.

# Project the RHS onto range(L): α' = α - ρ (vId⋅α). Returns a dense copy.
function _drazin_project(α::AbstractVector, ρ::SparseVector{ComplexF64,Int},
                         vId::AbstractVector)
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
function _drazin_gauge!(y::AbstractVector, ρ::SparseVector{ComplexF64,Int},
                        vId::AbstractVector)
    sy = dot(vId, y)
    @inbounds for k in 1:nnz(ρ)
        i = rowvals(ρ)[k]
        y[i] -= sy * nonzeros(ρ)[k]
    end
    return y
end

# One-pass sparsification with absolute/relative threshold.
function _drazin_sparsify(y::AbstractVector; rtol::Float64 = 1e-12,
                          atol::Float64 = 0.0)
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
    return sum(nu[k]^n * kron(conj(mJ[k]), mJ[k]) for k = 1:length(mJ))
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
function drazin_apply(L::SparseMatrixCSC{ComplexF64,Int},
                      α::SparseVector{ComplexF64,Int},
                      ρ::SparseVector{ComplexF64,Int},
                      vId::AbstractVector{ComplexF64};
                      F::Union{Nothing, Factorization, AbstractMatrix}=nothing,
                      rtol::Float64=1e-12,
                      atol::Float64=0.0)
    y = _drazin_project(α, ρ, vId)          # α' = α - ρ (vId⋅α), dense
    y = _drazin_linear_solve(F, L, y)       # L⁺ α'
    _drazin_gauge!(y, ρ, vId)               # re-impose trace-zero gauge in place
    return _drazin_sparsify(y; rtol = rtol, atol = atol)
end

# Dense-RHS variant: same pipeline, only the input vector is dense.
function drazin_apply(L::SparseMatrixCSC{ComplexF64,Int},
                      α::AbstractVector{ComplexF64},       # DENSE RHS here
                      ρ::SparseVector{ComplexF64,Int},
                      vId::SparseVector{ComplexF64,Int};
                      F::Union{Nothing,Factorization,AbstractMatrix}=nothing,
                      rtol::Float64=1e-12,
                      atol::Float64=0.0)
    y = _drazin_project(α, ρ, vId)
    y = _drazin_linear_solve(F, L, y)
    _drazin_gauge!(y, ρ, vId)
    return _drazin_sparsify(y; rtol = rtol, atol = atol)
end

# vId provided as a 1×N row (e.g., SparseMatrixCSC): flatten and forward.
function drazin_apply(L::SparseMatrixCSC{T,Int},
                      x::AbstractVector{T},
                      vrho_ss::AbstractVector{T},
                      vId_row::AbstractMatrix{T}) where {T<:Number}
    @assert size(vId_row,1) == 1
    vId_vec = vec(permutedims(vId_row)) :: Vector{T}
    return drazin_apply(L, x, vrho_ss, vId_vec)
end

# Dense-Liouvillian variant: returns a dense vector (no sparsification).
function drazin_apply(L::Matrix{ComplexF64},
                      α::AbstractVector{ComplexF64},
                      ρ::SparseVector{ComplexF64,Int},
                      vId::SparseVector{ComplexF64,Int};
                      F::Union{Nothing, Factorization, AbstractMatrix}=nothing)
    @assert size(L,1) == size(L,2) == length(α) == length(vId)
    y = _drazin_project(α, ρ, vId)
    y = _drazin_linear_solve(F, L, y)
    _drazin_gauge!(y, ρ, vId)
    return y  # Vector{ComplexF64}
end


