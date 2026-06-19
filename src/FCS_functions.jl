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
    n == 0 && return Number[]

    stirling = zeros(Int, n, n)
    stirling[1, 1] = 1

    for m = 2:n
        for j = 1:m
            previous_diagonal = j == 1 ? 0 : stirling[m - 1, j - 1]
            previous_column = j == m ? 0 : stirling[m - 1, j]
            stirling[m, j] = previous_diagonal - (m - 1) * previous_column
        end
    end

    return [sum(stirling[m, j] * cumulants[j] for j = 1:m) for m = 1:n]
end

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
    nu::AbstractVector{<:Real},
)
    if length(mJ) != length(nu)
        throw(ArgumentError("Length of mJ ($(length(mJ))) must match length of nu ($(length(nu)))."))
    end
    # Dimensions
    n = size(rho_ss, 1)           # matrix side
    l = n * n                     # vectorized length

    # Cached factorization of Liouvillian
    F = try
        lu(L)
    catch e
        if e isa SingularException
            nothing
        else
            rethrow()
        end
    end

    # Vectorized identity (diagonal entries of an n×n identity under vec)
    # Indices: 1:(n+1):l in column-major vectorization
    diag_idx = collect(1:(n+1):l)
    vId = SparseVector{ComplexF64,Int}(l, diag_idx, fill(1.0 + 0.0im, n))

    # d/dχ n-derivatives ℒ(n)
    Ln = [m_jumps(mJ; n = k, nu = nu) for k = 1:nC]

    # Vectorized steady state, normalized
    vrho_ss = SparseVector(vec(rho_ss ./ tr(rho_ss)))

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

        # y_n = L^D * valpha  (project+gauge inside drazin_apply), sparse result
        vrho[ncur] = drazin_apply(L, αbuf, vrho_ss, vId; rtol = 1e-12)

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
    nu::AbstractVector{<:Real},
)
    # Dimensions
    n = size(rho_ss, 1)
    l = n*n

    # Factorize dense L once (pivoted LU, with protection from singular exceptions)
    F = try
        lu(L)
    catch e
        if e isa SingularException
            nothing
        else
            rethrow()
        end
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
        vρ[ncur] = drazin_apply(L, αbuf, SparseVector(vrho1_dense), vId)

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

"""
    drazin(L, vrho_ss, vId, IdL)
    
Calculate the Drazin inverse of a Liouvillian defined by the Hamiltonian H and jump operators J.
 
# Arguments
* `L` : Liouvillian matrix
* `vrho_ss`: vectorised density matrix specifying the steady-state of the Liouvillian.
* `vId`: vectorised identity matrix (1×N row or vector)
* `IdL`: Identity matrix in Liouville space (N×N)

# Returns
    Drazin inverse as a (sparse)
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
    m_jumps(mJ; n=1; nu = vcat(fill(-1, Int(length(J)/2)),fill(1, Int(length(J)/2))))

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
                      F::Union{Nothing, Factorization}=nothing,
                      rtol::Float64=1e-12,
                      atol::Float64=0.0)

    # Project RHS onto range(L): α' = α - ρ * (vId⋅α)
    sα = dot(vId, α)                      # Complex scalar (uses conjugate in dot)
    αp = α - (ρ .* sα)                    # stays SparseVector

    # Solve y = L \\ αp   (reuse factorization if provided). If L is singular (expected),
    # fall back to dense pseudo-inverse solution.
    y = try
        F === nothing ? (L \ Vector(αp)) : (F \ Vector(αp))
    catch e
        if e isa SingularException
            pinv(Matrix(L)) * Vector(αp)
        else
            rethrow()
        end
    end

    # Enforce gauge: y ← y - ρ * (vId⋅y), but do it *in-place* without densifying ρ
    sy = dot(vId, y)
    @inbounds for k in 1:nnz(ρ)
        i = rowvals(ρ)[k]                 # index of k-th stored entry
        y[i] -= sy * nonzeros(ρ)[k]       # y[i] -= sy * ρ[i]
    end

    # One-pass sparsification with absolute/relative threshold
    thr = max(atol, rtol * norm(y, Inf))
    nzI = Int[]; nzV = ComplexF64[]
    @inbounds for i in eachindex(y)
        yi = y[i]
        if abs(yi) > thr
            push!(nzI, i); push!(nzV, yi)
        end
    end
    return sparsevec(nzI, nzV, length(y))  # SparseVector{ComplexF64,Int}
end


# Add a dense-RHS method
function drazin_apply(L::SparseMatrixCSC{ComplexF64,Int},
                      α::AbstractVector{ComplexF64},       # DENSE RHS here
                      ρ::SparseVector{ComplexF64,Int},
                      vId::SparseVector{ComplexF64,Int};
                      F::Union{Nothing,Factorization}=nothing,
                      rtol::Float64=1e-12,
                      atol::Float64=0.0)

    # Project RHS onto range(L): α' = α - ρ * (vId⋅α)
    sα = dot(vId, α)
    # work: copy α into y (we’ll reuse it as the solve output as well)
    y = copy(α)
    @inbounds for k in 1:nnz(ρ)
        i = rowvals(ρ)[k]
        y[i] -= sα * nonzeros(ρ)[k]
    end

    # Solve once (reuse factorization if given). If L is singular, fall back to dense pseudo-inverse.
    y = try
        if F === nothing
            L \ y
        else
            F \ y
        end
    catch e
        if e isa SingularException
            pinv(Matrix(L)) * y
        else
            rethrow()
        end
    end

    # Enforce gauge: y ← y - ρ * (vId⋅y)
    sy = dot(vId, y)
    @inbounds for k in 1:nnz(ρ)
        i = rowvals(ρ)[k]
        y[i] -= sy * nonzeros(ρ)[k]
    end

    # One-pass sparsification
    thr = max(atol, rtol * norm(y, Inf))
    nzI = Int[]; nzV = ComplexF64[]
    @inbounds for i in eachindex(y)
        yi = y[i]
        if abs(yi) > thr
            push!(nzI, i); push!(nzV, yi)
        end
    end
    return sparsevec(nzI, nzV, length(y))
end

# vId provided as a 1×N row (e.g., SparseMatrixCSC)
function drazin_apply(L::SparseMatrixCSC{T,Int},
                      x::AbstractVector{T},
                      vrho_ss::AbstractVector{T},
                      vId_row::AbstractMatrix{T}) where {T<:Number}
    @assert size(vId_row,1) == 1
    vId_vec = vec(permutedims(vId_row)) :: Vector{T}
    return drazin_apply(L, x, vrho_ss, vId_vec)
end

function drazin_apply(L::Matrix{ComplexF64},
                      α::AbstractVector{ComplexF64},
                      ρ::SparseVector{ComplexF64,Int},
                      vId::SparseVector{ComplexF64,Int};
                      F::Union{Nothing, Factorization}=nothing)

    @assert size(L,1) == size(L,2) == length(α) == length(vId)

    # α' = α - ρ * (vId⋅α)   (project RHS onto range(L))
    sα = dot(vId, α)
    y = copy(α)
    @inbounds for k = 1:nnz(ρ)
        i = rowvals(ρ)[k]
        y[i] -= sα * nonzeros(ρ)[k]
    end

    # Solve Ly = α'  (reuse factorization if provided)
    y = F === nothing ? (L \ y) : (F \ y)

    # Gauge: y ← y - ρ * (vId⋅y)
    sy = dot(vId, y)
    @inbounds for k = 1:nnz(ρ)
        i = rowvals(ρ)[k]
        y[i] -= sy * nonzeros(ρ)[k]
    end

    return y  # Vector{ComplexF64}
end
