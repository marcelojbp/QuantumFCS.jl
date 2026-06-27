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

# Matrix-free gauge-fixed operator A·x = L·x + ρ·(vId·x).
struct GaugeOp{TL,Tρ,TV}
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

# Prepared iterative solver: matrix-free operator + reusable preconditioner.
struct IterativeDrazinSolver{TA,TP,Tρ,TV} <: DrazinSolver
    A::TA              # matrix-free GaugeOp
    P::TP              # ILU preconditioner of (L − σI)
    ρ::Tρ
    vId::TV
    rtol::Float64      # Krylov relative tolerance
    atol::Float64      # Krylov absolute tolerance
    itmax::Int
    memory::Int        # GMRES restart memory
    sparsify_rtol::Float64
end

# More specific than the core catch-all, so this wins once the extension loads.
function QuantumFCS._prepare_iterative_drazin_solver(
        L::SparseMatrixCSC{ComplexF64,Int},
        ρ::SparseVector{ComplexF64,Int},
        vId::AbstractVector{ComplexF64};
        σ = nothing,
        τ::Float64 = 0.05,
        rtol::Float64 = 1e-8,
        atol::Float64 = 1e-12,
        itmax::Int = 200,
        memory::Int = 30,
        sparsify_rtol::Float64 = 1e-12)

    # Auto-scale the shift from the operator magnitude when not supplied. The shift
    # only needs to lift the near-zero mode enough for a stable incomplete LU; a
    # small fraction of the largest entry generalizes across systems.
    σeff = σ === nothing ? 0.01 * maximum(abs, nonzeros(L)) : Float64(σ)

    Ls = L - σeff * I                  # sparse, same sparsity pattern as L
    P  = IncompleteLU.ilu(Ls; τ = τ)
    A  = GaugeOp(L, ρ, vId)

    return IterativeDrazinSolver(A, P, ρ, vId, rtol, atol, itmax, memory, sparsify_rtol)
end

function QuantumFCS.drazin_solve(s::IterativeDrazinSolver, α::AbstractVector)
    # Project RHS onto range(L): α' = α - ρ (vId·α).
    αp = _drazin_project(α, s.ρ, s.vId)

    # Preconditioned GMRES on the matrix-free gauge-fixed operator.
    y, stats = Krylov.gmres(s.A, αp;
        M = s.P, ldiv = true,
        rtol = s.rtol, atol = s.atol,
        itmax = s.itmax, memory = s.memory)

    stats.solved || @warn "Iterative Drazin solve did not converge" niter=stats.niter rtol=s.rtol

    # Re-impose trace-zero gauge, then sparsify.
    _drazin_gauge!(y, s.ρ, s.vId)
    return _drazin_sparsify(y; rtol = s.sparsify_rtol)
end

end # module
