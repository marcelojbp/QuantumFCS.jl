# Backend bindings for QuantumToolbox.jl.
#
# QuantumToolbox represents every quantum object as a `QuantumObject{ObjType,...}`
# where `ObjType` (e.g. `Operator`, `SuperOperator`) is a *type tag* stored in the
# first type parameter, not the concrete struct. We therefore dispatch on
# `QuantumObject{<:Operator}` and peel the backing array off the `.data` field,
# exactly mirroring the QuantumOptics backend.
#
# Vectorization convention matches the core engine and QuantumOptics:
# `mat2vec == vec` (column-major), `spre(A) = kron(I, A)`, `spost(B) = kron(transpose(B), I)`,
# so the core's `kron(conj(L), L)` monitored-jump superoperator is consistent.

# Coerce any operator backing array into the sparse ComplexF64 form the core expects.
_qt_sp(A::AbstractArray) = SparseMatrixCSC{ComplexF64, Int}(sparse(A))

function QuantumFCS.fcscumulants_recursive(
    H::QuantumObject{<:Operator},
    J::AbstractVector{<:QuantumObject{<:Operator}},
    mJ::AbstractVector{<:QuantumObject{<:Operator}},
    nC::Integer,
    rho_ss::QuantumObject{<:Operator},
    nu::AbstractVector{<:Real},
)
    L = _qt_sp(liouvillian(H, J).data)
    mJ_mats = SparseMatrixCSC{ComplexF64, Int}[_qt_sp(m.data) for m in mJ]
    return QuantumFCS.fcscumulants_recursive(L, mJ_mats, nC, _qt_sp(rho_ss.data), nu)
end

# Convenience wrapper (H, J, ...) mirroring the QuantumOptics backend.
function QuantumFCS.drazin(
    H::QuantumObject{<:Operator},
    J,
    vrho_ss::AbstractVector,
    vId::AbstractVecOrMat,
    IdL::AbstractMatrix,
)
    L = liouvillian(H, J).data
    l = length(vrho_ss)
    IdL_eff = (size(IdL, 1) == l && size(IdL, 2) == l) ? IdL : Matrix{eltype(L)}(I, l, l)
    vId_row = (size(vId, 1) == 1) ? vId : (collect(vec(vId))')
    return QuantumFCS.drazin(L, vrho_ss, vId_row, IdL_eff)
end

# Compatibility wrapper for tests using QuantumToolbox: (H, J, ...) signature
function QuantumFCS.drazin_apply(
    H::QuantumObject{<:Operator},
    J,
    alphavec::AbstractVector,
    vrho_ss::AbstractVector,
    vId::AbstractVecOrMat;
    tol::Real = 1e-8,
    maxiter::Integer = 10_000,
)
    L = _qt_sp(liouvillian(H, J).data)
    αs = SparseVector(alphavec)
    ρs = SparseVector(vrho_ss)
    vId_vec = collect(vec(vId))
    return QuantumFCS.drazin_apply(L, αs, ρs, vId_vec)
end
