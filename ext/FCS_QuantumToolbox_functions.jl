# Backend bindings for QuantumToolbox.jl.
#
# QuantumToolbox represents every quantum object as a `QuantumObject`, with the
# object kind (operator, ket, super-operator, ...) stored as a *singleton value*
# in a type parameter — e.g. `Operator === OperatorQuantumObject()` is a value,
# not a type, so `QuantumObject{<:Operator}` is invalid. We therefore dispatch on
# `QuantumObject` directly and peel the backing array off the `.data` field,
# mirroring the QuantumOptics backend.
#
# Vectorization convention matches the core engine and QuantumOptics:
# `mat2vec == vec` (column-major), `spre(A) = kron(I, A)`, `spost(B) = kron(transpose(B), I)`,
# so the core's `kron(conj(L), L)` monitored-jump superoperator is consistent.

# Coerce any operator backing array into the sparse ComplexF64 form the core expects.
_qt_sp(A::AbstractArray) = SparseMatrixCSC{ComplexF64, Int}(sparse(A))

# --- Positional (H, J, mJ, nC, rho_ss, nu) API, mirroring the QuantumOptics backend ---
function QuantumFCS.fcscumulants_recursive(
    H::QuantumObject,
    J::AbstractVector{<:QuantumObject},
    mJ::AbstractVector{<:QuantumObject},
    nC::Integer,
    rho_ss::QuantumObject,
    nu::AbstractVector{<:Real};
    kwargs...,
)
    L = _qt_sp(liouvillian(H, J).data)
    mJ_mats = SparseMatrixCSC{ComplexF64, Int}[_qt_sp(m.data) for m in mJ]
    return QuantumFCS.fcscumulants_recursive(L, mJ_mats, nC, _qt_sp(rho_ss.data), nu; kwargs...)
end

# Convenience wrapper (H, J, ...) mirroring the QuantumOptics backend.
function QuantumFCS.drazin(
    H::QuantumObject,
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
    H::QuantumObject,
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

# --- FCSProblem backend hooks (QuantumToolbox) ---
#
# These let a `LindbladFCS` problem hold QuantumToolbox operators directly and be
# solved with the single-argument `fcscumulants_recursive(problem)`. We force
# sparse `ComplexF64` matrices so the results match the types expected by the core
# positional `fcscumulants_recursive`.
QuantumFCS._operator_data(x::QuantumObject) = _qt_sp(x.data)
QuantumFCS._state_data(x::QuantumObject) = _qt_sp(x.data)
QuantumFCS._build_liouvillian(H::QuantumObject, J) = _qt_sp(liouvillian(H, J).data)

# Convenience constructor: build a problem from H and J, deferring L to solve time.
function QuantumFCS.LindbladFCS(H::QuantumObject, J::AbstractVector{<:QuantumObject};
                                mJ, rho_ss, nu, nC::Integer = 2, kwargs...)
    return QuantumFCS.LindbladFCS(; H = H, J = J, mJ = mJ, rho_ss = rho_ss,
                                  nu = nu, nC = nC, kwargs...)
end
