# --- FCSProblem backend hooks (QuantumToolbox) ---
#
# QuantumToolbox stores the underlying matrix in the `.data` field of a
# `QuantumObject`, and provides `liouvillian(H, c_ops)`. We mirror the
# QuantumOptics hooks, forcing sparse matrices so the results match the types
# expected by the core positional `fcscumulants_recursive`.
QuantumFCS._operator_data(x::QuantumObject) = sparse(x.data)
QuantumFCS._state_data(x::QuantumObject) = sparse(x.data)
QuantumFCS._build_liouvillian(H::QuantumObject, J) = sparse(liouvillian(H, J).data)

# Convenience constructor: build a problem from H and J, deferring L to solve time.
function QuantumFCS.LindbladFCS(H::QuantumObject, J::AbstractVector{<:QuantumObject};
                                mJ, rho_ss, nu, nC::Integer = 2)
    return QuantumFCS.LindbladFCS(; H = H, J = J, mJ = mJ, rho_ss = rho_ss, nu = nu, nC = nC)
end
