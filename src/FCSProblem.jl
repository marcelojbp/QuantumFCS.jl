"""
    FCSProblem

Abstract supertype for full-counting-statistics problems. A concrete subtype
encapsulates everything `fcscumulants_recursive` needs — the Liouvillian (or the
Hamiltonian/jump operators it is built from), the monitored jumps, the steady
state, the weights, and the number of cumulants — so that a simulation can be
organised as a single object and solved with `fcscumulants_recursive(problem)`.
"""
abstract type FCSProblem end

"""
    LindbladFCS(; H=nothing, J=nothing, L=nothing, mJ, rho_ss, nu, nC=2)

A full-counting-statistics problem for a Lindblad master equation.

Construct it by keyword. Either supply a (vectorized) Liouvillian `L`, or both a
Hamiltonian `H` and a vector of jump operators `J` (in which case `L` is built
from them at solve time by the active backend). Fields may be plain dense/sparse
`ComplexF64` arrays, or backend operators (`QuantumOptics.Operator`,
`QuantumToolbox.QuantumObject`); the relevant package extension takes care of
extracting the underlying matrices.

# Fields
* `H`: optional Hamiltonian operator (backend type) used to build `L`.
* `J`: optional vector of jump operators (backend type) used to build `L`.
* `L`: optional (vectorized) Liouvillian. Takes precedence over `H`/`J`.
* `mJ`: vector of monitored jump operators/matrices.
* `rho_ss`: steady-state density matrix.
* `nu`: weights, one per monitored jump (`length(nu) == length(mJ)`).
* `nC`: number of cumulants to compute (default `2`).

Solve with [`fcscumulants_recursive`](@ref):

    p = LindbladFCS(; H=H, J=J, mJ=[Jc], rho_ss=ρss, nu=[1], nC=2)
    c1, c2 = fcscumulants_recursive(p)
"""
@kwdef struct LindbladFCS{TH,TJ,TL,TmJ,Tρ,Tν} <: FCSProblem
    H::TH      = nothing
    J::TJ      = nothing
    L::TL      = nothing
    mJ::TmJ
    rho_ss::Tρ
    nu::Tν
    nC::Int    = 2

    function LindbladFCS{TH,TJ,TL,TmJ,Tρ,Tν}(H, J, L, mJ, rho_ss, nu, nC) where {TH,TJ,TL,TmJ,Tρ,Tν}
        if L === nothing && (H === nothing || J === nothing)
            throw(ArgumentError("LindbladFCS requires either `L`, or both `H` and `J`."))
        end
        if length(mJ) != length(nu)
            throw(ArgumentError("Length of mJ ($(length(mJ))) must match length of nu ($(length(nu)))."))
        end
        return new{TH,TJ,TL,TmJ,Tρ,Tν}(H, J, L, mJ, rho_ss, nu, nC)
    end
end

# Non-parametric forwarding constructor: infers the type parameters from the
# arguments. This is what the `@kwdef`-generated keyword constructor calls.
function LindbladFCS(H::TH, J::TJ, L::TL, mJ::TmJ, rho_ss::Tρ, nu::Tν, nC::Integer) where {TH,TJ,TL,TmJ,Tρ,Tν}
    return LindbladFCS{TH,TJ,TL,TmJ,Tρ,Tν}(H, J, L, mJ, rho_ss, nu, nC)
end

# --- Backend-agnostic data extraction -------------------------------------
#
# These helpers normalize a problem's fields to the plain-array types the
# positional `fcscumulants_recursive` expects. Dispatch (not runtime `=== nothing`
# checks) selects the branch, so the return type is inferable and the solve path
# stays type-stable. Backends add methods for their operator types in extensions.

# Liouvillian: dispatch on the third type parameter (`TL`). When `L` is stored
# (`TL !== Nothing`) use it directly; when it is absent (`TL === Nothing`) build
# it from `H`/`J` via the active backend.
_liouvillian_data(p::LindbladFCS{<:Any,<:Any,Nothing}) = _build_liouvillian(p.H, p.J)
_liouvillian_data(p::LindbladFCS) = p.L

# No backend loaded: building L from H/J is impossible. Fail with a clear message.
_build_liouvillian(H, J) = throw(ArgumentError(
    "Cannot build a Liouvillian from `H` and `J`: no backend extension is loaded. " *
    "Load QuantumOptics or QuantumToolbox, or construct the problem with `L` directly."))

# Plain arrays pass through unchanged; backends override for their operator types.
_operator_data(x) = x
_state_data(x) = x

"""
    fcscumulants_recursive(problem::FCSProblem)

Solve a full-counting-statistics `problem`, returning its first `problem.nC`
zero-frequency cumulants. Equivalent to calling the positional method with the
problem's fields, after normalizing any backend operators to their underlying
matrices.
"""
function fcscumulants_recursive(p::FCSProblem)
    L  = _liouvillian_data(p)
    mJ = map(_operator_data, p.mJ)
    ρ  = _state_data(p.rho_ss)
    return fcscumulants_recursive(L, mJ, p.nC, ρ, p.nu)
end
