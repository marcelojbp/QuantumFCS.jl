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
* `method`: Drazin-solve backend, `:lu` (default) or `:iterative`.
* `σ`, `τ`, `rtol`, `itmax`, `memory`: options for the `:iterative` backend
  (diagonal shift, ILU drop tolerance, Krylov tolerance, iteration cap, and GMRES
  restart memory). Ignored by `:lu`. `σ=nothing` auto-scales the shift from `L`.
* `Pl`: an externally built preconditioner for the `:iterative` backend to reuse
  instead of building its own ILU (e.g. the ILU from the steady-state solve). See
  [`prepare_drazin_solver`](@ref) for the contract it must satisfy. `nothing`
  (default) builds the internal preconditioner; ignored by `:lu`.

Solve with [`fcscumulants_recursive`](@ref):

    p = LindbladFCS(; H=H, J=J, mJ=[Jc], rho_ss=ρss, nu=[1], nC=2)
    c1, c2 = fcscumulants_recursive(p)

    # large sparse Liouvillian (needs `using Krylov, IncompleteLU`):
    p = LindbladFCS(; L=L, mJ=[Jc], rho_ss=ρss, nu=[1], nC=3, method=:iterative)
"""
@kwdef struct LindbladFCS{TH,TJ,TL,TmJ,Tρ,Tν,TPl} <: FCSProblem
    H::TH      = nothing
    J::TJ      = nothing
    L::TL      = nothing
    mJ::TmJ
    rho_ss::Tρ
    nu::Tν
    nC::Int    = 2
    method::Symbol             = :lu
    σ::Union{Nothing,Float64}  = nothing
    τ::Float64                 = 0.05
    Pl::TPl                    = nothing
    rtol::Float64              = 1e-8
    itmax::Int                 = 200
    memory::Int                = 30

    function LindbladFCS{TH,TJ,TL,TmJ,Tρ,Tν,TPl}(H, J, L, mJ, rho_ss, nu, nC,
                                                 method, σ, τ, Pl, rtol, itmax, memory) where {TH,TJ,TL,TmJ,Tρ,Tν,TPl}
        if L === nothing && (H === nothing || J === nothing)
            throw(ArgumentError("LindbladFCS requires either `L`, or both `H` and `J`."))
        end
        if length(mJ) != length(nu)
            throw(ArgumentError("Length of mJ ($(length(mJ))) must match length of nu ($(length(nu)))."))
        end
        return new{TH,TJ,TL,TmJ,Tρ,Tν,TPl}(H, J, L, mJ, rho_ss, nu, nC,
                                           method, σ, τ, Pl, rtol, itmax, memory)
    end
end

# Non-parametric forwarding constructor: infers the type parameters from the
# arguments. This is what the `@kwdef`-generated keyword constructor calls.
function LindbladFCS(H::TH, J::TJ, L::TL, mJ::TmJ, rho_ss::Tρ, nu::Tν, nC::Integer,
                     method::Symbol, σ::Union{Nothing,Float64}, τ::Real, Pl::TPl,
                     rtol::Real, itmax::Integer, memory::Integer) where {TH,TJ,TL,TmJ,Tρ,Tν,TPl}
    return LindbladFCS{TH,TJ,TL,TmJ,Tρ,Tν,TPl}(H, J, L, mJ, rho_ss, nu, nC,
                                               method, σ, Float64(τ), Pl, Float64(rtol),
                                               Int(itmax), Int(memory))
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
_liouvillian_data(p::LindbladFCS{<:Any, <:Any, Nothing}) = _build_liouvillian(p.H, p.J)
_liouvillian_data(p::LindbladFCS) = p.L

# No backend loaded: building L from H/J is impossible. Fail with a clear message.
_build_liouvillian(H, J) = throw(
    ArgumentError(
        "Cannot build a Liouvillian from `H` and `J`: no backend extension is loaded. " *
            "Load QuantumOptics or QuantumToolbox, or construct the problem with `L` directly."
    )
)

# Plain arrays pass through unchanged; backends override for their operator types.
_operator_data(x) = x
_state_data(x) = x

# Drazin-solver options carried by a problem. The generic fallback keeps the
# established direct-LU behavior; `LindbladFCS` overrides it with its own fields.
_solver_opts(::FCSProblem) = (method = :lu, σ = nothing, τ = 0.05, Pl = nothing,
                              rtol = 1e-8, itmax = 200, memory = 30)
_solver_opts(p::LindbladFCS) = (method = p.method, σ = p.σ, τ = p.τ, Pl = p.Pl,
                                rtol = p.rtol, itmax = p.itmax, memory = p.memory)

"""
    fcscumulants_recursive(problem::FCSProblem)

Solve a full-counting-statistics `problem`, returning its first `problem.nC`
zero-frequency cumulants. Equivalent to calling the positional method with the
problem's fields, after normalizing any backend operators to their underlying
matrices.
"""
function fcscumulants_recursive(p::FCSProblem)
    L = _liouvillian_data(p)
    mJ = map(_operator_data, p.mJ)
    ρ  = _state_data(p.rho_ss)
    o  = _solver_opts(p)
    return fcscumulants_recursive(L, mJ, p.nC, ρ, p.nu;
        method = o.method, σ = o.σ, τ = o.τ, Pl = o.Pl,
        rtol = o.rtol, itmax = o.itmax, memory = o.memory)
end

# ============================================================================
#  Prepared FCS context — "prepare once, evaluate many observables"
# ============================================================================
#
# The Drazin solver depends only on the Liouvillian and steady state, never on the
# monitored jumps or weights. When several observables share the same `L` and
# `rho_ss` (hot/cold currents, several channels, different weight choices, ...),
# preparing the solver once and reusing it avoids repeating the expensive
# factorization / preconditioner build. A `PreparedLindbladFCS` holds that
# invariant data; each observable is then evaluated with a lightweight call.

"""
    PreparedLindbladFCS

A reusable full-counting-statistics *context*: the observable-independent data for
a fixed Liouvillian and steady state, with a Drazin solver prepared **once**.

Build it with [`prepare_fcs_context`](@ref) and evaluate any number of observables
that share the same `L` and `rho_ss` with

    fcscumulants_recursive(ctx; mJ, nu, nC=2)

Because the monitored jumps `mJ` and weights `nu` enter only the counting-field
super-operators and the recursion's right-hand sides — never the Drazin linear
algebra — a single prepared solver serves every observable, so the expensive
preparation (sparse LU factorization, or shifted-ILU build for the iterative
backend) is done only once.

This complements [`LindbladFCS`](@ref): use `LindbladFCS` for a single observable,
and a `PreparedLindbladFCS` context when many observables share one Liouvillian and
steady state.

# Fields
* `L`: the (sparse `ComplexF64`) vectorized Liouvillian.
* `rho_ss`: the (sparse `ComplexF64`) steady-state density matrix, kept for
  introspection and dimension checks.
* `vrho_ss`: the normalized, vectorized steady state (right null vector).
* `vId`: the vectorized identity / trace functional (left null vector).
* `solver`: the prepared [`DrazinSolver`](@ref) reused across observables.
"""
struct PreparedLindbladFCS{TL, Tρ, Tvρ, TvId, TS}
    L::TL
    rho_ss::Tρ
    vrho_ss::Tvρ
    vId::TvId
    solver::TS
end

"""
    prepare_fcs_context(; H=nothing, J=nothing, L=nothing, rho_ss,
                        method=:lu, σ=nothing, τ=0.05, Pl=nothing,
                        rtol=1e-8, itmax=200, memory=30) -> PreparedLindbladFCS

Prepare a reusable [`PreparedLindbladFCS`](@ref) context for computing many
full-counting-statistics observables that share the same Liouvillian and steady
state.

Supply either a (vectorized) Liouvillian `L`, or both a Hamiltonian `H` and jump
operators `J` (built into `L` by the active backend). `L`/`H`/`J`/`rho_ss` may be
plain dense/sparse `ComplexF64` arrays or backend operators
(`QuantumOptics.Operator`, `QuantumToolbox.QuantumObject`); the relevant package
extension extracts the underlying matrices. The Drazin solver is prepared once here
and then reused by every call to `fcscumulants_recursive(ctx; mJ, nu, nC)`.

The solver-backend keywords are identical to those of [`LindbladFCS`](@ref) /
[`fcscumulants_recursive`](@ref) and configure this one-time preparation:

* `method`: `:lu` (default) or `:iterative` (needs the `QuantumFCSIterativeExt`
  extension, `using Krylov, IncompleteLU`).
* `σ`, `τ`, `Pl`, `rtol`, `itmax`, `memory`: options for the `:iterative` backend,
  forwarded to [`prepare_drazin_solver`](@ref); ignored by `:lu`. `Pl` supplies an
  externally built preconditioner for the iterative backend to reuse.

# Example
```julia
ctx  = prepare_fcs_context(; L = L, rho_ss = ρss, method = :lu)   # prepares LU once
hot  = fcscumulants_recursive(ctx; mJ = mJ_hot,  nu = nu_hot,  nC = 2)
cold = fcscumulants_recursive(ctx; mJ = mJ_cold, nu = nu_cold, nC = 2)
```
"""
function prepare_fcs_context(;
        H = nothing, J = nothing, L = nothing, rho_ss,
        method::Symbol = :lu,
        σ::Union{Nothing, Float64} = nothing,
        τ::Float64 = 0.05,
        Pl = nothing,
        rtol::Float64 = 1e-8,
        itmax::Int = 200,
        memory::Int = 30,
    )
    if L === nothing && (H === nothing || J === nothing)
        throw(ArgumentError("prepare_fcs_context requires either `L`, or both `H` and `J`."))
    end

    # Normalize to the sparse ComplexF64 forms the core engine expects. A dense L
    # is sparsified so a single prepared-solver path serves both cases (reuse
    # matters for the larger, sparse problems anyway).
    Ldata = L === nothing ? _build_liouvillian(H, J) : L
    Lsp = SparseMatrixCSC{ComplexF64, Int}(Ldata)
    ρsp = SparseMatrixCSC{ComplexF64, Int}(_state_data(rho_ss))

    n = size(ρsp, 1)              # matrix side
    l = n * n                     # vectorized length
    size(Lsp, 1) == l && size(Lsp, 2) == l || throw(
        DimensionMismatch(
            "Liouvillian is $(size(Lsp)); expected ($l, $l) for a $n×$n steady state."
        )
    )

    # Vectorized identity (diagonal entries of an n×n identity under vec) and the
    # normalized, vectorized steady state — the two null vectors of `L`.
    diag_idx = collect(1:(n + 1):l)
    vId = SparseVector{ComplexF64, Int}(l, diag_idx, fill(1.0 + 0.0im, n))
    vrho_ss = SparseVector(vec(ρsp ./ tr(ρsp)))

    solver = prepare_drazin_solver(Lsp, vrho_ss, vId;
        method = method,
        rtol = (method === :lu ? 1e-12 : rtol),
        σ = σ, τ = τ, Pl = Pl, itmax = itmax, memory = memory)

    return PreparedLindbladFCS(Lsp, ρsp, vrho_ss, vId, solver)
end

"""
    fcscumulants_recursive(ctx::PreparedLindbladFCS; mJ, nu, nC=2, cumulant_type="")

Evaluate a full-counting-statistics observable on a prepared context, reusing its
Drazin solver. Returns the first `nC` zero-frequency cumulants.

`mJ` are the monitored jump operators/matrices (backend operators are accepted and
normalized) and `nu` their weights (`length(nu) == length(mJ)`). Each jump must be
`n×n`, where `n` is the side of the context's steady state. See
[`prepare_fcs_context`](@ref) for the "prepare once, evaluate many observables"
workflow and [`fcscumulants_recursive`](@ref) for `cumulant_type`.
"""
function fcscumulants_recursive(ctx::PreparedLindbladFCS;
        mJ, nu, nC::Integer = 2, cumulant_type::AbstractString = "")
    if length(mJ) != length(nu)
        throw(ArgumentError("Length of mJ ($(length(mJ))) must match length of nu ($(length(nu)))."))
    end
    n = size(ctx.rho_ss, 1)
    mJd = SparseMatrixCSC{ComplexF64, Int}[
        SparseMatrixCSC{ComplexF64, Int}(_operator_data(m)) for m in mJ
    ]
    for (k, m) in enumerate(mJd)
        size(m) == (n, n) || throw(
            DimensionMismatch(
                "mJ[$k] has size $(size(m)); expected ($n, $n) to match the prepared " *
                    "context (steady state is $n×$n)."
            )
        )
    end
    return _fcscumulants_recursive_prepared(ctx.solver, mJd, nC, ctx.vrho_ss, ctx.vId, nu;
        cumulant_type = cumulant_type)
end
