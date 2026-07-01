"""
Scaling benchmark: direct-LU vs iterative Drazin backend for FCS cumulants.

Builds the vectorized Liouvillian of a driven–dissipative Jaynes–Cummings model
(cavity ⊗ two-level system) at a sequence of cavity cutoffs and times
`fcscumulants_recursive` end to end with `method = :lu` and `method = :iterative`,
reporting wall time, the per-backend speedup, and LU-vs-ILU fill-in.

This reproduces the package's headline result: the iterative backend
(matrix-free rank-1 gauge + shifted-ILU GMRES) keeps a roughly constant iteration
count and bounded fill-in while direct-LU fill explodes, so the speedup grows with
the Hilbert-space dimension.

Run from an environment that has Krylov and IncompleteLU available, e.g.

    julia --project=path/with/Krylov+IncompleteLU scripts/benchmark_iterative_scaling.jl

Sizes are kept modest by default so the run is quick and direct LU stays feasible;
raise `CUTOFFS` to see the gap widen. Direct LU past cavity cutoff ~200 takes
minutes and large memory — increase deliberately.
"""

using QuantumFCS
using Krylov, IncompleteLU          # activate the iterative extension
using LinearAlgebra, SparseArrays, Printf

# Cavity cutoffs (Fock dimension = N+1). Liouville dimension l = (2(N+1))^2.
const CUTOFFS = [40, 80, 120]

# Vectorized JC Liouvillian: cavity(dim N+1) ⊗ TLS(dim 2), cavity loss √κ a.
function jc_liouvillian(N; g = 14.0, E = 5.0, κ = 1.0)
    Nc = N + 1
    a_c = spdiagm(1 => [sqrt(Float64(k)) for k in 1:(Nc - 1)])
    Ic, I2 = sparse(I, Nc, Nc), sparse(I, 2, 2)
    a = ComplexF64.(kron(a_c, I2)); adag = sparse(a')
    sm = ComplexF64.(kron(Ic, sparse([0.0 0.0; 1.0 0.0]))); sp = sparse(sm')
    H = g * (adag * sm + a * sp) + E * (-(a + adag))
    c = sqrt(κ) * a
    n = size(H, 1); Id = sparse(I, n, n); cdc = c' * c
    L = -im * (kron(Id, H) - kron(conj(H), Id)) +
        (kron(conj(c), c) - 0.5 * (kron(Id, cdc) + kron(conj(cdc), Id)))
    return SparseMatrixCSC{ComplexF64, Int}(L), c, n
end

# Steady state via the gauge-fixed direct system (one LU solve); self-contained.
function steady_state(L, n)
    l = n * n
    diag_idx = collect(1:(n + 1):l); k0 = diag_idx[1]
    A = copy(L); A[k0, :] .= 0
    for i in diag_idx
        A[k0, i] = 1.0 + 0im
    end
    dropzeros!(A)
    b = zeros(ComplexF64, l); b[k0] = 1.0
    v = A \ b
    ρ = reshape(v, n, n)
    return sparse(ρ ./ tr(ρ))
end

bestof(f; n = 2) = minimum(((@elapsed f()) for _ in 1:n))

function run_scaling(cutoffs = CUTOFFS; nC = 3)
    println("\n" * "="^96)
    println("  FCS cumulants (nC=$nC): direct-LU vs iterative Drazin backend — driven JC model")
    println("="^96)
    @printf(
        "%-6s %-10s %-12s %-12s %-9s %-10s %-10s\n",
        "N", "l", "LU (s)", "iter (s)", "speedup", "nnz(LU)", "nnz(L)"
    )
    println("-"^96)

    for N in cutoffs
        L, _, n = jc_liouvillian(N)
        l = n * n
        ρss = steady_state(L, n)
        mJ = [SparseMatrixCSC{ComplexF64, Int}(spdiagm(1 => [sqrt(Float64(k)) for k in 1:(n - 1)]))]
        # monitored jump = cavity loss operator a (same √κ a as the dissipator)
        Nc = N + 1
        a_c = spdiagm(1 => [sqrt(Float64(k)) for k in 1:(Nc - 1)])
        a = SparseMatrixCSC{ComplexF64, Int}(kron(a_c, sparse(I, 2, 2)))
        mJ = [a]; nu = [1.0]

        # warm up (compilation) once per backend
        fcscumulants_recursive(L, mJ, nC, ρss, nu; method = :lu)
        fcscumulants_recursive(L, mJ, nC, ρss, nu; method = :iterative)

        t_lu = bestof(() -> fcscumulants_recursive(L, mJ, nC, ρss, nu; method = :lu))
        t_it = bestof(() -> fcscumulants_recursive(L, mJ, nC, ρss, nu; method = :iterative))
        F = lu(L); nnzLU = nnz(F.L) + nnz(F.U)

        @printf(
            "%-6d %-10d %-12.3f %-12.3f %-9.1f %-10d %-10d\n",
            N, l, t_lu, t_it, t_lu / t_it, nnzLU, nnz(L)
        )
    end
    return println("="^96)
end

run_scaling()
