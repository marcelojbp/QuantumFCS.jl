"""
Benchmark script for fcscumulants_recursive — sparse and dense backends.

Run with:
    julia --project scripts/benchmark_fcscumulants.jl

Prints median wall-time and allocation counts for four cases:
  • 6-level system  (l = 36)  — sparse L
  • 6-level system  (l = 36)  — dense L
  • 15-level system (l = 225) — sparse L
  • 15-level system (l = 225) — dense L
"""

using QuantumFCS
using LinearAlgebra, SparseArrays, Printf

# ── helpers ──────────────────────────────────────────────────────────────────

function build_problem(n::Int; nC::Int = 4)
    # Simple single-mode Lindblad: H = ωa†a, J = [√γ a]
    # Build everything by hand so there is no dependency on QuantumOptics.
    l = n * n

    # Annihilation / creation operators for an n-level truncated Fock space
    # a|j⟩ = sqrt(j)|j-1⟩  →  a[j-1,j] = sqrt(j-1)  →  upper super-diagonal (+1)
    avals = [sqrt(Float64(k)) for k in 1:n-1]
    a_sp  = ComplexF64.(spdiagm(+1 => avals))   # upper super-diagonal (annihilation)
    a_d   = sparse(a_sp')                        # Hermitian adjoint    (creation)

    γ  = 1.0
    H  = ComplexF64.(Diagonal(0.5 .* Float64.(0:n-1)))  # ω = 0.5, H = ω a†a
    Id = ComplexF64.(sparse(I, n, n))

    # Liouvillian in vectorized form: L = -i(I⊗H - H̄⊗I) + γ(ā⊗a - ½(I⊗a†a + ā†ā⊗I))
    atda = a_d * a_sp                            # a†a  (number operator)
    L_sparse = SparseMatrixCSC{ComplexF64,Int}(
        -im * (kron(Id, H) - kron(conj.(H), Id))
        + γ  * (kron(conj.(a_sp), a_sp)
                - 0.5 * (kron(Id, atda) + kron(conj.(atda), Id)))
    )

    # Steady state: |0⟩⟨0| (vacuum state), the true steady state of this decay channel
    rho_ss_sp = spzeros(ComplexF64, n, n)
    rho_ss_sp[1,1] = 1.0

    # fcscumulants_recursive expects the raw n×n jump matrices; m_jumps builds the super-ops
    mJ_sp = [a_sp]
    nu    = [1.0]

    return L_sparse, mJ_sp, nC, rho_ss_sp, nu
end

function timed_runs(f, args...; nwarm::Int=3, nruns::Int=20)
    # Warmup
    for _ in 1:nwarm; f(args...); end
    # Timed runs
    times = Vector{Float64}(undef, nruns)
    for i in 1:nruns
        t = time_ns()
        f(args...)
        times[i] = (time_ns() - t) * 1e-6   # ms
    end
    allocs = @allocated f(args...)
    sort!(times)
    return (
        median = times[div(nruns,2)],
        min    = times[1],
        max    = times[end],
        allocs = allocs,
    )
end

# ── benchmark cases ───────────────────────────────────────────────────────────

function run_benchmarks()
    cases = [
        ("6-level  (l=36) ", 6),
        ("15-level (l=225)", 15),
    ]

    println("\n" * "="^72)
    println("  fcscumulants_recursive benchmark  (nC = 4, 20 runs each)")
    println("="^72)
    println(rpad("Case", 26), rpad("Backend", 9),
            lpad("Median(ms)", 12), lpad("Min(ms)", 10),
            lpad("Max(ms)", 10), lpad("Allocs(B)", 12))
    println("-"^72)

    for (label, n) in cases
        L_sp, mJ_sp, nC, rho_ss_sp, nu = build_problem(n; nC=4)
        L_de = Matrix(L_sp)

        r = timed_runs(fcscumulants_recursive, L_sp, mJ_sp, nC, rho_ss_sp, nu)
        println(rpad(label, 26), rpad("sparse", 9),
                lpad(@sprintf("%.3f", r.median), 12),
                lpad(@sprintf("%.3f", r.min),    10),
                lpad(@sprintf("%.3f", r.max),    10),
                lpad(string(r.allocs),            12))

        r = try
            timed_runs(fcscumulants_recursive, L_de, mJ_sp, nC, rho_ss_sp, nu)
        catch e
            (median = NaN, min = NaN, max = NaN, allocs = -1)
        end
        if isnan(r.median)
            println(rpad(label, 26), rpad("dense",  9),
                    lpad("ERROR", 12), lpad("", 10), lpad("", 10),
                    lpad("(singular — unfixed)", 20))
        else
            println(rpad(label, 26), rpad("dense",  9),
                    lpad(@sprintf("%.3f", r.median), 12),
                    lpad(@sprintf("%.3f", r.min),    10),
                    lpad(@sprintf("%.3f", r.max),    10),
                    lpad(string(r.allocs),            12))
        end
    end
    println("="^72)
end

run_benchmarks()
