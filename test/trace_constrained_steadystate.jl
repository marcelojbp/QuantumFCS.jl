using QuantumFCS
using QuantumOptics
using LinearAlgebra
using SparseArrays
using Krylov          # together with IncompleteLU, activates QuantumFCSIterativeExt
using IncompleteLU
using Test

# --- Reference helpers -------------------------------------------------------
# Self-contained replications of the *exact* application formulas
# (FCS.jl-application/src/{driven_dissipative_jc,nonlinear_model}.jl). The new
# package API must reproduce these bit-for-bit, so we build the reference here
# rather than importing the read-only application code.

function ref_trace_constrained(Ldata::SparseMatrixCSC{ComplexF64, Int}, n::Integer)
    weight = norm(Ldata, 1) / length(Ldata)         # entrywise 1-norm / n^4
    b = zeros(ComplexF64, n^2)
    b[1] = weight
    idx = collect(1:n)
    rows = ones(Int, n)
    cols = n .* (idx .- 1) .+ idx                    # == 1:(n+1):n^2 (column-major vec diag)
    vals = fill(ComplexF64(weight), n)
    tc = sparse(rows, cols, vals, n^2, n^2)
    A = Ldata + tc
    return (; A, b, weight)
end

function ref_shift(A::SparseMatrixCSC{ComplexF64, Int}; shift_factor = 1.0e-6)
    scale = norm(A, 1) / size(A, 1)
    return shift_factor * max(real(scale), eps(Float64))
end

@testset "Trace-constrained steady state" begin
    @test Base.get_extension(QuantumFCS, :QuantumFCSIterativeExt) !== nothing

    # Quantum-dot heat engine: small, analytic cumulants (matches fcsproblem.jl).
    b1 = FockBasis(1)
    d = destroy(b1)
    d_dag = create(b1)
    ϵd, κc, κh = 1.0, 0.1, 0.5
    Hdot = ϵd * d_dag * d
    Jcloss = sqrt(κc) * d
    Jhgain = sqrt(κh) * d_dag
    Jdot = [Jcloss, Jhgain]
    ρss_dot = steadystate.eigenvector(Hdot, Jdot)
    Ldot = SparseMatrixCSC{ComplexF64, Int}(liouvillian(Hdot, Jdot).data)
    ndot = size(ρss_dot.data, 1)

    c1_analytical = κc * κh / (κc + κh)
    c2_analytical = (κh^2 + κc^2) / (κc + κh)^2 * c1_analytical

    # Driven, damped cavity: bigger sparse L, meaningful iterative test.
    bc = FockBasis(6)
    a = destroy(bc)
    Hcav = 0.5 * (a' * a) + 1.0 * (a + a')
    κ = 1.0
    Jcav = [sqrt(κ) * a]
    ρss_cav = steadystate.eigenvector(Hcav, Jcav)
    Lcav = SparseMatrixCSC{ComplexF64, Int}(liouvillian(Hcav, Jcav).data)
    ncav = size(ρss_cav.data, 1)

    @testset "system-build equivalence (matches application formula)" begin
        sys = trace_constrained_system(Lcav)
        r = ref_trace_constrained(Lcav, ncav)
        @test sys.A == r.A
        @test sys.b == r.b

        # vId is the column-major vectorized identity (diagonal indices).
        vId_ref = SparseVector{ComplexF64, Int}(
            ncav^2, collect(1:(ncav + 1):(ncav^2)),
            fill(1.0 + 0.0im, ncav)
        )
        @test sys.vId == vId_ref

        # `weight` override is honored.
        sys2 = trace_constrained_system(Lcav; weight = 0.5)
        @test sys2.b[1] == ComplexF64(0.5)

        # Build from H, J agrees with build from L.
        sysHJ = trace_constrained_system(Hcav, Jcav)
        @test sysHJ.A == r.A
        @test sysHJ.b == r.b
    end

    @testset "shifted_ilu_preconditioner (matches application formula)" begin
        sys = trace_constrained_system(Lcav)
        P = shifted_ilu_preconditioner(sys.A)                 # defaults τ=1e-3, shift_factor=1e-6
        shift = ref_shift(sys.A)
        Pref = IncompleteLU.ilu(sys.A + shift * I; τ = 1.0e-3)

        # Same incomplete factors ⇒ same preconditioner action.
        x = randn(ComplexF64, ncav^2)
        @test P \ x ≈ Pref \ x rtol = 1.0e-12
        @test nnz(P.L) == nnz(Pref.L)
        @test nnz(P.U) == nnz(Pref.U)

        # Explicit shift is honored.
        Pshift = shifted_ilu_preconditioner(sys.A; shift = 1.0e-4)
        Pshift_ref = IncompleteLU.ilu(sys.A + 1.0e-4 * I; τ = 1.0e-3)
        @test Pshift \ x ≈ Pshift_ref \ x rtol = 1.0e-12
    end

    @testset "method=:lu steady state matches eigenvector" begin
        ss = trace_constrained_steadystate(Ldot; method = :lu)
        @test ss isa TraceConstrainedSteadyState
        @test ss.rho_ss isa SparseMatrixCSC{ComplexF64, Int}

        ρ = Matrix(ss.rho_ss)
        @test tr(ρ) ≈ 1 atol = 1.0e-12
        @test norm(ρ - ρ') < 1.0e-12                            # hermitian
        @test ρ ≈ Matrix(ρss_dot.data) rtol = 1.0e-8           # physical steady state
        @test norm(Ldot * vec(ρ)) < 1.0e-9                      # in the kernel of L

        @test ss.stats.converged
        @test ss.stats.trace_error < 1.0e-10
        @test ss.stats.hermiticity_error < 1.0e-10
    end

    @testset "method=:iterative steady state + reusable Pl" begin
        ss = trace_constrained_steadystate(Lcav; method = :iterative)
        @test ss.Pl !== nothing
        @test ss.stats.converged

        ρ = Matrix(ss.rho_ss)
        @test tr(ρ) ≈ 1 atol = 1.0e-8
        @test norm(ρ - ρ') < 1.0e-10
        @test ρ ≈ Matrix(ρss_cav.data) rtol = 1.0e-6
        @test norm(Lcav * vec(ρ)) < 1.0e-7

        # Convenience H, J signature agrees.
        ss_hj = trace_constrained_steadystate(Hcav, Jcav; method = :iterative)
        @test Matrix(ss_hj.rho_ss) ≈ ρ rtol = 1.0e-6
    end

    @testset "FCS bridge reuses Pl without rebuilding the ILU" begin
        ss = trace_constrained_steadystate(Lcav; method = :iterative)
        ctx = prepare_fcs_context(ss; method = :iterative)
        @test ctx isa PreparedLindbladFCS
        # The injected preconditioner is the *same object* — no second ILU build.
        @test ctx.solver.P === ss.Pl

        # Cumulants match the direct LU reference for the same steady state.
        mJ = [sparse((sqrt(κ) * a).data)]
        c_it = fcscumulants_recursive(ctx; mJ = mJ, nu = [1.0], nC = 3)
        c_lu = fcscumulants_recursive(
            LindbladFCS(
                Hcav, Jcav; mJ = [sqrt(κ) * a], rho_ss = ρss_cav, nu = [1.0],
                nC = 3, method = :lu
            )
        )
        @test c_it ≈ c_lu rtol = 1.0e-5

        # method=:lu bridge ignores Pl and still works.
        ss_lu = trace_constrained_steadystate(Ldot; method = :lu)
        ctx_lu = prepare_fcs_context(ss_lu; method = :lu)
        c = fcscumulants_recursive(ctx_lu; mJ = [sparse(Jcloss.data)], nu = [1], nC = 2)
        @test c ≈ [c1_analytical, c2_analytical] atol = 1.0e-10
    end

    @testset "low-level system overload supports warm start (continuation)" begin
        sys = trace_constrained_system(Lcav)
        Pl = shifted_ilu_preconditioner(sys.A)

        ss_cold = trace_constrained_steadystate(sys; Pl = Pl)
        @test ss_cold.stats.converged
        @test ss_cold.Pl === Pl                               # reuses the supplied Pl

        # Warm-start from a nearby solution: same answer, fewer iterations expected.
        u0 = vec(Matrix(ss_cold.rho_ss))
        ss_warm = trace_constrained_steadystate(sys; Pl = Pl, u0 = u0)
        @test ss_warm.stats.converged
        @test Matrix(ss_warm.rho_ss) ≈ Matrix(ss_cold.rho_ss) rtol = 1.0e-6
    end

    @testset "backend acceptance and validation" begin
        # Prebuilt sparse L and QuantumOptics H, J both accepted for :lu.
        ss_L = trace_constrained_steadystate(Ldot; method = :lu)
        ss_HJ = trace_constrained_steadystate(Hdot, Jdot; method = :lu)
        @test Matrix(ss_L.rho_ss) ≈ Matrix(ss_HJ.rho_ss) rtol = 1.0e-8

        # Non-square (bad) Liouvillian dimension.
        Lbad = sparse(ComplexF64.(reshape(1:12, 3, 4)))
        @test_throws DimensionMismatch trace_constrained_system(Lbad)

        # Unknown method.
        @test_throws ArgumentError trace_constrained_steadystate(Ldot; method = :bogus)
    end
end
