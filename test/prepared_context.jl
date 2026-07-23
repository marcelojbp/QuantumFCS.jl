using QuantumFCS
using QuantumOptics
using LinearAlgebra
using SparseArrays
using Krylov          # together with IncompleteLU, activates QuantumFCSIterativeExt
using IncompleteLU
using Test

@testset "Prepared FCS context" begin
    # Quantum-dot heat engine (same system/analytics as fcsproblem.jl). A single
    # Liouvillian and steady state serve several monitored observables — exactly the
    # case `prepare_fcs_context` targets.
    b = FockBasis(1)
    d = destroy(b)
    d_dag = create(b)
    ϵd, κc, κh = 1.0, 0.1, 0.5
    H = ϵd * d_dag * d
    Jcloss = sqrt(κc) * d              # jumps into the cold reservoir
    Jhgain = sqrt(κh) * d_dag          # jumps from the hot reservoir
    J = [Jcloss, Jhgain]
    ρss = steadystate.eigenvector(H, J)

    c1_analytical = κc * κh / (κc + κh)
    c2_analytical = (κh^2 + κc^2) / (κc + κh)^2 * c1_analytical

    @testset "prepared-LU equivalence (backend operators)" begin
        ctx = prepare_fcs_context(; H = H, J = J, rho_ss = ρss, method = :lu)
        @test ctx isa PreparedLindbladFCS
        @test ctx.solver isa QuantumFCS.DrazinSolver

        c1, c2 = fcscumulants_recursive(ctx; mJ = [Jcloss], nu = [1], nC = 2)
        @test c1 ≈ c1_analytical atol = 1.0e-10
        @test c2 ≈ c2_analytical atol = 1.0e-10

        # Identical to the ordinary single-observable LU path.
        ref = fcscumulants_recursive(
            LindbladFCS(H, J; mJ = [Jcloss], rho_ss = ρss, nu = [1], nC = 2, method = :lu)
        )
        @test [c1, c2] ≈ ref atol = 1.0e-12
    end

    @testset "prepared LU from a prebuilt Liouvillian" begin
        L = SparseMatrixCSC{ComplexF64, Int}(liouvillian(H, J).data)
        ρm = sparse(ρss.data)
        ctx = prepare_fcs_context(; L = L, rho_ss = ρm, method = :lu)
        c = fcscumulants_recursive(ctx; mJ = [sparse(Jcloss.data)], nu = [1], nC = 2)
        @test c ≈ [c1_analytical, c2_analytical] atol = 1.0e-10
    end

    @testset "a dense Liouvillian is accepted (sparsified internally)" begin
        ctx = prepare_fcs_context(;
            L = Matrix(liouvillian(H, J).data), rho_ss = Matrix(ρss.data), method = :lu
        )
        c = fcscumulants_recursive(ctx; mJ = [sparse(Jcloss.data)], nu = [1], nC = 2)
        @test c ≈ [c1_analytical, c2_analytical] atol = 1.0e-10
    end

    @testset "reuse across observables (one prepared solver)" begin
        ctx = prepare_fcs_context(; H = H, J = J, rho_ss = ρss, method = :lu)

        # Cold current and its noise vs. an independent high-level call.
        cold = fcscumulants_recursive(ctx; mJ = [Jcloss], nu = [1], nC = 2)
        cold_ref = fcscumulants_recursive(
            LindbladFCS(H, J; mJ = [Jcloss], rho_ss = ρss, nu = [1], nC = 2)
        )
        @test cold ≈ cold_ref atol = 1.0e-12

        # A different monitored observable (hot channel) on the *same* context.
        hot = fcscumulants_recursive(ctx; mJ = [Jhgain], nu = [1], nC = 2)
        hot_ref = fcscumulants_recursive(
            LindbladFCS(H, J; mJ = [Jhgain], rho_ss = ρss, nu = [1], nC = 2)
        )
        @test hot ≈ hot_ref atol = 1.0e-12

        # The very same prepared solver object backed both evaluations.
        s = ctx.solver
        fcscumulants_recursive(ctx; mJ = [Jhgain], nu = [1], nC = 2)
        @test ctx.solver === s
    end

    @testset "cumulant_type passthrough" begin
        ctx = prepare_fcs_context(; H = H, J = J, rho_ss = ρss, method = :lu)
        ordinary = fcscumulants_recursive(ctx; mJ = [Jcloss], nu = [1], nC = 3)
        factorial = fcscumulants_recursive(
            ctx; mJ = [Jcloss], nu = [1], nC = 3,
            cumulant_type = "factorial"
        )
        @test factorial ≈ factorial_cumulants(ordinary) atol = 1.0e-12
    end

    @testset "prepared iterative equivalence" begin
        @test Base.get_extension(QuantumFCS, :QuantumFCSIterativeExt) !== nothing

        ref = fcscumulants_recursive(
            LindbladFCS(H, J; mJ = [Jcloss], rho_ss = ρss, nu = [1], nC = 3, method = :lu)
        )

        ctx = prepare_fcs_context(; H = H, J = J, rho_ss = ρss, method = :iterative)
        @test ctx.solver isa QuantumFCS.DrazinSolver
        c_it = fcscumulants_recursive(ctx; mJ = [Jcloss], nu = [1], nC = 3)
        @test c_it ≈ ref rtol = 1.0e-5

        # Injected preconditioner (Pl) is forwarded to the iterative preparation.
        L = SparseMatrixCSC{ComplexF64, Int}(liouvillian(H, J).data)
        σ = 0.01 * maximum(abs, nonzeros(L))
        Pl = IncompleteLU.ilu(L - σ * I; τ = 0.01)
        ctx_pl = prepare_fcs_context(;
            L = L, rho_ss = sparse(ρss.data),
            method = :iterative, Pl = Pl
        )
        c_pl = fcscumulants_recursive(ctx_pl; mJ = [sparse(Jcloss.data)], nu = [1], nC = 3)
        @test c_pl ≈ ref rtol = 1.0e-5
    end

    @testset "validation" begin
        ctx = prepare_fcs_context(; H = H, J = J, rho_ss = ρss, method = :lu)

        # Mismatched mJ / nu lengths.
        @test_throws ArgumentError fcscumulants_recursive(ctx; mJ = [Jcloss], nu = [1, 2])

        # A monitored jump of the wrong Hilbert-space dimension.
        big = destroy(FockBasis(3))
        @test_throws DimensionMismatch fcscumulants_recursive(ctx; mJ = [big], nu = [1])

        # Neither L nor (H and J) supplied.
        @test_throws ArgumentError prepare_fcs_context(; rho_ss = ρss)

        # Liouvillian inconsistent with the steady-state dimension.
        L_wrong = SparseMatrixCSC{ComplexF64, Int}(
            liouvillian(
                destroy(FockBasis(3)),
                [destroy(FockBasis(3))]
            ).data
        )
        @test_throws DimensionMismatch prepare_fcs_context(; L = L_wrong, rho_ss = sparse(ρss.data))
    end
end
