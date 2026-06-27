@testset "Iterative Drazin backend" begin
    # Requires the QuantumFCSIterativeExt extension. It is activated transitively
    # by QuantumToolbox (loaded in runtests.jl), which pulls in Krylov + IncompleteLU.
    @test Base.get_extension(QuantumFCS, :QuantumFCSIterativeExt) !== nothing

    @testset "matches :lu on a driven damped cavity" begin
        # A driven, damped cavity: nonzero steady state and nonzero photon current,
        # so the cumulants are a meaningful test (not the trivial vacuum case).
        b = FockBasis(6)
        a = destroy(b)
        H = 0.5 * (a' * a) + 1.0 * (a + a')   # detuning + coherent drive
        κ = 1.0
        J = [sqrt(κ) * a]
        # Use the exact (eigenvector) steady state. With an approximate steady
        # state the singular-LU and gauge-fixed-iterative paths treat the near-null
        # space slightly differently, so higher cumulants only agree to the
        # steady-state accuracy — not a solver discrepancy.
        ρss = steadystate.eigenvector(H, J)
        mJ = [sqrt(κ) * a]
        nu = [1.0]

        c_lu = QuantumFCS.fcscumulants_recursive(
            LindbladFCS(H, J; mJ = mJ, rho_ss = ρss, nu = nu, nC = 3, method = :lu))
        c_it = QuantumFCS.fcscumulants_recursive(
            LindbladFCS(H, J; mJ = mJ, rho_ss = ρss, nu = nu, nC = 3, method = :iterative))

        @test c_lu ≈ c_it rtol = 1e-5
    end

    @testset "low-level prepare_drazin_solver / drazin_solve" begin
        # Build a small sparse Liouvillian and exercise the solver directly:
        # the result must satisfy the trace-zero gauge and solve the projected system.
        b = FockBasis(5)
        a = destroy(b)
        H = 0.3 * (a' * a) + 0.8 * (a + a')
        J = [sqrt(1.0) * a]
        ρss = steadystate.iterative(H, J)

        L = SparseMatrixCSC{ComplexF64,Int}(liouvillian(H, J).data)
        n = size(ρss.data, 1)
        l = n * n
        diag_idx = collect(1:(n + 1):l)
        vId = SparseVector{ComplexF64,Int}(l, diag_idx, fill(1.0 + 0.0im, n))
        vρ = SparseVector(vec(Matrix(ρss.data) ./ tr(ρss.data)))

        # Qualify with `QuantumFCS.` so these hit the package functions that the
        # extension extends (runtests.jl also `include`s the source into Main).
        solver = QuantumFCS.prepare_drazin_solver(L, vρ, vId; method = :iterative, rtol = 1e-10)
        @test solver isa QuantumFCS.DrazinSolver

        # Consistent RHS lying in range(L).
        x = randn(ComplexF64, l)
        α = L * x
        y = QuantumFCS.drazin_solve(solver, α)

        # Gauge: trace of the (de-vectorized) result vanishes.
        @test abs(dot(vId, y)) < 1e-8
        # Solves the projected system L y ≈ α' = α − ρ (vId·α).
        αp = α .- vρ .* dot(vId, α)
        @test norm(L * Vector(y) .- αp) / norm(αp) < 1e-6
    end

    @testset "analytic quantum-dot current and noise" begin
        # Same system/analytics as the LU path in fcsproblem.jl, via :iterative.
        b = FockBasis(1)
        d = destroy(b)
        d_dag = create(b)
        ϵd, κc, κh = 1.0, 0.1, 0.5
        H = ϵd * d_dag * d
        Jcloss = sqrt(κc) * d
        Jhgain = sqrt(κh) * d_dag
        J = [Jcloss, Jhgain]
        ρss = steadystate.iterative(H, J)

        c1_analytical = κc * κh / (κc + κh)
        c2_analytical = (κh^2 + κc^2) / (κc + κh)^2 * c1_analytical

        p = LindbladFCS(H, J; mJ = [Jcloss], rho_ss = ρss, nu = [1], nC = 2,
                        method = :iterative)
        c1, c2 = QuantumFCS.fcscumulants_recursive(p)
        @test c1 ≈ c1_analytical rtol = 1e-6
        @test c2 ≈ c2_analytical rtol = 1e-6
    end
end
