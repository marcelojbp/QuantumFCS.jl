using QuantumFCS
using QuantumOptics
using Test

@testset "FCSProblem (QuantumOptics backend)" begin
    # Same quantum-dot heat engine as qd_heat_engine.jl, but organised as a
    # LindbladFCS problem object and solved with the single-argument method.
    b = FockBasis(1)
    d = destroy(b)
    d_dag = create(b)
    ϵd = 1.0
    κc = 0.1
    κh = 0.5
    H = ϵd * d_dag * d
    Jcloss = sqrt(κc) * d              # jumps into the cold reservoir
    Jhgain = sqrt(κh) * d_dag          # jumps from the hot reservoir
    J = [Jcloss, Jhgain]
    ρss = steadystate.iterative(H, J)
    nu = [1]
    mJ = [Jcloss]

    c1_analytical = κc * κh / (κc + κh)
    c2_analytical = (κh^2 + κc^2) / (κc + κh)^2 * c1_analytical

    @testset "build from H and J (deferred L)" begin
        p = LindbladFCS(; H = H, J = J, mJ = mJ, rho_ss = ρss, nu = nu, nC = 2)
        c1, c2 = QuantumFCS.fcscumulants_recursive(p)
        @test c1 ≈ c1_analytical atol = 1.0e-10
        @test c2 ≈ c2_analytical atol = 1.0e-10
        # Type stability of the solve path.
        @test (@inferred QuantumFCS.fcscumulants_recursive(p)) ≈ [c1_analytical, c2_analytical] atol = 1.0e-10
    end

    @testset "convenience positional constructor" begin
        p = LindbladFCS(H, J; mJ = mJ, rho_ss = ρss, nu = nu, nC = 2)
        c1, c2 = QuantumFCS.fcscumulants_recursive(p)
        @test c1 ≈ c1_analytical atol = 1.0e-10
        @test c2 ≈ c2_analytical atol = 1.0e-10
    end

    @testset "prebuilt Liouvillian" begin
        L = liouvillian(H, J).data
        p = LindbladFCS(; L = L, mJ = mJ, rho_ss = ρss, nu = nu, nC = 2)
        c1, c2 = QuantumFCS.fcscumulants_recursive(p)
        @test c1 ≈ c1_analytical atol = 1.0e-10
        @test c2 ≈ c2_analytical atol = 1.0e-10
        @test (@inferred QuantumFCS.fcscumulants_recursive(p)) ≈ [c1_analytical, c2_analytical] atol = 1.0e-10
    end

    @testset "validation" begin
        # Neither L nor (H and J) provided.
        @test_throws ArgumentError LindbladFCS(; mJ = mJ, rho_ss = ρss, nu = nu)
        # Mismatched mJ / nu lengths.
        @test_throws ArgumentError LindbladFCS(; H = H, J = J, mJ = mJ, rho_ss = ρss, nu = [1, 2])
    end
end
