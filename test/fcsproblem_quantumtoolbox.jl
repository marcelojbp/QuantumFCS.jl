@testset "FCSProblem (QuantumToolbox backend)" begin
    # Same quantum-dot heat engine, expressed with QuantumToolbox QuantumObjects,
    # to exercise the QuantumToolbox extension hooks.
    N = 2                              # Fock space {|0>, |1>} (matches FockBasis(1))
    a = QuantumToolbox.destroy(N)
    a_dag = a'
    ϵd = 1.0
    κc = 0.1
    κh = 0.5
    H = ϵd * a_dag * a
    Jcloss = sqrt(κc) * a
    Jhgain = sqrt(κh) * a_dag
    J = [Jcloss, Jhgain]
    ρss = QuantumToolbox.steadystate(H, J)
    nu = [1]
    mJ = [Jcloss]

    c1_analytical = κc * κh / (κc + κh)
    c2_analytical = (κh^2 + κc^2) / (κc + κh)^2 * c1_analytical

    p = LindbladFCS(; H = H, J = J, mJ = mJ, rho_ss = ρss, nu = nu, nC = 2)
    c1, c2 = QuantumFCS.fcscumulants_recursive(p)
    @test c1 ≈ c1_analytical atol = 1e-10
    @test c2 ≈ c2_analytical atol = 1e-10
    @test (@inferred QuantumFCS.fcscumulants_recursive(p)) ≈ [c1_analytical, c2_analytical] atol = 1e-10

    # Convenience constructor.
    p2 = LindbladFCS(H, J; mJ = mJ, rho_ss = ρss, nu = nu, nC = 2)
    @test QuantumFCS.fcscumulants_recursive(p2) ≈ [c1_analytical, c2_analytical] atol = 1e-10
end
