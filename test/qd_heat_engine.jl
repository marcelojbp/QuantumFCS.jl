using QuantumFCS
using QuantumOptics
using Test

@testset "QD heat engine" begin
    # Basis for the single quantum dot
    b = FockBasis(1)
    # Operators
    d = destroy(b)
    d_dag = create(b)
    id = identityoperator(b)
    # Parameters
    ϵd = 1.0                             # Energy level of the quantum dot
    κc = 0.1                             # Coupling strength to cold reservoir
    κh = 0.5                             # Coupling strength to hot reservoir
    # Hamiltonian and jump operators
    H = ϵd * d_dag * d
    Jcloss = sqrt(κc) * d                # Jumps into the cold reservoir
    Jhgain = sqrt(κh) * d_dag            # Jumps from the hot reservoir
    J = [Jcloss, Jhgain]
    # Steady state
    ρss = steadystate.iterative(H, J)
    # Weight vector
    nu = [1]
    # Monitored jump operator (particles entering the cold reservoir)
    mJ = [Jcloss]
    # Calculating the first two cumulants
    c1, c2 = QuantumFCS.fcscumulants_recursive(H, J, mJ, 2, ρss, nu)
    c1_analytical = κc * κh / (κc + κh)
    c2_analytical = (κh^2 + κc^2) / (κc + κh)^2 * c1_analytical
    @test c1 ≈ c1_analytical atol = 10^(-10)
    @test c2 ≈ c2_analytical atol = 10^(-10)
end
