using QuantumFCS
using QuantumOptics
using LinearAlgebra
using Test

@testset "Drazin comparison" begin
    # Defining the example systems
    basis = FockBasis(4)
    adagger = create(basis)
    a = destroy(basis)
    H = adagger * a
    J = [a]

    # Defining the input state and calculating the steady state
    α = Complex((1 + im) / sqrt(2))
    ρα = coherentstate(basis, α) ⊗ dagger(coherentstate(basis, α))
    valpha = vec(Matrix(ρα.data))
    rho_ss = steadystate.iterative(H, J)
    vrho_ss = vec(Matrix(rho_ss.data))
    l = length(rho_ss)

    # Identity in Liouville space (this will usually be passed from the higher level function)
    IdL = Matrix{ComplexF64}(I, l, l)

    # Defining the vectorized identity (this will usually be passed from the higher level function)
    vId = vec(Matrix{ComplexF64}(I, size(rho_ss.data)))'

    # testing whether drazin_apply and drazin yield the same result
    @test norm(QuantumFCS.drazin_apply(H, J, valpha, vrho_ss, vId) - QuantumFCS.drazin(H, J, vrho_ss, vId, IdL) * valpha) ≈ 0 atol = 10^(-10)
end
