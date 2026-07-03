using QuantumFCS
using QuantumOptics
using LinearAlgebra
using Test

@testset "Drazin inverse" begin
    # Analytic expression of the drazin inverse of the Liouvillian of a qubit coupled to a heat bath. with H = σz
    qubit_drazin(γm, γp) = [-γm / (γm + γp)^2 0 0 γp / (γm + γp)^2; 0 -2 / (-4im + γm + γp) 0 0; 0 0 -2 / (4im + γm + γp) 0; γm / (γm + γp)^2 0 0 -γp / (γm + γp)^2]
    b = SpinBasis(1 // 2)
    sm = sigmam(b)
    sp = sigmap(b)
    sz = sigmaz(b)
    Id = Matrix{ComplexF64}(I, 2, 2)
    γm = 1.0
    γp = 1.0
    H = sz
    J = [√γm * sm, √γp * sp]
    ρss = steadystate.eigenvector(H, J)
    vρss = vec(ρss.data)

    vId = ComplexF64.(vec(Id))'
    IdL = Matrix{ComplexF64}(I, length(vρss), length(vρss))

    LDtest = QuantumFCS.drazin(H, J, vρss, vId, IdL)
    @test  isapprox(norm(LDtest - qubit_drazin(γm, γp)), 0, atol = 1.0e-10)
end
