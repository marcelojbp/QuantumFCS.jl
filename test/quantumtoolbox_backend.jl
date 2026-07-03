# QuantumToolbox backend tests.
#
# `@safetestset` runs this file in its own module, so the QuantumToolbox names
# (`Operator`, `destroy`, `create`, `steadystate`, ...) that clash with
# QuantumOptics never collide with the QuantumOptics-based test files.

using QuantumFCS
using QuantumToolbox
using Test
using LinearAlgebra
using SparseArrays

# The QuantumFCSQuantumToolboxExt extension (and thus the methods below) is
# auto-loaded by Julia as soon as both QuantumFCS and QuantumToolbox are imported.

@testset "QuantumToolbox: single quantum dot (analytic)" begin
    # Mirror of test/qd_heat_engine.jl using the QuantumToolbox API.
    # FockBasis(1) in QuantumOptics ↔ a 2-dimensional Fock space (n = 0, 1).
    d = destroy(2)
    d_dag = create(2)

    ϵd = 1.0   # Energy level of the quantum dot
    κc = 0.1   # Coupling strength to cold reservoir
    κh = 0.5   # Coupling strength to hot reservoir

    H = ϵd * d_dag * d
    Jcloss = sqrt(κc) * d        # Jumps into the cold reservoir
    Jhgain = sqrt(κh) * d_dag    # Jumps from the hot reservoir
    J = [Jcloss, Jhgain]

    ρss = steadystate(H, J)
    nu = [1]
    mJ = [Jcloss]   # Monitor particles entering the cold reservoir

    c1, c2 = QuantumFCS.fcscumulants_recursive(H, J, mJ, 2, ρss, nu)

    c1_analytical = κc * κh / (κc + κh)
    c2_analytical = (κh^2 + κc^2) / (κc + κh)^2 * c1_analytical
    @test c1 ≈ c1_analytical atol = 1.0e-10
    @test c2 ≈ c2_analytical atol = 1.0e-10
end

@testset "QuantumToolbox vs matrix Liouvillian (two-level system)" begin
    # Amplitude-damped qubit. We check that the QuantumToolbox backend produces
    # the same first two cumulants as feeding the core engine a Liouvillian that
    # is assembled by hand as a plain matrix.
    ω = 1.3   # qubit splitting
    γ = 0.7   # decay rate

    # --- QuantumToolbox path ---
    H = 0.5 * ω * sigmaz()
    J = [sqrt(γ) * sigmam()]
    mJ = J
    nu = [1.0]
    ρss = steadystate(H, J)
    c_qt = QuantumFCS.fcscumulants_recursive(H, J, mJ, 2, ρss, nu)

    # --- Hand-built matrix Liouvillian path ---
    # Use the same operator data, but assemble the (vectorized) Liouvillian and
    # steady state independently with plain matrices, mirroring the column-major
    # convention spre(A) = kron(I, A), spost(B) = kron(transpose(B), I).
    Hm = Matrix{ComplexF64}(H.data)
    Jm = Matrix{ComplexF64}(J[1].data)
    n = size(Hm, 1)
    Id = Matrix{ComplexF64}(I, n, n)
    spre(A) = kron(Id, A)
    spost(B) = kron(transpose(B), Id)

    JdJ = Jm' * Jm
    Lham = -1im * (spre(Hm) - spost(Hm))
    Ldiss = kron(conj(Jm), Jm) - 0.5 * (spre(JdJ) + spost(JdJ))
    Lm = Lham + Ldiss   # 4×4 vectorized Liouvillian

    # Steady state as the (normalized) kernel of the Liouvillian.
    vss = nullspace(Lm)[:, 1]
    ρm = reshape(vss, n, n)
    ρm ./= tr(ρm)

    c_mat = QuantumFCS.fcscumulants_recursive(
        sparse(Lm),
        [sparse(Jm)],
        2,
        sparse(ρm),
        nu,
    )

    @test c_qt ≈ c_mat atol = 1.0e-10
end
