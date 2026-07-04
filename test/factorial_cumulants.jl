using QuantumFCS
using QuantumOptics
using LinearAlgebra
using SparseArrays
using Test

@testset "Factorial cumulants" begin
    # First check the conversion formula directly for a generic set of ordinary
    # cumulants. These values are intentionally not from a special distribution,
    # so the test verifies the signed-Stirling transformation itself.
    c = [2.0, 7.0, 13.0, 31.0]

    # factorial_cumulants should return f_m = sum_j s(m,j)c_j, where s(m,j)
    # are signed Stirling numbers of the first kind.
    f = QuantumFCS.factorial_cumulants(c)

    # The first four signed-Stirling rows are:
    # s(1,:) = [1]
    # s(2,:) = [-1, 1]
    # s(3,:) = [2, -3, 1]
    # s(4,:) = [-6, 11, -6, 1]
    # These tests spell out the expected combinations order by order.
    @test f[1] ≈ c[1]
    @test f[2] ≈ c[2] - c[1]
    @test f[3] ≈ c[3] - 3 * c[2] + 2 * c[1]
    @test f[4] ≈ c[4] - 6 * c[3] + 11 * c[2] - 6 * c[1]

    # For a Poisson process all ordinary cumulants are equal to the rate/mean λ.
    # Its factorial cumulant generating function is linear, so every factorial
    # cumulant above first order should vanish.
    λ = 2.5

    # Use six orders here to exercise the recurrence beyond the explicitly
    # checked first four rows.
    poisson_cumulants = fill(λ, 6)
    poisson_factorial = QuantumFCS.factorial_cumulants(poisson_cumulants)

    # The first factorial cumulant equals the mean count λ.
    @test poisson_factorial[1] ≈ λ

    # Higher orders should be numerically zero. Use an absolute tolerance because
    # these are cancellation checks, and the expected value is exactly zero.
    @test all(x -> isapprox(x, 0.0; atol = 1e-12), poisson_factorial[2:end])

    # For a binomial random variable with N trials and success probability p,
    # the factorial cumulants have the closed form
    # f_m = (-1)^(m-1) * factorial(m-1) * N * p^m.
    N = 8
    p = 0.3

    # Ordinary binomial cumulants through fourth order. These are the inputs to
    # the function under test; the expected values below are factorial cumulants.
    binomial_cumulants = [
        N * p,
        N * p * (1 - p),
        N * p * (1 - p) * (1 - 2 * p),
        N * p * (1 - p) * (1 - 6 * p * (1 - p)),
    ]
    binomial_factorial = QuantumFCS.factorial_cumulants(binomial_cumulants)

    # Build the closed-form factorial cumulants order by order, then compare the
    # whole vector. This also checks that the output ordering is [f1, f2, ...].
    expected_binomial = [
        (-1)^(m - 1) * factorial(m - 1) * N * p^m for m = 1:4
    ]
    @test binomial_factorial ≈ expected_binomial
end

@testset "fcscumulants_recursive cumulant_type keyword" begin
    # Build the same two-state quantum-dot heat engine in two representations:
    # plain sparse matrices for the core Liouvillian API, and QuantumOptics
    # operators for the extension API. The keyword should only affect the final
    # post-processing step, not the recursive cumulant calculation itself.
    ϵd = 1.0
    κc = 0.1
    κh = 0.5

    # Matrix representation. The basis ordering is [empty, occupied].
    occupied = complex.([0, 1])
    empty = complex.([1, 0])
    id = sparse(Matrix{ComplexF64}(I, 2, 2))
    H = sparse(ϵd * (occupied * occupied'))

    # In the large-bias model, particles leave through the cold reservoir and
    # enter from the hot reservoir. We monitor only the cold-loss jump.
    Jcloss = sparse(sqrt(κc) * (empty * occupied'))
    Jhgain = sparse(sqrt(κh) * (occupied * empty'))
    J = [Jcloss, Jhgain]
    mJ = [Jcloss]
    nu = [1]

    # Construct the vectorized Lindblad generator L by hand for the matrix API.
    # This mirrors the package examples: unitary part plus jump and anticommutator
    # contributions, using column-major vectorization.
    L = -im * (kron(id, H) - kron(transpose(H), id))
    for jump in J
        L += kron(conj(jump), jump)
        L += -0.5 * kron(id, jump' * jump)
        L += -0.5 * kron(transpose(jump' * jump), id)
    end

    # The steady state of this two-state rate equation is diagonal: probability
    # κc/(κc+κh) for empty and κh/(κc+κh) for occupied.
    rho_ss = sparse(ComplexF64[
        κc / (κc + κh) 0
        0 κh / (κc + κh)
    ])

    # Sparse Liouvillian dispatch: omitting the keyword is the original public
    # call style and must continue to return ordinary cumulants.
    sparse_ordinary = QuantumFCS.fcscumulants_recursive(L, mJ, 3, rho_ss, nu)
    sparse_explicit_ordinary = QuantumFCS.fcscumulants_recursive(
        L,
        mJ,
        3,
        rho_ss,
        nu;
        cumulant_type = "",
    )
    sparse_factorial = QuantumFCS.fcscumulants_recursive(
        L,
        mJ,
        3,
        rho_ss,
        nu;
        cumulant_type = "factorial",
    )

    @test sparse_explicit_ordinary ≈ sparse_ordinary
    @test sparse_factorial ≈ QuantumFCS.factorial_cumulants(sparse_ordinary)
    @test_throws ArgumentError QuantumFCS.fcscumulants_recursive(
        L,
        mJ,
        3,
        rho_ss,
        nu;
        cumulant_type = "ordinary",
    )

    # Dense Liouvillian dispatch: the same keyword behavior should hold when L
    # is a dense Matrix{ComplexF64}. Use first order here so the test covers the
    # public dense dispatch without entering the existing dense Drazin solve for
    # singular Liouvillians.
    dense_ordinary = QuantumFCS.fcscumulants_recursive(Matrix(L), mJ, 1, rho_ss, nu)
    dense_explicit_ordinary = QuantumFCS.fcscumulants_recursive(
        Matrix(L),
        mJ,
        1,
        rho_ss,
        nu;
        cumulant_type = "",
    )
    dense_factorial = QuantumFCS.fcscumulants_recursive(
        Matrix(L),
        mJ,
        1,
        rho_ss,
        nu;
        cumulant_type = "factorial",
    )

    @test dense_ordinary ≈ sparse_ordinary[1:1]
    @test dense_explicit_ordinary ≈ dense_ordinary
    @test dense_factorial ≈ QuantumFCS.factorial_cumulants(dense_ordinary)
    @test_throws ArgumentError QuantumFCS.fcscumulants_recursive(
        Matrix(L),
        mJ,
        1,
        rho_ss,
        nu;
        cumulant_type = "factorial-cumulants",
    )

    # QuantumOptics dispatch: build the same model with operators and verify the
    # extension passes cumulant_type through to the core implementation.
    b = FockBasis(1)
    d = destroy(b)
    d_dag = create(b)
    H_qo = ϵd * d_dag * d
    Jcloss_qo = sqrt(κc) * d
    Jhgain_qo = sqrt(κh) * d_dag
    J_qo = [Jcloss_qo, Jhgain_qo]
    rho_ss_qo = steadystate.iterative(H_qo, J_qo)

    qo_ordinary = QuantumFCS.fcscumulants_recursive(H_qo, J_qo, [Jcloss_qo], 3, rho_ss_qo, nu)
    qo_explicit_ordinary = QuantumFCS.fcscumulants_recursive(
        H_qo,
        J_qo,
        [Jcloss_qo],
        3,
        rho_ss_qo,
        nu;
        cumulant_type = "",
    )
    qo_factorial = QuantumFCS.fcscumulants_recursive(
        H_qo,
        J_qo,
        [Jcloss_qo],
        3,
        rho_ss_qo,
        nu;
        cumulant_type = "factorial",
    )

    @test qo_explicit_ordinary ≈ qo_ordinary
    @test qo_factorial ≈ QuantumFCS.factorial_cumulants(qo_ordinary)
    @test_throws ArgumentError QuantumFCS.fcscumulants_recursive(
        H_qo,
        J_qo,
        [Jcloss_qo],
        3,
        rho_ss_qo,
        nu;
        cumulant_type = "central",
    )
end
