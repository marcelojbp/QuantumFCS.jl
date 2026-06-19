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
