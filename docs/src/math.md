# [Mathematical Background](@id math)

`QuantumFCS.jl` computes zero-frequency current cumulants for time-independent
Lindblad master equations. The system dynamics are written as

```math
\frac{d\rho}{dt}
= -i[H,\rho] + \sum_{k=1}^r \mathcal{D}[J_k]\rho
\equiv \mathcal{L}\rho,
```

with dissipator

```math
\mathcal{D}[A]\rho
= A\rho A^\dagger - \frac{1}{2}\{A^\dagger A,\rho\}.
```

Here, `H` is the Hamiltonian, `J = (J_1,\ldots,J_r)` is the list of jump
operators, and `L` is the Liouvillian superoperator. The recursive construction
assumes that ``\mathcal{L}`` has a unique stationary state ``\rho_{\rm ss}``,
normalised by ``\operatorname{tr}\rho_{\rm ss}=1``.

## Monitored jumps and weights

A stochastic current is defined by choosing a list of monitored jumps

```math
\mathcal{M} = (M_1,\ldots,M_p), \qquad M_l \in \{J_1,\ldots,J_r\},
```

and a matching list of weights

```math
\nu = (\nu_1,\ldots,\nu_p).
```

In the package, these are the inputs `mJ` and `nu`. The weights set both the sign
convention and the units of the current: they can be dimensionless values such as
``\pm 1`` for particle or photon counts, charges for electric currents, or energy
quanta for heat currents.

Let ``N_l(t)`` be the number of jumps through monitored channel ``M_l`` in the
interval ``[0,t]``. The integrated current is

```math
N(t) = \sum_{l=1}^p \nu_l N_l(t)
= \int_0^t dt'\, I(t'),
```

which implicitly defines the stochastic current ``I(t)``.

## Tilted Liouvillian

To generate the statistics of ``N(t)``, introduce the counting-field-resolved
density matrix

```math
\rho_\chi(t)
= \sum_{n\in\mathcal{N}} e^{i\chi n}\rho(n;t),
```

where ``\operatorname{tr}\rho(n;t)`` is the probability that the integrated
current has value ``n``. It evolves under the tilted Liouvillian

```math
\mathcal{L}_\chi \rho_\chi
= \mathcal{L}\rho_\chi + \delta\mathcal{L}_\chi\rho_\chi,
```

with

```math
\delta\mathcal{L}_\chi\rho
= \sum_{l=1}^p \left(e^{i\chi\nu_l}-1\right)M_l\rho M_l^\dagger.
```

This sign convention matches the definition of ``\rho_\chi`` above and the package
implementation. The integrated-current cumulant generating function is

```math
C_N(\chi;t)=\log\operatorname{tr}\rho_\chi(t),
```

and the long-time current cumulants are

```math
c_n \equiv \langle\!\langle I^n\rangle\!\rangle
= \left.(-i\partial_\chi)^n
\lim_{t\to\infty}\frac{C_N(\chi;t)}{t}\right|_{\chi=0}.
```

## Recursive cumulant scheme

The package computes these cumulants without differentiating the dominant
eigenvalue of ``\mathcal{L}_\chi`` numerically. In vectorised notation,
``|A\rangle\rangle`` denotes the vector form of an operator and
``\langle\langle 1|`` is the trace functional. Define

```math
\mathcal{L}^{(n)}
= \left.(-i\partial_\chi)^n \mathcal{L}_\chi\right|_{\chi=0}
= \sum_{l=1}^p \nu_l^n M_l^* \otimes M_l.
```

Starting from ``|\sigma_0\rangle\rangle = |\rho_{\rm ss}\rangle\rangle``, the
cumulants and auxiliary pseudo-states are generated recursively as

```math
\langle\!\langle I^n\rangle\!\rangle
= \sum_{m=1}^n {n\choose m}
\langle\!\langle 1|\mathcal{L}^{(m)}|\sigma_{n-m}\rangle\!\rangle,
```

```math
|\sigma_n\rangle\rangle
= \mathcal{L}^+
\sum_{m=1}^n {n\choose m}
\left(\langle\!\langle I^m\rangle\!\rangle\mathbb{I}
- \mathcal{L}^{(m)}\right)
|\sigma_{n-m}\rangle\rangle.
```

Here, ``\mathcal{L}^+`` is the Drazin inverse of the Liouvillian on the
subspace orthogonal to the stationary state. Numerically, `QuantumFCS.jl` prepares
a solver for applying ``\mathcal{L}^+`` and reuses that solver across cumulant
orders; see [Drazin solvers](@ref solvers).

## Mapping to package inputs

The mathematical objects above map directly to the main package interface:

- `(H, J)` or a preassembled `L` define the Liouvillian ``\mathcal{L}``.
- `rho_ss` supplies the stationary state ``\rho_{\rm ss}``.
- `mJ` is the monitored-jump list ``\mathcal{M}``.
- `nu` is the weight list ``\nu``.
- `nC` is the number of cumulants to return.

These inputs can be passed directly to [`fcscumulants_recursive`](@ref), or bundled
in a [`LindbladFCS`](@ref) problem.

## References

- **FCS in Lindblad master equations**: Potts (2024). "Quantum Thermodynamics" [arXiv](https://doi.org/10.48550/arXiv.2406.19206)

- **FCS, recursive scheme, vectorization, Drazin inverse**: Landi et al. (2024). "Current fluctuations in open quantum systems: Bridging the gap between quantum continuous measurements and full counting statistics", PRX Quantum 5, 020201 (2024) [arXiv](https://arxiv.org/abs/2303.04270) 

- **Detailed exposition of the recursive scheme**: Flindt et al. (2010). "Counting statistics of transport through Coulomb blockade nanostructures: High-order cumulants and non-Markovian effects" Phys. Rev. B 82, 155407 [arXiv](https://arxiv.org/abs/1002.4506)
