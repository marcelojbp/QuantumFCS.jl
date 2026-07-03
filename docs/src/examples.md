# [Examples](@id examples)

## Quantum dot heat engine 

We study a quantum dot coupled to two fermionic reservoirs, for details on the model see [Patrick P. Potts, 2024](https://arxiv.org/pdf/2406.19206). 

```julia
using QuantumFCS, LinearAlgebra, SparseArrays

```

Define parameters and system

```julia
# System parameters
ϵd = 1.0;  # Energy level of the quantum dot
κc = 0.1;  # Coupling strength to cold reservoir
κh = 0.5;  # Coupling strength to hot reservoir
nc = 0.0;  # Occupation number of cold reservoir
nh = 1.0;  # Occupation number of hot reservoir

# Basis states and identity 
occupied = complex.([0,1]);
empty = complex.([1,0]);
id = sparse(1.0I,2,2);

## Operators
# Hamiltonian 
H = sparse(ϵd * (occupied * occupied'))  
# Jump operators
Jcloss = sparse(sqrt(κc * (1-nc)) * (empty * occupied'));
Jcgain = sparse(sqrt(κc * nc) * (occupied * empty'));
Jhgain = sparse(sqrt(κh * nh) * (occupied * empty'));
Jhloss = sparse(sqrt(κh * (1-nh)) * (empty * occupied'));
```

We construct the Liouvillian 

```julia
## Constructing the Liouvillian
# Unitary part 
L = -im * (kron(id,H) - kron(transpose(H),id))
# Jump terms from the dissipators 
L += kron(conj(Jcloss),Jcloss) + kron(conj(Jhloss),Jhloss) + kron(conj(Jhgain),Jhgain)
# Non-jump terms from the dissipators
L += (-1/2 * kron(id,Jcloss'*Jcloss) 
     -1/2 * kron(transpose(Jcloss'*Jcloss),id) 
     -1/2 * kron(id,Jcgain'*Jcgain) 
     -1/2 * kron(transpose(Jcgain'*Jcgain),id) 
     -1/2 * kron(id,Jhloss'*Jhloss) 
     -1/2 * kron(transpose(Jhloss'*Jhloss),id)
     -1/2 * kron(id,Jhgain'*Jhgain) 
     -1/2 * kron(transpose(Jhgain'*Jhgain),id))  
```  

 and determine the steady state 

```julia
 # Determining the steady state
E = eigen(Matrix(L))                       # compute all eigenpairs (fine for small L)
idx = argmin(abs.(E.values))               # index of eigenvalue closest to zero
v = E.vectors[:, idx]                      # right eigenvector (vec(ρ))

# reshape back to n×n (vec stacks columns, so reshape(v, n, n) gives the matrix)
ρ_ss = reshape(v, 2, 2);

# numeric cleanup: make Hermitian and normalize trace
ρ_ss = (ρ_ss + ρ_ss') / 2  ;               # enforce Hermiticity (conjugate-transpose)
ρ_ss = ρ_ss / tr(ρ_ss) ;                   # normalize trace to 1
```  

We choose to monitor the electrons entering the cold reservoir 

```julia
# Vector with weights 
nu = [1];

# Monitored jump operators (photons entering the cold reservoir)
mJ = [Jcloss];
```  

```julia
# Computing the first two cumulants
c1, c2 = fcscumulants_recursive(L, mJ, 2, sparse(ρ_ss), nu);
println("\nFull Counting Statistics (numerics):")
println("First cumulant : $c1")
println("Second cumulant : $c2") 
```

```julia
Full Counting Statistics (numerics):
First cumulant : 0.08333333333333334
Second cumulant : 0.06018518518518519
```

In the large bias regime we have the following analytical solutions [Patrick P. Potts, 2024](https://arxiv.org/pdf/2406.19206)

```julia
c1_analytical = κc*κh/(κc+κh);
c2_analytical = (κh^2+κc^2)/(κc+κh)^2*c1_analytical;
println("\nFull Counting Statistics (analytical):")
println("First cumulant : $c1_analytical")
println("Second cumulant : $c2_analytical") 
```  

```julia
Full Counting Statistics (analytical):
First cumulant : 0.08333333333333334
Second cumulant : 0.0601851851851852
``` 

## Driven cavity  

We study a simple driven cavity using the *QuantumOptics.jl* package to model the system

```julia
using QuantumOptics
using QuantumFCS
```

Define parameters and system 

```julia

# Define the Hilbert space dimension 
N = 10  # Fock space cutoff

# Create the Fock basis for the cavity mode
basis = FockBasis(N)

# Define operators
a = destroy(basis)  # annihilation operator
a_dag = create(basis)  # creation operator
n = number(basis)   # number operator

# System parameters
ω = 1.0     # cavity frequency
κ = 0.1     # cavity decay rate
Ω = 0.5     # drive strength (coherent drive)

# Hamiltonian (driven cavity)
H = ω * n + Ω * (a_dag + a)  # free evolution + coherent drive

# Jump operators (cavity decay)
J = [sqrt(κ) * a]  # photon loss

# Vector with weights (for one monitored channel)
nu = [-1]

# Monitored jump operators (photons leaving the cavity)
mJ = [sqrt(κ) * a]

# Initial state (vacuum state)
ψ0 = fockstate(basis, 0)
ρ0 = dm(ψ0)
```

Determine the steady state 

```julia
ρss = steadystate.iterative(H, J);

```

Compute the first three cumulants

```julia
c1, c2, c3 = fcscumulants_recursive(H, J, mJ, 3, ρss, nu)

println("\nFull Counting Statistics:")
println("First cumulant (mean photon flux): $c1")
println("Second cumulant (variance): $c2") 
println("Third cumulant (skewness): $c3")
```

```julia
Full Counting Statistics:
First cumulant (mean photon flux): -0.02104572040194588
Second cumulant (variance): 0.012890993774510551
Third cumulant (skewness): 0.011185533737017767
```

## Applications from the paper (coming soon)

A companion paper with worked applications of `QuantumFCS.jl` is in preparation. Its
examples will be added here as runnable notebooks/scripts once the manuscript is public.
