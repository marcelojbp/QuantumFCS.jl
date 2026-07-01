using QuantumFCS
using QuantumOptics
using LinearAlgebra
using SparseArrays

# Minimal benchmark/problem instance for type inference and profiling
function demo_instance(dim::Int = 6; jumps = 2)
    b = FockBasis(dim)
    a = destroy(b); ad = create(b)
    H = (ad * a)
    J = [a]
    if jumps == 2
        push!(J, ad)
    end
    ρss = steadystate.iterative(H, J)
    # Monitor both jumps
    mJ = J
    nC = 3
    return H, J, mJ, nC, ρss
end

function run_fcscumulants(dim::Int = 6)
    H, J, mJ, nC, ρss = demo_instance(dim)
    return fcscumulants_recursive(H, J, mJ, nC, ρss)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println(run_fcscumulants())
end
