# core API that does not require QuantumOptics
module QuantumFCS
    using LinearAlgebra
    using SparseArrays
    include("FCS_functions.jl")
    export fcscumulants_recursive, factorial_cumulants
end
