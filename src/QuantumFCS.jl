# core API that does not require QuantumOptics
module QuantumFCS
    using LinearAlgebra
    using SparseArrays
    include("FCS_functions.jl")
    include("FCSProblem.jl")
    export fcscumulants_recursive
    export FCSProblem, LindbladFCS
    export prepare_drazin_solver, drazin_solve, DrazinSolver
end
