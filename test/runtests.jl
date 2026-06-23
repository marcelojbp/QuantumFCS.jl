using QuantumFCS
using QuantumOptics
import QuantumToolbox   # imported (not `using`) to avoid clashing with QuantumOptics' exported names
using Test
using LinearAlgebra
using SparseArrays

## including the functions that are not exported by the package
include("../src/FCS_functions.jl")
include("../ext/FCS_QuantumOptics_functions.jl")
@testset "QuantumFCS.jl" begin
    include("drazin_comparison.jl")
    include("drazin_inverse.jl")
    include("qd_heat_engine.jl")
    include("fcsproblem.jl")
    include("fcsproblem_quantumtoolbox.jl")
end

