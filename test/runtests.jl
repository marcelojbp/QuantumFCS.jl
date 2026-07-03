using SafeTestsets

# High-level runner: this file only shuttles to the individual test scripts. Each
# is wrapped in `@safetestset` so it runs in its own module (no value leakage) and
# is fully reproducible in isolation (`julia --project=. test/<file>.jl`).
@time @safetestset "Drazin comparison"          include("drazin_comparison.jl")
@time @safetestset "Drazin inverse"             include("drazin_inverse.jl")
@time @safetestset "QD heat engine"             include("qd_heat_engine.jl")
@time @safetestset "FCSProblem (QuantumOptics)" include("fcsproblem.jl")
@time @safetestset "FCSProblem (QuantumToolbox)" include("fcsproblem_quantumtoolbox.jl")
@time @safetestset "QuantumToolbox backend"     include("quantumtoolbox_backend.jl")
@time @safetestset "Iterative Drazin"           include("iterative_drazin.jl")
