# docs/make.jl  — classic Documenter deploy to gh-pages

import Pkg
Pkg.activate(@__DIR__)    
Pkg.instantiate()

using Documenter
using QuantumFCS

DocMeta.setdocmeta!(QuantumFCS, :DocTestSetup, :(using QuantumFCS); recursive=true)

makedocs(
    sitename = "QuantumFCS.jl",
    modules  = [QuantumFCS],
    format   = Documenter.HTML(),
    doctest  = true,
    checkdocs = :none,
    pages    = [
        "Home"       => "index.md",
        "Quickstart" => "quickstart.md",
        "Mathematical Background" => "math.md",
        "Examples"   => "examples.md",
        "API"        => "api.md",
    ],
)

deploydocs(
    repo      = "github.com/marcelojbp/QuantumFCS.jl",
    devbranch = "main",
)
