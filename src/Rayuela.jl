module Rayuela

using Clustering, Distances

using IterativeSolvers

### Load and initialize the HDF library ###
const depsfile = joinpath(dirname(@__DIR__), "deps", "deps.jl")
if isfile(depsfile)
    include(depsfile)
else
    error("Rayuela not properly installed. Please run Pkg.build(\"Rayuela\")")
end

# === Utility functions mostly ===
include("utils.jl")
include("Linscan.jl")
include("codebook_update.jl")

# === Quantizers ===
include("PQ.jl")  # Product Quantizer
include("OPQ.jl") # Optimized Product Quantizer
include("ChainQ.jl") # Chain (Tree) Quantization
include("LSQ.jl") # Local search quantization

# package code goes here

end # module
