module Rayuela

using Clustering, Distances, Distributions

using IterativeSolvers

# === Utility functions mostly ===
include("utils.jl")
include("codebook_update.jl")

# === Quantizers ===
include("PQ.jl")  # Product Quantizer
include("OPQ.jl") # Optimized Product Quantizer
include("ChainQ.jl") # Chain (Tree) Quantization

# package code goes here

end # module
