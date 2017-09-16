module Rayuela

using Clustering, Distances, Distributions

# === Utility functions mostly ===
include("utils.jl")

# === Quantizers ===
include("PQ.jl")  # Product Quantizer
include("OPQ.jl") # Optimized Product Quantizer

# package code goes here

end # module
