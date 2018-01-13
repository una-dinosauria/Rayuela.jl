module Rayuela

# For PQ/OPQ training and encoding
using Clustering, Distances

# For LSQR
using IterativeSolvers

# For TODO
using Distances

# For sampling
using Distributions

# For LSQ encoding in the GPU
# using CUDAdrv, CUBLAS # NO GPU in this branch

### Load and initialize the linscan binaries ###
const depsfile = joinpath(dirname(@__DIR__), "deps", "deps.jl")
if isfile(depsfile)
  include(depsfile)
else
  error("Rayuela is not properly installed. Please run Pkg.build(\"Rayuela\")")
end
# cudautilsptx = cudautils[1:end-2] * "ptx"

# === Utility functions mostly ===
include("utils.jl")
include("qerrors.jl")

include("Linscan.jl")
include("codebook_update.jl")

# === Orthogonal quantizers ===
include("PQ.jl")  # Product Quantizer
include("OPQ.jl") # Optimized Product Quantizer

# === Non-orthogonal quantizers ===
include("ChainQ.jl") # Chain (Tree) Quantization
include("RQ.jl")

# === LSQ Quantizer ===
# include("CudaUtilsModule.jl")
include("LSQ.jl") # Local search quantization
# include("LSQ_GPU.jl") # Local search quantization

end # module
