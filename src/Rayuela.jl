module Rayuela

# For PQ/OPQ training and encoding
using Clustering, Distances

# For LSQR
using IterativeSolvers

# For LSQ encoding in the GPU
using CUDAdrv, CUBLAS

### Load and initialize the linscan binaries ###
const depsfile = joinpath(dirname(@__DIR__), "deps", "deps.jl")
if isfile(depsfile)
  include(depsfile)
else
  error("Rayuela is not properly installed. Please run Pkg.build(\"Rayuela\")")
end
cudautils = cudautils[1:end-2] * "ptx"

# === Utility functions mostly ===
include("utils.jl")
include("Linscan.jl")
include("codebook_update.jl")

# === Quantizers ===
include("PQ.jl")  # Product Quantizer
include("OPQ.jl") # Optimized Product Quantizer
include("ChainQ.jl") # Chain (Tree) Quantization

# === LSQ Quantizer ===
include("CudaUtilsModule.jl")
include("LSQ.jl") # Local search quantization

end # module
