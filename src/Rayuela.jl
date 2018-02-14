module Rayuela

# For PQ/OPQ training and encoding
using Clustering, Distances

# For LSQR
using IterativeSolvers

# For sampling
using Distributions

# For LSQ encoding in the GPU
using CUDAdrv, CUBLAS # NO GPU in this branch

### Load and initialize the linscan binaries ###
const depsfile = joinpath(dirname(@__DIR__), "deps", "deps.jl")
if isfile(depsfile)
  include(depsfile)
else
  error("Rayuela is not properly installed. Please run Pkg.build(\"Rayuela\")")
end
cudautilsptx = cudautils[1:end-2] * "ptx"

# === Functions to read data ===
# TODO refactor these with metaprogrammming
include("bvecs_read.jl")
include("fvecs_read.jl")
include("ivecs_read.jl")
include("read_datasets.jl")

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
include("RVQ.jl")
include("ERVQ.jl")

# === LSQ Quantizer ===
include("LSQ.jl") # Local search quantization
include("SR_perturbations.jl")
include("SR.jl")

# === CUDA ports ===
include("CudaUtilsModule.jl")
include("LSQ_GPU.jl") # Local search quantization

end # module
