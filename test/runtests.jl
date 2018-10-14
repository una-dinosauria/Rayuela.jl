using Rayuela
using Test

# Common utility functions
include("common.jl")

# IO - fvecs and ivecs read/write
include("xvecs.jl")

# Codebook update
include("codebook_update.jl")

# Chain quantization
# Test cpp viterbi encoding implementation
@testset "Viterbi encoding" begin
  d, n, m, h = 32, Int(1e3), 4, 256
  X, C, B = generate_random_dataset(Float32, Int16, d, n, m, h)

  Bj, _ = Rayuela.quantize_chainq(X, C) # Julia

  use_cuda = true
  use_cpp = false

  Bc, _ = Rayuela.quantize_chainq(X, C, use_cuda, use_cpp) # C
  @test all(Bj .== Bc)
end
