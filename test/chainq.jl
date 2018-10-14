

# Make sure the fast version of codebook update is still okay
@testset "Chain codebook update" begin
  d, n, m, h, V, rho = 32, 10_000, 4, 256, false, 1e-4
  X, _, B = generate_random_dataset(Float64, Int16, d, n, m, h)

  # These two methods are equivalent, but the second should be faster
  C1, _ = Rayuela.update_codebooks_chain(X, B, h, V)
  C2, _ = Rayuela.update_codebooks_chain_bin(X, B, h, V, rho)
  @test isapprox(C1, C2)
end


# Chain quantization
@testset "Chain encoding" begin
  d, n, m, h = 32, Int(1e3), 4, 256
  X, C, B = generate_random_dataset(Float32, Int16, d, n, m, h)

  B1, _ = Rayuela.quantize_chainq(X, C) # Julia

  use_cuda, use_cpp = true, false
  B2, _ = Rayuela.quantize_chainq(X, C, use_cuda, use_cpp) # Cuda

  use_cuda, use_cpp = false, true
  B3, _ = Rayuela.quantize_chainq(X, C, use_cuda, use_cpp) # C++
  @test all(B1 .== B2 .== B3) # C++ implememtation
end
