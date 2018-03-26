
using Rayuela
using Base.Test

# Generate random data for tests. d is the size of the dataset.
function generate_random_dataset(T1, T2, d, n, m, h)
  X = rand(T1, d, n)
  C = Vector{Matrix{T1}}(m)
  for i=1:m; C[i]=rand(T1,d,h); end
  B = convert(Matrix{T2},rand(1:h,m,n))
  X, C, B
end

# Test cpp viterbi encoding implementation
# @testset "Viterbi encoding" begin
#   d, n, m, h = 32, Int(1e3), 4, 256
#   X, C, B = generate_random_dataset(Float32, Int16, d, n, m, h)
#
#   Bj, _ = Rayuela.quantize_chainq(X, C) # Julia
#   Bc, _ = Rayuela.quantize_chainq(X, C, true) # C
#   @test all(Bj .== Bc)
# end

# xvecs_read and xvecs_write
@testset "xvecs" begin
  fn = tempname()
  d, n, m, h = 32, Int(1e3), 4, 256
  X, C, B = generate_random_dataset(Float32, Int16, d, n, m, h)

  # fvecs
  fvecs_write(X, fn)
  X2 = fvecs_read(n, fn)
  @test all(X .== X2)
  rm(fn)

  Xint = convert(Matrix{Int32}, floor.(X.-0.5f0)*1000)
  ivecs_write(Xint, fn)
  Xint2 = ivecs_read(n, fn)
  @test all(Xint .== Xint2)
  rm(fn)
end
