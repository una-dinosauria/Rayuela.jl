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
@testset "Viterbi encoding" begin
  d, n, m, h = 128, Int(1e4), 7, 256
  X, C, B = generate_random_dataset(Float32,Int16,d,n,m,h)

  Rayuela.quantize_chainq(X[:,1:100], C) # Julia
  Rayuela.quantize_chainq(X[:,1:100], C, true) # C
  @profile j_B, _ = Rayuela.quantize_chainq(X, C) # Julia

  @time j_B, _ = Rayuela.quantize_chainq(X, C) # Julia
  @time c_B, _ = Rayuela.quantize_chainq(X, C, true) # C


  @show j_B[:,1], c_B[:,1]
  @test all(j_B .== c_B)
end
