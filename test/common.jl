
# Generate random data for tests. d is the size of the dataset.
function generate_random_dataset(T1, T2, d, n, m, h)
  X = rand(T1, d, n) * 10
  C = Vector{Matrix{T1}}(undef, m)
  for i=1:m; C[i]=rand(T1, d, h); end
  B = convert(Matrix{T2},rand(1:h, m, n))
  X, C, B
end
