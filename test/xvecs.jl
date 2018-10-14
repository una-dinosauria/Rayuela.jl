
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
