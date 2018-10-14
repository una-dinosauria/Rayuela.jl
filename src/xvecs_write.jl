

# export bvecs_read, fvecs_read, ivecs_read
export fvecs_write, ivecs_write

# TODO bvecs_write
# TODO refactor these with metaprogrammming

"Write a dataset in fvecs format"
function fvecs_write(X::Matrix{Float32}, filename::AbstractString)
  d, n = size(X)
  Xd = Matrix{Float32}(undef, d+1, n)
  Xd[1, :] .= reinterpret(Float32, Int32(d))
  Xd[2:end, :] = X
  write(filename, Xd)
end

"Write a dataset in ivecs format"
function ivecs_write(X::Matrix{Int32}, filename::AbstractString)
  d, n = size(X)
  Xd = Matrix{Int32}(undef, d+1, n)
  Xd[1, :] .= Int32(d)
  Xd[2:end, :] = X
  write(filename, Xd)
end
