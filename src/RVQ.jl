
export train_rvq, quantize_rvq

"""
    quantize_rq(X::Matrix{T}, C::Vector{Matrix{T}}, V::Bool=false) where T <: AbstractFloat

Quantize using a residual quantizer
"""
function quantize_rvq(
  X::Matrix{T},         # d-by-n. Data to encode
  C::Vector{Matrix{T}}, # codebooks
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  d, n = size( X )
  m    = length( C )
  h    = size( C[1], 2 )
  B    = Vector{Vector{Int}}(m) # codes
  for i = 1:m; B[i] = zeros(Int,n); end # Allocate codes

  # auxiliary variables for update_assignments! function
  costs     = zeros(Float32, n)
  counts    = zeros(Int, h)
  to_update = zeros(Bool, h)
  unused    = Int[]

  # Residual after encoding the ith codebook
  Xr = copy(X)

  for i = 1:m
    if V print("Encoding on codebook $i / $m... ") end

    # Find distances from X to codebook
    dmat = Distances.pairwise( Distances.SqEuclidean(), C[i], Xr )

    # Update the codes
    Clustering.update_assignments!( dmat, true, B[i], costs, counts, to_update, unused )

    # Update the residual
    Xr .-= C[i][:,B[i]]

    if V println("done"); end
  end

  B = hcat(B...)
  B = convert(Matrix{Int16}, B)
  B'
end

"""
    train_rq(X::Matrix{T}, m::Integer, h::Integer, V::Bool=false) where T <: AbstractFloat

Trains a residual quantizer.
"""
function train_rvq(
  X::Matrix{T},  # d-by-n. Data to learn codebooks from
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of k-means iterations for training
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  d, n = size( X )

  C = Vector{Matrix{T}}(m) # codebooks
  B = zeros(Int16, n, m) # codes

  # Residual
  Xr = copy(X)

  for i = 1:m
    if V print("Working on codebook $i / $m... "); end
    # FAISS uses 25 iterations by default
    # See https://github.com/facebookresearch/faiss/blob/master/Clustering.cpp#L28
    cluster = kmeans(Xr, h, init=:kmpp, maxiter=niter)
    C[i], B[:,i] = cluster.centers, cluster.assignments

    # Update the residual
    Xr .-= C[i][:,B[:,i]]

    if V
      subdim_cost = cluster.totalcost ./ n
      nits        = cluster.iterations
      converged   = cluster.converged

      println("done.")
      println("  Ran for $nits iterations")
      println("  Error after $(m)-codebook is $subdim_cost")
      println("  Converged: $converged")
    end
  end
  B = B'
  error = qerror(X, B, C)
  return C, B, error
end
