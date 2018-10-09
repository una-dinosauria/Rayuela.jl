
export train_pq, quantize_pq


"""
    quantize_pq(X, C, V=false) -> B

Given data and PQ codeboks, quantize.

# Arguments
- `X::Matrix{T}`: `d`-by-`n` data to quantize
- `C::Vector{Matrix{T}}`: `m`-long vector with `d`-by-`h` matrix entries. Each matrix is a PQ codebook.
- `V::Bool`: Whether to print progress

# Returns
- `B::Matrix{Int16}`: `m`-by-`n` matrix with the codes that approximate `X`
"""
function quantize_pq(
  X::Matrix{T},
  C::Vector{Matrix{T}},
  V::Bool=false) where T <: AbstractFloat

  d, n = size(X)
  m    = length(C)
  h    = size(C[1], 2)
  B    = Vector{Vector{Int}}(undef, m) # codes
  for i = 1:m; B[i] = zeros(Int, n) end # Allocate codes

  subdims = splitarray(1:d, m)

  # auxiliary variables for update_assignments! function
  costs     = zeros(Float32, n)
  counts    = zeros(Int, h)
  to_update = zeros(Bool, h)
  unused    = Int[]

  for i = 1:m
    if V print("Encoding on codebook $i / $m... ") end
    # Find distances from X to codebook
    dmat = Distances.pairwise(Distances.SqEuclidean(), C[i], X[subdims[i],:])
    Clustering.update_assignments!(dmat, true, B[i], costs, counts, to_update, unused)
    if V println("done"); end
  end

  B = hcat(B...)
  B = convert(Matrix{Int16}, B)
  collect(B')
end


"""
    train_pq(X, m, h, niter=25, V=false) -> C, B, error

Trains a product quantizer.

# Arguments
- `X::Matrix{T}`: `d`-by-`n` data to quantize
- `m::Integer`: Number of codebooks
- `h::Integer`: Number of entries in each codebook (typically 256)
- `niter::Integer`: Number of k-means iterations to use
- `V::Bool`: Whether to print progress

# Returns
- `B::Matrix{Int16}`: `m`-by-`n` matrix with the codes
- `C::Vector{Matrix{T}}`: `m`-long vector with `d`-by-`h` matrix entries. Each matrix is a codebook of size approximately `d/m`-by-`h`.
- `error::T`: The quantization error after training
"""
function train_pq(
  X::Matrix{T},
  m::Integer,
  h::Integer,
  niter::Integer=25,
  V::Bool=false) where T <: AbstractFloat

  d, n = size(X)

  C = Vector{Matrix{T}}(undef, m) # codebooks

  B       = zeros(Int16, n, m) # codes
  subdims = splitarray(1:d, m) # subspaces

  for i = 1:m
    if V print("Working on codebook $i / $m... "); end
    # FAISS uses 25 iterations by default
    # https://github.com/facebookresearch/faiss/blob/master/Clustering.cpp#L28
    cluster = kmeans(X[ subdims[i],: ], h, init=:kmpp, maxiter=niter)
    C[i], B[:,i] = cluster.centers, cluster.assignments

    if V
      println("done.")
      println("  Ran for $(cluster.iterations) iterations")
      println("  Error in subspace is $(cluster.totalcost ./ n)")
      println("  Converged: $(cluster.converged)")
    end
  end
  B = collect(B')
  error = qerror_pq(X, B, C)
  return C, B, error
end

"""
  Run a complete experiment: train, encode base and compute recall.
"""
function experiment_pq(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  Xb::Matrix{T}, # d-by-n. Base set
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of k-means iterations for training
  knn::Integer=1000,
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  # === Train ===
  C, B, train_error = train_pq(Xt, m, h, niter, V)
  if V; @printf("Error in training is %e\n", train_error); end

  # === Encode the base set ===
  B_base     = quantize_pq(Xb, C, V)
  base_error = qerror_pq(Xb, B_base, C)
  if V; @printf("Error in base is %e\n", base_error); end

  # === Compute recall ===
  if V; println("Querying m=$m ... "); end
  b = Int(log2(h) * m)
  @time dists, idx = linscan_pq(B_base, Xq, C, b, knn)
  if V; println("done"); end

  recall = eval_recall(gt, idx, knn)
  return C, B, train_error, B_base, recall
end

"""
  Run a complete experiment where the training set and base set are the same.
"""
function experiment_pq_query_base(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of k-means iterations for training
  knn::Integer=1000,
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  # === Train ===
  C, B, train_error = train_pq(Xt, m, h, niter, V)
  if V; @printf("Error in training is %e\n", train_error); end

  # === Compute recall ===
  if V; println("Querying m=$m ... "); end
  b = Int(log2(h) * m)
  @time dists, idx = linscan_pq(B, Xq, C, b, knn)
  if V; println("done"); end

  recall = eval_recall(gt, idx, knn)
  return C, B, train_error, recall
end
