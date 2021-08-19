
export train_rvq, quantize_rvq, experiment_rvq

"""
    quantize_rvq(X, C, V=false) -> B, singletons

Given data and full-dimensional codebooks, quantize.

# Arguments
- `X::Matrix{T}`: `d`-by-`n` data to quantize
- `C::Vector{Matrix{T}}`: `m`-long vector with `d`-by-`h` matrix entries. Each matrix is a codebook.
- `V::Bool`: Whether to print progress

# Returns
- `B::Matrix{Int16}`: `m`-by-`n` codes that approximate `X`
- `singletons::Vector{Matrix{T}}`: `m` matrices with unused codebook entries
"""
function quantize_rvq(
  X::Matrix{T},         # d-by-n. Data to encode
  C::Vector{Matrix{T}}, # codebooks
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  d, n = size( X )
  m    = length( C )
  h    = size( C[1], 2 )
  B    = Vector{Vector{Int}}(undef, m) # codes
  for i = 1:m; B[i] = zeros(Int,n); end # Allocate codes

  singletons = Vector{Matrix{T}}(undef, m)

  # Residual after encoding the ith codebook
  Xr = copy(X)

  for i = 1:m
    if V print("RVQ encoding on codebook $i / $m... ") end

    # Find distances from X to codebook
    dmat = Distances.pairwise(Distances.SqEuclidean(), C[i], Xr)

    # Update the codes
    costs     = zeros(T, n)    # Out. The cost of assignment
    counts    = zeros(Int, h)  # Out. Number of points assigned to this centre
    to_update = ones(Bool, h)  # Out. Whether this centre needs update
    unused    = Vector{Int}()  # Out. Indices of centres that have no assignments

    Clustering.update_assignments!(dmat, true, B[i], costs, counts, to_update, unused)

    # Create new codebook entries that we are missing
    if !isempty(unused)
      temp_codebook = similar(C[i]) # Out. The new codebooks will be added here
      Clustering.repick_unused_centers(Xr, costs, temp_codebook, unused, Distances.SqEuclidean())
      singletons[i] = temp_codebook[:, unused]
    end

    # Update the residual
    Xr .-= C[i][:, B[i]]

    if V println("done"); end
  end

  B = hcat(B...)
  B = convert(Matrix{Int16}, B)
  B = collect(B')

  B, singletons
end

"""
    train_rvq(X, m, h, niter=25, V=false) -> C, B, error

Train a residual quantizer.

# Arguments
- `X::Matrix{T}`: `d`-by-`n` data to quantize
- `m::Integer`: Number of codebooks
- `h::Integer`: Number of entries in each codebook (typically 256)
- `niter::Integer`: Number of iterations to use
- `V::Bool`: Whether to print progress

# Returns
- `C::Vector{Matrix{T}}`: `m`-long vector with `d`-by-`h` matrix entries. Each matrix is a codebook of size approximately `d/m`-by-`h`.
- `B::Matrix{Int16}`: `m`-by-`n` matrix with the codes
- `error::T`: The quantization error after training
"""
function train_rvq(
  X::Matrix{T},       # d-by-n. Data to learn codebooks from
  m::Integer,         # number of codebooks
  h::Integer,         # number of entries per codebook
  niter::Integer=25,  # Number of k-means iterations for training
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  d, n = size( X )

  C = Vector{Matrix{T}}(undef, m) # codebooks
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
      println("done.")
      println("  Ran for $(cluster.iterations) iterations")
      println("  Error after codebook $(i) is $(cluster.totalcost ./ n)")
      println("  Converged: $(cluster.converged)")
    end
  end

  B = collect(B')
  error = qerror(X, B, C)

  C, B, error
end


function experiment_rvq(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  Xb::Matrix{T}, # d-by-n. Base set
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of k-means iterations for training
  knn::Integer=1000,
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  # === RVQ train ===
  d, _ = size(Xt)
  C, B, train_error = Rayuela.train_rvq(Xt, m, h, niter, V)
	norms_B, norms_C = get_norms_codebook(B, C)

  # === Encode the base set ===
  B_base, _ = Rayuela.quantize_rvq(Xb, C, V)
  base_error = qerror(Xb, B_base, C)
  if V; @printf("Error in base is %e\n", base_error); end

  # Compute and quantize the database norms
  B_base_norms, _ = quantize_norms(B_base, C, norms_C)
  db_norms        = vec(norms_C[ B_base_norms ])

  # === Compute recall ===
  # B_base       = convert(Matrix{UInt8}, B_base-1)
  # B_base_norms = convert(Vector{UInt8}, B_base_norms-1)

  if V; print("Querying m=$m ... "); end
  @time dists, idx = linscan_lsq(B_base, Xq, C, db_norms, Matrix{Float32}(1.0I, d, d), knn)
  if V; println("done"); end

  recall = eval_recall(gt, idx, knn)
  return C, B, train_error, B_base, recall
end


function experiment_rvq_query_base(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of k-means iterations for training
  knn::Integer=1000,
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  # === RVQ train ===
  d, _ = size(Xt)
  C, B, train_error = Rayuela.train_rvq(Xt, m, h, niter, V)
	norms_B, norms_C = get_norms_codebook(B, C)
  db_norms     = vec(norms_C[ norms_B ])

  # === Compute recall ===
  # B_base       = convert(Matrix{UInt8}, B_base-1)
  # B_base_norms = convert(Vector{UInt8}, B_base_norms-1)
  if V; print("Querying m=$m ... "); end
  @time dists, idx = linscan_lsq(B, Xq, C, db_norms, Matrix{Float32}(1.0I, d, d), knn)
  if V; println("done"); end

  recall = eval_recall(gt, idx, knn)
  return C, B, train_error, recall
end
