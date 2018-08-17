
# Optimized Product Quantization. Adapted from Mohammad Norouzi's code.
export train_opq, quantize_opq

function quantize_opq(
  X::Matrix{T},          # d-by-n matrix of data points to quantize
  R::Matrix{T},          # d-by-d matrix. Learned rotation for X
  C::Vector{Matrix{T}},  # m-long array. Each entry is a d-by-h codebooks
  V::Bool=false) where T <: AbstractFloat

  # Apply rotation and quantize as in PQ
  return quantize_pq(R'*X, C, V)
end

function train_opq(
  X::Matrix{T},      # d-by-n matrix of data points to train on.
  m::Integer,        # number of codebooks
  h::Integer,        # number of entries per codebook
  niter::Integer,    # number of optimization iterations
  init::String,      # how to initialize the optimization
  V::Bool=false) where T <: AbstractFloat

  if V; @printf("Training an optimized product quantizer\n"); end

  d, n = size( X )

  C = Vector{Matrix{T}}(m) # codebooks

  obj = zeros( Float32, niter+1 )

  # Number of bits in the final codes.
  nbits = log2(h) * m
  CB    = zeros(T, size(X))

  if init == "natural" # Initialize R with identity
    R = eye(T, d)
  elseif init == "random"
    R, _, _ = svd( randn( T, d, d ))
  else
    error("Intialization $init unknown")
  end

  RX = R' * X # Rotate the data

  subdims = splitarray( 1:d, m )

  # Initialization sampling RX
  for i = 1:m
    perm = sample(1:n, h, replace=false)
    C[i] = RX[ subdims[i], perm ]
  end

  # Variables needed for methods in Clustering.kmeans
  costs      = zeros(Float32, n)
  counts     = zeros(Int, h)
  cweights   = zeros(Float32, h)
  to_update  = zeros(Bool, h)
  to_update2 = ones(Bool, h)
  unused     = Int[]

  # Initialize the codes -- B
  B    = Vector{Vector{Int}}(m) # codes
  for i = 1:m; B[i] = zeros(Int, n); end # Allocate codes

  for i=1:m
    dmat = Distances.pairwise(Distances.SqEuclidean(), C[i], RX[subdims[i],:])
    Clustering.update_assignments!(dmat, true, B[i], costs, counts, to_update, unused)
    CB[subdims[i],:] .= C[i][:, B[i]]
  end

  for iter=0:niter
    if V; tic(); end # Take time if asked to

    # Compute objective function
    obj[iter+1] = sum( (R*CB - X).^2 ) ./ n
    if V; @printf("%3d %e... ", iter, obj[iter+1]); end

    # update R
    U, S, VV = svd(X * CB', thin=true)
    R = U * VV'

    # update R*X
    RX = R' * X

    for i=1:m
      # update C
      Clustering.update_centers!(RX[subdims[i],:], nothing, B[i], to_update2, C[i], cweights)

      # update B
      dmat = Distances.pairwise(Distances.SqEuclidean(), C[i], RX[subdims[i],:])
      Clustering.update_assignments!(dmat, false, B[i], costs, counts, to_update, unused)

      # update D*B
      CB[subdims[i], :] .= C[i][:, B[i]]
    end # for i=1:m

    if V; @printf("done in %.2f secs\n", toq()); end
  end # for iter=0:niter

  B = hcat(B...)
  B = convert(Matrix{Int16}, B)
  C = convert(Vector{Matrix{T}}, C)

  return C, B', R, obj
end


function experiment_opq(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  Xb::Matrix{T}, # d-by-n. Base set
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  init::String="natural", # initialization method for the rotation
  niter::Integer=25, # Number of k-means iterations for training
  knn::Integer=1000,
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  # === Train ===
  C, B, R, train_error = train_opq(Xt, m, h, niter, init, V)
  if V; @printf("Error in training is %e\n", train_error[end]); end

  # === Encode the base set ===
  B_base     = quantize_opq(Xb, R, C, V)
  base_error = qerror_opq(Xb, B_base, C, R)
  if V; @printf("Error in base is %e\n", base_error); end

  # === Compute recall ===
  if V; println("Querying m=$m ... "); end
  b = Int(log2(h) * m)
  @time dists, idx = linscan_opq(B_base, Xq, C, b, R, knn)
  if V; println("done"); end

  recall = eval_recall(gt, idx, knn)
  return C, B, R, train_error, B_base, recall
end


function experiment_opq_query_base(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  init::String="natural", # initialization method for the rotation
  niter::Integer=25, # Number of k-means iterations for training
  knn::Integer=1000,
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  # === Train ===
  C, B, R, train_error = train_opq(Xt, m, h, niter, init, V)
  if V; @printf("Error in training is %e\n", train_error[end]); end

  # === Compute recall ===
  if V; println("Querying m=$m ... "); end
  b = Int(log2(h) * m)
  @time dists, idx = linscan_opq(B, Xq, C, b, R, knn)
  if V; println("done"); end

  recall = eval_recall(gt, idx, knn)
  return C, B, R, train_error, recall
end
