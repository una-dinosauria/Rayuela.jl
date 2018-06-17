
### Stochastic relaxations for LSQ

function train_sr(
  X::Matrix,                  # d-by-n matrix of data points to train on.
  m::Integer,                 # number of codebooks
  h::Integer,                 # number of entries per codebook
  R::Matrix{Float32},         # init rotation
  B::Matrix{Int16},     # init codes
  C::Vector{Matrix{Float32}}, # init codebooks
  niter::Integer,             # number of optimization iterations
  ilsiter::Integer,           # number of ILS iterations to use during encoding
  icmiter::Integer,           # number of iterations in local search
  randord::Bool,              # whether to use random order
  npert::Integer,             # The number of codes to perturb
  method::AbstractString,     # The SR method to use. Either SR_C or SR_D
  p::Float32,                 # SR-D power parameter
  cpp::Bool=true,             # whether to use ICM's cpp implementation
  V::Bool=false)              # whether to print progress

  if V
  println("**********************************************************************************************")
  println("Doing local search with $m codebooks, $npert perturbations, $icmiter icm iterations and random order = $randord");
  println("**********************************************************************************************")
  end

  if !(method in ["SR_C", "SR_D"]); error("SR method unknown"); end
  d, n = size( X )

  RX = R' * X
  if V; @printf("Random error: %e\n", qerror( RX, B, C )); end

  if method == "SR_C"
    # In SR-C we add noise to X
    RX_noisy = SR_C_perturb( RX, 0, niter, p )
    C = update_codebooks_fast_bin( RX_noisy, B, h, V )
  else
    # In SR-D we add noise to C
    C = update_codebooks_fast_bin( RX, B, h, V )
    obj = qerror( RX, B, C )
    if V; @printf("%3d %e \n", -1, obj); end
    C = SR_D_perturb( C, 1, niter, p )
  end

  obj = qerror( RX, B, C );
  if V; @printf("%3d %e \n", -1, obj); end

  # Initialize B
  @time B = encoding_icm( RX, B, C, ilsiter, icmiter, randord, npert, cpp, V )
  obj = qerror( RX, B, C )
  @printf("%3d %e \n", -1, obj)

  obj     = Inf
  objlast = Inf
  objarray = zeros( Float32, niter+1 )

  for iter = 1:niter

    objlast = obj
    obj = qerror( RX, B, C  )
    objarray[iter] = obj
    if V; @printf("%3d %e (%e better) \n", iter, obj, objlast - obj); end

    if method == "SR_C"
      # In SR-C we add noise to X
      RX_noisy = SR_C_perturb( RX, iter, niter, p )
      C = update_codebooks_fast_bin( RX_noisy, B, h, V )
    else
      C = update_codebooks_fast_bin( RX, B, h, V )
      C = SR_D_perturb( C, iter, niter, p )
    end

    # Update the codes with local search
    @time B = encoding_icm( RX, B, C, ilsiter, icmiter, randord, npert, cpp, V )

  end

  objarray[niter+1] = qerror( RX, B, C )

  # Apply the rotation to the codebooks
  for i = 1:m; C[i] = R * C[i]; end

  return C, B, objarray
end

# TODO this should be doable with metaprogramming

function train_sr_cuda(
  X::Matrix,                  # d-by-n matrix of data points to train on.
  m::Integer,                 # number of codebooks
  h::Integer,                 # number of entries per codebook
  R::Matrix{Float32},         # init rotation
  B::Matrix{Int16},     # init codes
  C::Vector{Matrix{Float32}}, # init codebooks
  niter::Integer,             # number of optimization iterations
  ilsiter::Integer,           # number of ILS iterations to use during encoding
  icmiter::Integer,           # number of iterations in local search
  randord::Bool,              # whether to use random order
  npert::Integer,             # The number of codes to perturb
  method::AbstractString,     # The SR method to use. Either SR_C or SR_D
  schedule::Integer,          # 1-3 Schedule to use
  p::AbstractFloat=0.5,       # Temperature decay parameter
  nsplits::Integer=1,         # The number of splits for icm encoding (for limited memory GPUs)
  V::Bool=false)              # whether to print progress

  if V
  println("**********************************************************************************************");
  println("Doing local search with $m codebooks, $npert perturbations, $icmiter icm iterations and random order = $randord");
  println("**********************************************************************************************");
  end

  if !(method in ["SR_C", "SR_D"]); error("SR method unknown"); end
  d, n = size( X )

  RX = R' * X
  if V; @printf("Random error: %e\n", qerror(RX, B, C)); end

  if method == "SR_C"
    # In SR-C we add noise to X
    RX_noisy = SR_C_perturb(RX, 0, niter, schedule, p)
    C = update_codebooks_fast_bin(RX_noisy, B, h, V)
  else
    # In SR-D we add noise to C
    C = update_codebooks_fast_bin(RX, B, h, V)
    obj = qerror(RX, B, C)
    if V; @printf("%3d %e \n", -1, obj); end
    C = SR_D_perturb(C, 1, niter, schedule, p)
  end

  obj = qerror(RX, B, C)
  if V; @printf("%3d %e \n", -1, obj); end

  # Initialize B
  @time B, _ = encode_icm_cuda(RX, B, C, [ilsiter], icmiter, npert, randord, nsplits, V)
  # gc()
  B = B[end]
  obj = qerror( RX, B, C )
  @printf("%3d %e \n", -1, obj)

  obj     = Inf
  objlast = Inf
  objarray = zeros(Float32, niter+1)

  for iter = 1:niter

    objlast = obj
    obj = qerror(RX, B, C)
    objarray[iter] = obj
    if V; @printf("%3d %e (%e better) \n", iter, obj, objlast - obj); end

    if method == "SR_C"
      # In SR-C we add noise to X
      RX_noisy = SR_C_perturb(RX, iter, niter, schedule, p)
      C = update_codebooks_fast_bin(RX_noisy, B, h, V)
    else
      C = update_codebooks_fast_bin(RX, B, h, V)
      C = SR_D_perturb(C, iter, niter, schedule, p)
    end

    # Update the codes with local search
    @time B, _ = encode_icm_cuda(RX, B, C, [ilsiter], icmiter, npert, randord, nsplits, V)
    # gc()
    B = B[end]

    C = update_codebooks_fast_bin(RX, B, h, V)
  end

  objarray[niter+1] = qerror(RX, B, C)

  # Apply the rotation to the codebooks
  for i = 1:m; C[i] = R * C[i]; end

  return C, B, objarray
end


function experiment_sr(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  B::Matrix{T2}, # codes
  C::Vector{Matrix{T}}, # codebooks
  R::Matrix{T}, # rotation
  Xb::Matrix{T}, # d-by-n. Base set
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of C-B update updates
  ilsiter::Integer=8, # number of ILS iterations
  icmiter::Integer=4, # number of ICM iterations
  randord::Bool=true, # whether to explore the codes in random order
  npert::Integer=4, # number of codes to perturb in each ILS iterations
  knn::Integer=1000,
  sr_method::String="SR_D",
  schedule::Integer=1, # schedule to use
  p::AbstractFloat=0.5, # Temperature decay parameter
  V::Bool=false) where {T <: AbstractFloat, T2 <: Integer} # whether to print progress

  # TODO maybe epose this? cpp is always faster
  cpp     = true

  # Train LSQ
  d, _ = size(Xt)
  C, B, obj = train_sr(Xt, m, h, R, B, C, niter, ilsiter, icmiter, randord, npert, sr_method, schbedule, p, cpp, V)
  norms_B, norms_C = get_norms_codebook(B, C)

  # === Encode the base set ===
  B_base = convert(Matrix{Int16}, rand(1:h, m, size(Xb,2)))
  B_base = encoding_icm(Xb, B_base, C, ilsiter, icmiter, randord, npert, cpp, V)
  base_error = qerror(Xb, B_base, C)
  if V; @printf("Error in base is %e\n", base_error); end

  # Compute and quantize the database norms
  B_base_norms, db_norms_X = quantize_norms(B_base, C, norms_C)
  db_norms     = vec( norms_C[ B_base_norms ] )

  if V; print("Querying m=$m ... "); end
  @time dists, idx = linscan_lsq(B_base, Xq, C, db_norms, eye(Float32, d), knn)
  if V; println("done"); end

  rec = eval_recall(gt, idx, knn)
end

"Runs an lsq experiment/demo"
function experiment_sr(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  Xb::Matrix{T}, # d-by-n. Base set
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of k-means iterations for training
  knn::Integer=1000,
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  # OPQ initialization
  C, B, R, _ = train_opq(Xt, m, h, niter, "natural", V)

  # ChainQ (second initialization)
  # C, B, R, train_error = train_chainq(Xt, m, h, R, B, C, niter, V)

  # Actual experiment
  experiment_sr(Xt, B, C, R, Xb, Xq, gt, m, h, niter, knn, V)
end


function experiment_sr_cuda(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  B::Matrix{T2}, # codes
  C::Vector{Matrix{T}}, # codebooks
  R::Matrix{T}, # rotation
  Xb::Matrix{T}, # d-by-n. Base set
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of C-B update updates
  ilsiter::Integer=8, # number of ILS iterations
  icmiter::Integer=4, # number of ICM iterations
  randord::Bool=true, # whether to explore the codes in random order
  npert::Integer=4, # number of codes to perturb in each ILS iterations
  knn::Integer=1000,
  nsplits_train::Integer=1,
  nsplits_base::Integer=1, # Number of splits for training data so GPU does not run out of memory
  sr_method::String="SR_D",
  schedule::Integer=1, # schedule to use
  p::AbstractFloat=0.5, # Temperature decay parameter
  V::Bool=false) where {T <: AbstractFloat, T2 <: Integer} # whether to print progress

  # Train LSQ
  d, _ = size(Xt)
  C, B, train_error = train_sr_cuda(Xt, m, h, R, B, C, niter, ilsiter, icmiter, randord, npert, sr_method, schedule, p, nsplits_train, V)
  norms_B, norms_C = get_norms_codebook(B, C)

  # === Encode the base set ===
  B_base = convert(Matrix{Int16}, rand(1:h, m, size(Xb,2)))

  # ilsiters = [16, 32, 64, 128, 256]
  ilsiters = [ilsiter * 4]
  Bs_base, _ = encode_icm_cuda(Xb, B_base, C, ilsiters, icmiter, npert, randord, nsplits_base, V)

  # for (idx, ilsiter) in enumerate(ilsiters)
    B_base = Bs_base[end]
    base_error = qerror(Xb, B_base, C)
    if V; @printf("Error in base is %e\n", base_error); end

    # Compute and quantize the database norms
    B_base_norms, db_norms_X = quantize_norms(B_base, C, norms_C)
    db_norms = vec( norms_C[ B_base_norms ] )

    if V; print("Querying m=$m ... "); end
    # @time dists, idx = linscan_lsq(B_base, Xq, C, db_norms_X, eye(Float32, d), knn)
    @time dists, idx = linscan_lsq(B_base, Xq, C, db_norms, eye(Float32, d), knn)
    if V; println("done"); end

    recall = eval_recall(gt, idx, knn)
  # end
    return C, B, R, train_error, B_base, recall
end

function experiment_sr_cuda_query_base(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  B::Matrix{T2}, # codes
  C::Vector{Matrix{T}}, # codebooks
  R::Matrix{T}, # rotation
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of C-B update updates
  ilsiter::Integer=8, # number of ILS iterations
  icmiter::Integer=4, # number of ICM iterations
  randord::Bool=true, # whether to explore the codes in random order
  npert::Integer=4, # number of codes to perturb in each ILS iterations
  knn::Integer=1000,
  nsplits_train::Integer=1,
  sr_method::String="SR_D",
  schedule::Integer=1, # schedule to use
  p::AbstractFloat=0.5, # Temperature decay parameter
  V::Bool=false) where {T <: AbstractFloat, T2 <: Integer} # whether to print progress

  # Train LSQ
  d, _ = size(Xt)
  if V; print("Running CUDA $(sr_method) training... "); end
  C, B, train_error = train_sr_cuda(Xt, m, h, R, B, C, niter, ilsiter, icmiter, randord, npert, sr_method, schedule, p, nsplits_train, V)
  if V; @printf("done\n"); end

  norms_B, norms_C = get_norms_codebook(B, C)
  db_norms         = vec(norms_C[norms_B])

  if V; print("Querying m=$m ... "); end
  @time dists, idx = linscan_lsq(B, Xq, C, db_norms, eye(Float32, d), knn)
  if V; println("done"); end

  recall = eval_recall(gt, idx, knn)
  return C, B, R, train_error, recall
end


"Runs an lsq experiment/demo"
function experiment_sr_cuda(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  Xb::Matrix{T}, # d-by-n. Base set
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of k-means iterations for training
  knn::Integer=1000,
  nsplits_train::Integer=1,
  nsplits_base::Integer=1,
  sr_method::String="SR_D",
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  # OPQ initialization
  if V; @printf("Running OPQ initialization... "); end
  niter_opq = 25
  C, B, R, _ = train_opq(Xt, m, h, niter_opq, "natural", V)
  if V; @printf("done\n"); end

  # ChainQ (second initialization)
  # C, B, R, train_error = train_chainq(Xt, m, h, R, B, C, niter, V)

  # Actual experiment
  experiment_sr_cuda(Xt, B, C, R, Xb, Xq, gt, m, h, niter, knn, nsplits_train, nsplits_base, sr_method, V)
end

"Runs an lsq experiment/demo"
function experiment_sr_cuda_query_base(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of k-means iterations for training
  knn::Integer=1000,
  nsplits_train::Integer=1,
  sr_method::String="SR_D",
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  # OPQ initialization
  if V; @printf("Running OPQ initialization... "); end
  niter_opq = 25
  C, B, R, opq_error = train_opq(Xt, m, h, niter_opq, "natural", V)
  if V; @printf("done\n"); end

  # ChainQ (second initialization)
  if V; @printf("Running ChainQ initialization... "); end
  niter_chainq = 25
  C, B, R, chainq_error = train_chainq(Xt, m, h, R, B, C, niter_chainq, V)
  if V; @printf("done\n"); end

  # Actual experiment
  experiment_sr_cuda_query_base(Xt, B, C, R, Xq, gt, m, h, niter, knn, nsplits_train, sr_method, V), opq_error
end
