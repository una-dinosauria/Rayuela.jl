time_ns
export encode_icm_cuda, train_lsq_cuda

"Encodes a database with ILS in cuda"
function encode_icm_cuda_single(
  RX::Matrix{Float32},         # in. The data to encode
  B::Matrix{Int16},            # in. Initial list of codes
  C::Vector{Matrix{Float32}},  # in. Codebooks
  ilsiters::Vector{Int64},     # in. ILS iterations to record Bs and obj function. Its max is the total number of iteration we will run.
  icmiter::Integer,            # in. Number of ICM iterations
  npert::Integer,              # in. Number of entries to perturb
  randord::Bool,               # in. Whether to randomize the order in which nodes are visited in ILS
  V::Bool=false)               # in. Whether to print progress

  d, n = size( RX )
  m    = length( C )
  _, h = size( C[1] )

  # Number of results to keep track of
  nr   = length( ilsiters )

  # Make space for the outputs
  Bs   = Vector{Matrix{Int16}}(undef, nr)
  objs = Vector{Float32}(undef, nr)

  # === Compute binary terms (products between all codebook pairs) ===
  binaries, cbi = get_binaries(C)
  _, ncbi       = size(cbi)

  # Create a transposed copy of the binaries for cache-friendliness
  binaries_t = similar(binaries)
  for j = 1:ncbi
    binaries_t[j] = collect(binaries[j]')
  end

  # Create an index from codebook pairs to indices
  cbpair2binaryidx = zeros(Int32, m, m)
  for j = 1:ncbi
    cbpair2binaryidx[ cbi[1,j], cbi[2,j] ] = j
  end

  dev = CuDevice(0)
  ctx = CuContext(dev)

  # Initialize the cuda module, and choose the GPU
  gpuid = 0
  CudaUtilsModule.init( gpuid, cudautilsptx )
  # CUDArt.device( devlist[gpuid] )

  # === Create a state for random number generation ===
  if V; @printf("Creating %d random states... ", n); start_time = time_ns(); end
  d_state = CUDAdrv.Mem.alloc(n*64)
  CudaUtilsModule.setup_kernel(cld(n, 1024), 1024, Cint(n), d_state)

  CUDAdrv.synchronize(ctx)
  if V; @printf("done in %.2f seconds\n", (time_ns() - start_time)/1e9); end

  # Measure time for encoding
  if V; start_time = time_ns(); end

  # Copy X and C to the GPU
  d_RX = CuArrays.CuArray(RX)
  d_C  = CuArrays.CuArray(cat(C..., dims=2))

  # === Get unaries in the gpu ===
  d_prevcost = CuArrays.CuArray{Cfloat}(n)
  d_newcost  = CuArrays.CuArray{Cfloat}(n)

  # CUBLAS.cublasCreate_v2( CUBLAS.cublashandle )
  d_unaries   = Vector{CuArrays.CuArray{Float32}}(undef, m)
  d_codebooks = Vector{CuArrays.CuArray{Float32}}(undef, m)
  for j = 1:m
    d_codebooks[j] = CuArrays.CuArray(C[j])
    # -2 * C' * X
    d_unaries[j] = CuArrays.BLAS.gemm('T', 'N', -2.0f0, d_codebooks[j], d_RX)
    # d_unaries[j] = -2.0f0 * d_codebooks[j]' * d_RX <-- thus runs out of memory real fast

    # Add self-codebook interactions || C_{i,i} ||^2
    CudaUtilsModule.vec_add(n, (1,h), d_unaries[j].buf, CuArrays.CuArray(diag( C[j]' * C[j] )).buf, Cint(n), Cint(h))
  end

  # === Get binaries to the GPU ===
  d_binaries  = Vector{CuArrays.CuArray{Float32}}(undef, ncbi)
  d_binariest = Vector{CuArrays.CuArray{Float32}}(undef, ncbi)
  for j = 1:ncbi
    d_binaries[j]  = CuArrays.CuArray(binaries[j])
    d_binariest[j] = CuArrays.CuArray(binaries_t[j])
  end

  # Allocate space for temporary results
  bbs = Vector{Matrix{Cfloat}}(undef, m-1)

  # Initialize the previous cost
  prevcost = Rayuela.veccost(RX, B, C)
  IDX = 1:n;

  # For codebook i, we have to condition on these codebooks
  to_look      = 1:m
  to_condition = zeros(Int32, m-1, m)
  for j = 1:m
    tmp = collect(1:m)
    splice!( tmp, j )
    to_condition[:,j] = tmp
  end

  # Loop for the number of requested ILS iterations
  for i = 1:maximum(ilsiters)

    # @show i, ilsiters
    # @time begin

    to_look_r      = to_look
    to_condition_r = to_condition

    # Compute the cost of the previous assignments
    # CudaUtilsModule.veccost(n, (1, d), d_RX, d_C, CuArray( convert(Matrix{Cuchar}, (B')-1) ), d_prevcost, Cint(m), Cint(n))
    CudaUtilsModule.veccost2(n, (1, d), d_RX.buf, d_C.buf, CuArrays.CuArray( convert(Matrix{Cuchar}, collect(B').-1) ).buf, d_prevcost.buf, Cint(d), Cint(m), Cint(n))
    CUDAdrv.synchronize(ctx)
    prevcost = Array(d_prevcost)

    newB = copy(B)

    # Randomize the visit order in ICM
    if randord
      to_look_r      = randperm( m )
      to_condition_r = to_condition[:, to_look_r]
    end

    d_newB = CuArrays.CuArray(convert(Matrix{Cuchar}, newB.-1 ))

    # Perturn npert entries in each code
    CudaUtilsModule.perturb(n, (1,m), d_state, d_newB.buf, Cint(n), Cint(m), Cint(npert))

    newB = Array( d_newB )
    Bt   = collect(newB')
    d_Bs = CuArrays.CuArray( Bt )

    CUDAdrv.synchronize(ctx)

    # Run the number of ICM iterations requested
    for j = 1:icmiter

      kidx = 1;
      # Loop through each MRF node
      for k = to_look_r

        lidx = 1;
        for l = to_condition_r[:, kidx]
          # Determine the pairwise tables that we'll use in this conditioning
          if k < l
            binariidx = cbpair2binaryidx[ k, l ]
            bbs[lidx] = binaries[ binariidx ]
          else
            binariidx = cbpair2binaryidx[ l, k ]
            bbs[lidx] = binaries_t[ binariidx ]
          end
          lidx = lidx+1;
        end

        # Transfer pairwise tables to the GPU
        d_bbs = CuArrays.CuArray( convert(Matrix{Cfloat}, cat(bbs..., dims=2)) )
        # Sum binaries (condition) and minimize
        CudaUtilsModule.condition_icm3(
          n, (1, h),
          d_unaries[k].buf, d_bbs.buf, d_Bs.buf, Cint(k-1), Cint(m), Cint(n) )
        CUDAdrv.synchronize(ctx)

        kidx = kidx + 1;
      end # for k = to_look_r
    end # for j = 1:icmiter

    newB = Array( d_Bs )
    newB = convert(Matrix{Int16}, newB')
    newB .+= 1

    # Keep only the codes that improved
    CudaUtilsModule.veccost2(n, (1, d), d_RX.buf, d_C.buf, d_Bs.buf, d_newcost.buf, Cint(d), Cint(m), Cint(n))
    CUDAdrv.synchronize(ctx)

    newcost = Array( d_newcost )

    areequal = newcost .== prevcost
    if V; @printf(" ILS iteration %d/%d done. ", i, maximum(ilsiters)); end
    if V; @printf("%5.2f%% new codes are equal. ", 100*sum(areequal)/n ); end

    arebetter = newcost .< prevcost
    if V; @printf("%5.2f%% new codes are better.\n", 100*sum(arebetter)/n ); end

    newB[:, .~arebetter] = B[:, .~arebetter]
    B = copy( newB )
    # end # @time

    # Check if this # of iterations was requested
    if i in ilsiters

      ithidx = findall( i .== ilsiters )[1]

      # Compute and save the objective
      obj = qerror( RX, B, C )
      # @show obj
      objs[ ithidx ] = obj
      # @show size(B)
      Bs[ ithidx ] = B

    end # end if i in ilsiters
  end # end for i=1:max(ilsiters)

  # CUBLAS.cublasDestroy_v2( CUBLAS.cublashandle )
  CudaUtilsModule.finit()

  destroy!(ctx)
  # end # do devlist

  if V; @printf(" Encoding done in %.2f seconds\n", (time_ns() -start_time)/1e9); end

  return Bs, objs
end

function encode_icm_cuda(
  RX::Matrix{Float32},         # in. The data to encode
  B::Matrix{Int16},            # in. Initial list of codes
  C::Vector{Matrix{Float32}},  # in. Codebooks
  ilsiters::Vector{Int64},     # in. ILS iterations to record Bs and obj function. Its max is the total number of iteration we will run.
  icmiter::Integer,            # in. Number of ICM iterations
  npert::Integer,              # in. Number of entries to perturb
  randord::Bool,               # in. Whether to randomize the order in which nodes are visited in ILS
  nsplits::Integer=2,          # in. Number of splits of the data (for limited memory GPUs)
  V::Bool=false)

  # TODO check that splits >= 1
  if nsplits == 1
    Bs, objs =  encode_icm_cuda_single(RX, B, C, ilsiters, icmiter, npert, randord, V)
    GC.gc()
    return Bs, objs
  end

  # Split the data
  d, n = size(RX)
  splits = splitarray(1:n, nsplits)
  nr = length(ilsiters)

  Bs   = Vector{Matrix{Int16}}(undef, nr)
  objs = zeros(Float32, nr)

  # Create storage space for the codes
  for i = 1:nr; Bs[i] = Matrix{Int16}(undef, size(B)); end

  # Run encoding in the GPU for each split
  for i = 1:nsplits
    aaBs, _ = encode_icm_cuda_single(RX[:,splits[i]], B[:,splits[i]], C, ilsiters, icmiter, npert, randord, V)
    GC.gc()
    for j = 1:nr
      # Save the codes
      Bs[j][:,splits[i]] = aaBs[j]
    end
  end

  # Compute the cost again
  for j = 1:nr
    objs[j] = qerror(RX, Bs[j], C)
  end

  GC.gc()
  return Bs, objs
end

"LSQ training but some things happen in the GPU"
function train_lsq_cuda(
  X::Matrix{T},         # d-by-n matrix of data points to train on.
  m::Integer,           # number of codebooks
  h::Integer,           # number of entries per codebook
  R::Matrix{T},         # init rotation
  B::Matrix{Int16},     # init codes
  C::Vector{Matrix{T}}, # init codebooks
  niter::Integer,       # number of optimization iterations
  ilsiter::Integer,     # number of ILS iterations to use during encoding
  icmiter::Integer,     # number of iterations of ICM
  randord::Bool,        # whether to use random order
  npert::Integer,       # The number of codes to perturb
  nsplits::Integer=1,   # The number of splits for icm encoding (for limited memory GPUs)
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  # Xt, m, h, R, B, C, niter, ilsiter, icmiter, randord, npert, nsplits_train, V
  # @show B
  # @show size(B)
  # @show niter, ilsiter, icmiter, randord, npert, nsplits, V

  if V
    println()
    println("**********************************************************************************************");
    println("Doing local search with $m codebooks, $npert perturbations, $icmiter icm iterations and random order = $randord");
    println("**********************************************************************************************");
  end

  d, n = size( X )

  # Initialize C
  RX = R' * X
  # C = update_codebooks( RX, B, h, V, "lsqr" )
  C = update_codebooks_fast_bin( RX, B, h, V )
  # C = update_codebooks_fast( RX, B, h, V)

  # Apply the rotation to the codebooks
  for i = 1:m; C[i] = R * C[i]; end
  if V; @printf("%3d %e \n", -2, qerror( X, B, C )); end

  # Initialize B
  B, _ = encode_icm_cuda(X, B, C, [ilsiter], icmiter, npert, randord, nsplits, V)
  B    = B[end]
  if V; @printf("%3d %e \n", -1, qerror( X, B, C )); end

  obj = zeros( T, niter )

  for iter = 1:niter
    obj[iter] = qerror( X, B, C )
    if V; @printf("%3d %e \n", iter, obj[iter]); end

    # Update the codebooks
    # C = update_codebooks( X, B, h, V, "lsqr" )
    C = update_codebooks_fast_bin( X, B, h, V )

    # Update the codes B
    # B = convert(Matrix{Int16}, rand(1:h, m, n))
    B, _ = encode_icm_cuda(X, B, C, [ilsiter], icmiter, npert, randord, nsplits, V)
    B    = B[end]
  end

  return C, B, obj
end


function experiment_lsq_cuda(
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
  npert::Integer=4, # numebr of codes to perturb in each ILS iterations
  knn::Integer=1000, # Compute recall @N for this value of N
  nsplits_train::Integer=1, # Number of splits for training data so GPU does not run out of memory
  nsplits_base::Integer=1, # Number of splits for training data so GPU does not run out of memory
  V::Bool=false) where {T <: AbstractFloat, T2 <: Integer} # whether to print progress

  # Train LSQ
  d, _ = size(Xt)
  @printf("Running CUDA LSQ training... ")
  C, B, train_error = Rayuela.train_lsq_cuda(Xt, m, h, R, B, C, niter, ilsiter, icmiter, randord, npert, nsplits_train, V)
  @printf("done\n")

  norms_B, norms_C = get_norms_codebook(B, C)

  # === Encode the base set ===
  B_base = convert(Matrix{Int16}, rand(1:h, m, size(Xb,2)))
  Bs_base, _ = Rayuela.encode_icm_cuda(Xb, B_base, C, [ilsiter * 4], icmiter, npert, randord, nsplits_base, V)
  B_base = Bs_base[end]
  base_error = qerror(Xb, B_base, C)
  if V; @printf("Error in base is %e\n", base_error); end

  # Compute and quantize the database norms
  B_base_norms, db_norms_X = quantize_norms( B_base, C, norms_C )
  db_norms     = vec( norms_C[ B_base_norms ] )

  if V; print("Querying m=$m ... "); end
  @time dists, idx = linscan_lsq(B_base, Xq, C, db_norms, Matrix{Float32}(1.0I, d, d), knn)
  # @time dists, idx = linscan_lsq(B_base, Xq, C, db_norms_X, eye(Float32, d), knn)
  if V; println("done"); end

  recall = eval_recall(gt, idx, knn)
  return C, B, R, train_error, B_base, recall
end

function experiment_lsq_cuda_query_base(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  B::Matrix{T2}, # codes
  C::Vector{Matrix{T}}, # codebooks
  R::Matrix{T}, # rotation
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of C-B update updates
  ilsiter::Integer=8, # Number of ILS iterations
  icmiter::Integer=4, # Number of ICM iterations
  randord::Bool=true, # Whether to explore the codes in random order
  npert::Integer=4, # Number of codes to perturb in each ILS iterations
  knn::Integer=1000, # Compute recall @N for this value of N
  nsplits_train::Integer=1, # Number of splits for training data so GPU does not run out of memory
  V::Bool=false) where {T <: AbstractFloat, T2 <: Integer} # whether to print progress

  # Train LSQ
  d, _ = size(Xt)
  if V; @printf("Running CUDA LSQ training... "); end
  C, B, train_error = Rayuela.train_lsq_cuda(Xt, m, h, R, B, C, niter, ilsiter, icmiter, randord, npert, nsplits_train, V)
  if V; @printf("done\n"); end

  norms_B, norms_C = get_norms_codebook(B, C)
  db_norms         = vec(norms_C[ norms_B ])

  if V; print("Querying m=$m ... "); end
  @time dists, idx = linscan_lsq(B, Xq, C, db_norms, eye(Float32, d), knn)
  if V; println("done"); end

  recall = eval_recall(gt, idx, knn)
  return C, B, R, train_error, recall
end


"Runs an lsq experiment/demo"
function experiment_lsq_cuda(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  Xb::Matrix{T}, # d-by-n. Base set
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of C-B update updates
  ilsiter::Integer= 8, # number of ILS iterations
  icmiter::Integer=4, # number of ICM iterations
  randord::Bool=true, # whether to explore the codes in random order
  npert::Integer=4, # numebr of codes to perturb in each ILS iterations
  knn::Integer=1000,
  nsplits_train::Integer=1, # Number of splits for training data so GPU does not run out of memory
  nsplits_base::Integer=1, # Number of splits for training data so GPU does not run out of memory
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  # OPQ initialization
  @printf("Running OPQ initialization... ")
  C, B, R, _ = train_opq(Xt, m, h, niter, "natural", V)
  @printf("done\n")

  # ChainQ (second initialization)
  # @printf("Running ChainQ initialization... ")
  # C, B, R, train_error = train_chainq(Xt, m, h, R, B, C, niter, V)
  # @printf("done\n")

  # Actual experiment
  experiment_lsq_cuda(Xt, B, C, R, Xb, Xq, gt, m, h, niter, ilsiter, icmiter, randord, npert, knn, nsplits_train, nsplits_base, V)
end

function experiment_lsq_cuda_query_base(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of C-B update updates
  ilsiter::Integer= 8, # number of ILS iterations
  icmiter::Integer= 4, # number of ICM iterations
  randord::Bool=true, # whether to explore the codes in random order
  npert::Integer= 4, # numebr of codes to perturb in each ILS iterations
  init::String="natural",
  niter_opq::Integer=25,
  noter_chainq::Integer=25,
  knn::Integer=1000,
  nsplits_train::Integer=1, # Number of splits for training data so GPU does not run out of memory
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  # OPQ initialization
  if V; @printf("Running OPQ initialization... "); end
  C, B, R, opq_error = train_opq(Xt, m, h, niter_opq, init, V)
  if V; @printf("done\n"); end

  # ChainQ (second initialization)
  if V; @printf("Running ChainQ initialization... "); end
  C, B, R, chainq_error = train_chainq(Xt, m, h, R, B, C, niter_chainq, V)
  if V; @printf("done\n"); end

  # Actual experiment
  experiment_lsq_cuda_query_base(Xt, B, C, R, Xq, gt, m, h, niter, knn, nsplits_train, V), opq_error
end
