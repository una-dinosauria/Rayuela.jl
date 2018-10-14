
export train_lsq, encoding_icm

"Randomly perturbs codes"
function perturb_codes!(
  B::Union{Matrix{Int16},SharedMatrix{Int16}}, # in/out. Codes to perturb
  npert::Integer,         # in. Number of entries to perturb in each code
  h::Integer,             # in. The number of codewords in each codebook
  IDX::UnitRange{Int64},  # in. Subset of codes in B to perturb
  replace::Bool=true) # in. Whether to sample with replacement

  m, _ = size(B)
  n    = length(IDX)

  # Sample random perturbation indices (places to perturb) in B
  if replace
    # With replacements this is easy
    pertidx  = rand(1:m, npert, n)
  else
    # Each call to sample has about 8 allocations, so this can results in
    # several Million alloc calls too.
    pertidx  = Matrix{Integer}(npert, n)
    for i = 1:n
      # Sample npert unique values out of m
      Distributions.sample!(1:m, view(pertidx,:,i), replace=false, ordered=true)
    end
  end

  # Sample the values that will replace the new ones
  pertvals = rand(1:h, npert, n)
  # Perturb the solutions
  for i = 1:n
    for j = 1:npert
      B[ pertidx[j,i], IDX[i] ] = pertvals[j,i]
    end
  end

  return B
end

"Run iterated conditional modes on N problems"
function iterated_conditional_modes_cpp!(
  B::Matrix{UInt8},  # in/out. Initialization, and the place where the results are saved.
  unaries::Vector{Matrix{T}},     # in. Unary terms
  binaries::Vector{Matrix{T}},    # in. The binary terms
  binaries_t::Vector{Matrix{T}},  # in. Transposed version of the above
  cbpair2binaryidx::Matrix{Int32},
  cbi::Matrix{Int32},           # in. 2-by-ncbi. indices for inter-codebook interactions
  to_look::Union{UnitRange{Int64},Vector{Int64}},
  to_condition::Matrix{Int32},
  icmiter::Integer,             # in. number of iterations for block-icm
  npert::Integer,               # in. number of codes to perturb per iteration
  IDX::UnitRange{Int64},        # in. Index to save the result
  ub::Matrix{T}, bb::Matrix{T}, # in. Preallocated memory
  h::Integer, n::Integer, m::Integer,
  V::Bool) where T <: AbstractFloat

  cbpair2binaryidx = convert(Matrix{Int32}, cbpair2binaryidx.-1)
  to_condition     = convert(Matrix{Int32}, to_condition.-1)

  binaries   = hcat(binaries...)
  binaries_t = hcat(binaries_t...)

  @inbounds for i=1:icmiter # Do the number of passed iterations

    # for j = to_look
    for j = 1:m
      # Get the unaries that we will work on
      copyto!(ub, unaries[to_look[j]])

      ccall(("condition", encode_icm_so), Nothing,
        (Ptr{Cuchar}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},
        Ptr{Cint}, Ptr{Cint}, Cint, Cint, Cint),
        B, ub, binaries, binaries_t,
        cbpair2binaryidx, to_condition[:,j], to_look[j]-1, n, m)

    end # for j = 1:m
  end # for i=1:icmiter

end

"Run iterated conditional modes on N problems"
function iterated_conditional_modes!(
  B::Union{Matrix{Int16},SharedMatrix{Int16}},  # in/out. Initialization, and the place where the results are saved.
  unaries::Vector{Matrix{T}},     # in. Unary terms
  binaries::Vector{Matrix{T}},    # in. The binary terms
  binaries_t::Vector{Matrix{T}},  # in. Transposed version of the above
  cbpair2binaryidx::Matrix{Int32},
  cbi::Matrix{Int32},           # in. 2-by-ncbi. indices for inter-codebook interactions
  to_look::Union{UnitRange{Int64},Vector{Int64}},
  to_condition::Matrix{Int32},
  icmiter::Integer,             # in. number of iterations for block-icm
  npert::Integer,               # in. number of codes to perturb per iteration
  IDX::UnitRange{Int64},        # in. Index to save the result
  ub::Matrix{T}, bb::Matrix{T}, # in. Preallocated memory
  h::Integer, n::Integer, m::Integer,
  V::Bool) where T <: AbstractFloat

  @inbounds for i=1:icmiter # Do the number of passed iterations

    # This is the codebook that we are updating (i.e., the node)
    jidx = 1;
    for j = to_look
      # Get the unaries that we will work on
      copyto!(ub, unaries[j])

      # These are all the nodes that we are conditioning on (i.e., the edges to 'absorb')
      for k = to_condition[:, jidx]

        # Determine the pairwise interactions that we'll use (for cache-friendliness)
        binariidx = cbpair2binaryidx[ j, k ]
        if j < k
          bb = binaries[ binariidx ]
        else
          bb = binaries_t[ binariidx ]
        end

        # Traverse the unaries, absorbing the appropriate binaries
        for l=1:n
          codek = B[k, IDX[l]]
          for ll = 1:h
            ub[ll, l] += bb[ ll, codek ]
          end
        end
      end #for k=to_condition

      # Once we are done conditioning, traverse the absorbed unaries to find mins
      inidx = 1
      for idx=IDX
        minv = ub[1, inidx]
        mini = 1
        for k = 2:h
          ubi = ub[k, inidx]
          if ubi < minv
            minv = ubi
            mini = k
          end
        end

        B[j, idx] = mini
        inidx = inidx .+ 1
      end

      jidx = jidx .+ 1

    end # for j=to_look
  end # for i=1:icmiter

end

# Encode using iterated conditional modes
function encode_icm_fully!(
  oldB::Matrix{Int16},  # in/out. Initialization, and the place where the results are saved.
  X::Matrix{T},                 # in. d-by-n data to encode
  C::Vector{Matrix{T}},         # in. m-long vector with d-by-h codebooks
  binaries::Vector{Matrix{T}},  # in. The binary terms
  cbi::Matrix{Int32},           # in. 2-by-ncbi. indices for inter-codebook interactions
  ilsiter::Integer,             # in. number of ILS iterations
  icmiter::Integer,             # in. number of ICM iterations
  randord::Bool,                # in. whether to randomize search order
  npert::Integer,               # in. number of codes to perturb per iteration
  IDX::UnitRange{Int64},        # in. Index to save the result
  cpp::Bool=true,               # in. Whether to use the cpp implementation
  V::Bool=true) where T <: AbstractFloat

  # Compute unaries
  @time begin
  unaries = get_unaries(X, C, V)

  h, n = size(unaries[1])
  m, _ = size(oldB)

  if cpp && h != 256
    error("The C++ implementation of ICM encoding only supports codebooks with 256 entries")
  end

  ncbi = length(binaries)

  # Create a transposed copy of the binaries
  binaries_t = similar(binaries)
  for i = 1:ncbi
    binaries_t[i] = collect(binaries[i]')
  end

  # Create an index from codebook pairs to indices
  cbpair2binaryidx = zeros(Int32, m, m)
  for i = 1:ncbi
    cbpair2binaryidx[ cbi[1,i], cbi[2,i] ] = i
    cbpair2binaryidx[ cbi[2,i], cbi[1,i] ] = i
  end

  # Preallocate some space
  bb = zeros(T, h, h)
  ub = zeros(T, h, n)
  end

  B = zeros(Int16, m, n)

  for ilstrack = 1:ilsiter

    prevcost = veccost( X, oldB, C )
    if nworkers() == 1
      B = zeros(Int16, m, n)
    else
      B = SharedArray{Int16}(m, n)
    end
    copyto!(B, oldB)

    # For codebook i, we have to condition on these codebooks
    to_look      = 1:m
    to_condition = zeros(Int32, m-1, m)
    for i = 1:m
      tmp = collect(1:m)
      splice!( tmp, i )
      to_condition[:,i] = tmp
    end
    # Make the order random
    if randord
      to_look      = randperm(m)
      to_condition = to_condition[:, to_look]
    end

    # Perturb the codes
    B = perturb_codes!(B, npert, h, IDX)

    # Run iterated conditional modes
    if cpp
      B = convert(Matrix{UInt8}, B .- one(Int16))
      iterated_conditional_modes_cpp!(B, unaries,
        binaries, binaries_t, cbpair2binaryidx, cbi,
        to_look, to_condition, icmiter, npert, IDX, ub, bb, h, n, m, V)
      B = convert(Matrix{Int16}, B) .+ one(Int16)
    else
      iterated_conditional_modes!(B, unaries,
        binaries, binaries_t, cbpair2binaryidx, cbi,
        to_look, to_condition, icmiter, npert, IDX, ub, bb, h, n, m, V)
    end

    # Keep only the codes that improved
    newcost = veccost( X, B, C )
    areequal = newcost .== prevcost
    arebetter = newcost .< prevcost
    if V; @printf(" ILS iteration %d/%d done. ", ilstrack, ilsiter); end
    if V; @printf("%5.2f%% new codes are equal. ", 100*sum(areequal)/n ); end
    if V; @printf("%5.2f%% new codes are better.\n", 100*sum(arebetter)/n ); end

    B[:, .~arebetter] = oldB[:, .~arebetter]
    copyto!(oldB, B)
  end

  return B
end

"""
    encoding_icm(X, oldB, C, ilsiter, icmiter, randord, npert, cpp=true, V=false) -> B

Given data and chain codebooks, find codes using iterated local search with ICM.

# Arguments
- `X::Matrix{T}`: `d`-by-`n` data to quantize
- `OldB::Matrix{Int16}`: `m`-by-`n` initial set of codes
- `ilsiter::Integer`: Number of iterated local search (ILS) iterations
- `icmiter::Integer`: Number of iterated conditional modes (ICM) iterations
- `randord::Bool`: Whether to use random order
- `npert::Integer`: Number of codes to perturb
- `cpp::Bool=true`: Whether to use the c++ implementation
- `V::Bool=false`: Whehter to print progress

# Returns
- `B::Matrix{Int16}`: `m`-by-`n` matrix with the new codes
"""
function encoding_icm(
  X::Matrix{T},         # d-by-n matrix. Data to encode
  oldB::Matrix{Int16},  # m-by-n matrix. Previous encoding
  C::Vector{Matrix{T}}, # m-long vector with d-by-h codebooks
  ilsiter::Integer,     # in. number of ILS iterations
  icmiter::Integer,     # in. number of ICM iterations
  randord::Bool,        # whether to use random order
  npert::Integer,       # the number of codes to perturb
  cpp::Bool=true,        # whether to use the cpp implementation
  V::Bool=true) where T <: AbstractFloat # whether to print progress

  d, n =    size( X )
  m    =  length( C )
  _, h = size( C[1] )

  # Compute binaries between all codebook pairs
  binaries, cbi = get_binaries( C )
  _, ncbi       = size( cbi )

  B = encode_icm_fully!(oldB, X, C, binaries, cbi, ilsiter, icmiter, randord, npert, 1:n, cpp, V)

  return B
end


"""
    train_lsq(X, m, h, R, B, C, niter, ilsiter, icmiter, randord, npert, cpp=true, V=false) -> C, B, obj

Train a local-search quantizer.
This method is typically initialized by [Chain quantization (ChainQ)](@ref)

# Arguments
- `X::Matrix{T}`: `d`-by-`n` data to quantize
- `m::Integer`: Number of codebooks
- `h::Integer`: Number of entries in each codebook (typically 256)
- `R::Matrix{T}`: `d`-by-`d` rotation matrix for initialization
- `B::Matrix{Int16}`: `m`-by-`n` matrix with pre-trained codes for initialization
- `C::Vector{Matrix{T}}`: `m`-long vector with `d`-by-`h` matrices. Each matrix is a pretrained codebook of size approximately `d`-by-`h`
- `niter::Integer`: Number of iterations to use
- `ilster::Integer`: Number of iterated local search (ILS) iterations
- `icmiter::Integer`: Number of iterated conditional modes (ICM) iterations
- `randord::Bool`: Whether to visit the nodes in a random order in ICM
- `npert::Integer`: Number of codes to perturb
- `cpp::Bool`: Whether to use a c++ implementation for encoding
- `V::Bool`: Whether to print progress

# Returns
- `C::Vector{Matrix{T}}`: `m`-long vector with `d`-by-`h` matrix entries. Each matrix is a codebook of size approximately `d`-by-`h`
- `B::Matrix{Int16}`: `m`-by-`n` matrix with the codes
- `obj::Vector{T}`: `niter`-long vector with the quantization error after each iteration
"""
function train_lsq(
  X::Matrix{T},         # d-by-n matrix of data points to train on.
  m::Integer,           # number of codebooks
  h::Integer,           # number of entries per codebook
  R::Matrix{T},         # init rotation
  B::Matrix{Int16},     # init codes
  C::Vector{Matrix{T}}, # init codebooks
  niter::Integer,       # number of optimization iterations
  ilsiter::Integer,     # number of ILS iterations to use during encoding
  icmiter::Integer,     # number of iterations in local search
  randord::Bool,        # whether to use random order
  npert::Integer,       # The number of codes to perturb
  cpp::Bool=true,       # whether to use ICM's cpp implementation
  V::Bool=true) where T <: AbstractFloat

  # if V
  println("**********************************************************************************************");
  println("Training LSQ with $m codebooks, $npert perturbations, $icmiter icm iterations and random order = $randord");
  println("**********************************************************************************************");
  # end

  d, n = size( X )

  # Initialize C
  RX = R' * X
  C = update_codebooks(RX, B, h, V, "fastbin")

  # Apply the rotation to the codebooks
  for i = 1:m; C[i] = R * C[i]; end
  @printf("%3d %e \n", -2, qerror( X, B, C ))

  # Initialize B
  @time B = encoding_icm(X, B, C, ilsiter, icmiter, randord, npert, cpp, V)
  @printf("%3d %e \n", -1, qerror( X, B, C ))

  obj = zeros(T, niter)

  for iter = 1:niter
    obj[iter] = qerror(X, B, C)
    @printf("%3d %e \n", iter, obj[iter])

    # Update the codebooks C
    C = update_codebooks(X, B, h, V, "fastbin")

    # Update the codes B
    @time B = encoding_icm(X, B, C, ilsiter, icmiter, randord, npert, cpp, V)
  end

  return C, B, obj
end

function experiment_lsq_query_base(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  B::Matrix{T2}, # codes
  C::Vector{Matrix{T}}, # codebooks
  R::Matrix{T}, # rotation
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of k-means iterations for training
  knn::Integer=1000, # Compute recall @N for this value of N
  V::Bool=false) where {T <: AbstractFloat, T2 <: Integer} # whether to print progress

  # TODO expose these parameters
  ilsiter = 8
  icmiter = 4
  randord = true
  npert   = 4
  cpp     = true

  # Train LSQ
  d, _ = size(Xt)
  if V; @printf("Running CUDA LSQ training... "); end
  C, B, train_error = Rayuela.train_lsq(Xt, m, h, R, B, C, niter, ilsiter, icmiter, randord, npert, cpp, V)
  if V; @printf("done\n"); end

  norms_B, norms_C = get_norms_codebook(B, C)
  db_norms         = vec( norms_C[ norms_B ] )

  if V; print("Querying m=$m ... "); end
  @time dists, idx = linscan_lsq(B, Xq, C, db_norms, eye(Float32, d), knn)
  if V; println("done"); end

  recall = eval_recall(gt, idx, knn)
  return C, B, R, train_error, recall
end

function experiment_lsq(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  B::Matrix{T2}, # codes
  C::Vector{Matrix{T}}, # codebooks
  R::Matrix{T}, # rotation
  Xb::Matrix{T}, # d-by-n. Base set
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of k-means iterations for training
  knn::Integer=1000,
  V::Bool=false) where {T <: AbstractFloat, T2 <: Integer} # whether to print progress

  # TODO expose these parameters
  ilsiter_train = 8
  ilsiter_base = 32
  icmiter = 4
  randord = true
  npert   = 4
  cpp     = false

  # Train LSQ
  d, _ = size(Xt)
  C, B, train_error = Rayuela.train_lsq(Xt, m, h, R, B, C, niter, ilsiter_train, icmiter, randord, npert, cpp, V)
  norms_B, norms_C = get_norms_codebook(B, C)

  # === Encode the base set ===
  B_base = convert(Matrix{Int16}, rand(1:h, m, size(Xb,2)))
  @time B_base = Rayuela.encoding_icm(Xb, B_base, C, ilsiter_base, icmiter, randord, npert, cpp, V)
  base_error = qerror(Xb, B_base, C)
  if V; @printf("Error in base is %e\n", base_error); end

  # Compute and quantize the database norms
  B_base_norms, db_norms_X = quantize_norms( B_base, C, norms_C )
  db_norms     = vec( norms_C[ B_base_norms ] )

  if V; print("Querying m=$m ... "); end
  @time dists, idx = linscan_lsq(B_base, Xq, C, db_norms, eye(Float32, d), knn)
  if V; println("done"); end

  recall = eval_recall(gt, idx, knn)
  return C, B, R, train_error, B_base, recall
end

"Runs an lsq experiment/demo"
function experiment_lsq(
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
  experiment_lsq(Xt, B, C, R, Xb, Xq, gt, m, h, niter, knn, V)
end
