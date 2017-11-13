
export train_lsq, encoding_icm, encode_icm_cuda

"Encodes a database with ILS in cuda"
function encode_icm_cuda(
  RX::Matrix{Float32},
  B::Matrix{Int16},
  C::Vector{Matrix{Float32}},
  cbnorms::Vector{Float32},
  ilsiters::Vector{Int64},
  icmiter::Integer,
  npert::Integer,
  randord::Bool,
  qdbnorms::Bool)

  d, n = size( RX )
  m    = length( C )
  _, h = size( C[1] )

  # Number of results to keep track of
  nr   = length( ilsiters );

  # Make space for the outputs
  Bs   = Vector{Matrix{Int16}}(nr);
  objs = Vector{Float32}(nr);

  # === Compute binary terms (products between all codebook pairs) ===
  binaries, cbi = Rayuela.get_binaries( C );
  _, ncbi       = size( cbi );

  # Create a transposed copy of the binaries for cache-friendliness
  binaries_t = similar( binaries );
  for j = 1:ncbi
    binaries_t[j] = binaries[j]';
  end

  # Create an index from codebook pairs to indices
  cbpair2binaryidx   = zeros(Int32, m, m);
  for j = 1:ncbi
    cbpair2binaryidx[ cbi[1,j], cbi[2,j] ] = j;
  end

  # CUDArt.devices( dev->true ) do devlist
  dev = CuDevice(0);
  ctx = CuContext(dev);

  # Initialize the cuda module, and choose the GPU
  gpuid = 0
  CudaUtilsModule.init( gpuid, cudautilsptx )
  # CUDArt.device( devlist[gpuid] )

  # === Create a state for random number generation ===
  @printf("Creating %d random states... ", n); tic()
  # d_state = CUDArt.malloc( Ptr{Void}, n*64 )
  d_state = CUDAdrv.Mem.alloc( n*64 )
  CudaUtilsModule.setup_kernel( cld(n, 1024), 1024, Cint(n), d_state )

  # CUDArt.device_synchronize()
  CUDAdrv.synchronize(ctx)
  @printf("done in %.2f secnds\n", toq())

  # Copy X and C to the GPU
  d_RX = CuArray( RX );
  d_C  = CuArray( cat(2, C... ))

  # === Get unaries in the gpu ===
  d_prevcost = CuArray{Cfloat}(n)
  d_newcost  = CuArray{Cfloat}(n)

  # CUBLAS.cublasCreate_v2( CUBLAS.cublashandle )
  d_unaries   = Vector{CuArray{Float32}}(m)
  d_codebooks = Vector{CuArray{Float32}}(m)
  for j = 1:m
    d_codebooks[j] = CuArray(C[j])
    # -2 * C' * X
    d_unaries[j] = CUBLAS.gemm('T', 'N', -2.0f0, d_codebooks[j], d_RX)
    # Add self-codebook interactions || C_{i,i} ||^2
    CudaUtilsModule.vec_add( n, (1,h), d_unaries[j], CuArray(diag( C[j]' * C[j] )), Cint(n), Cint(h) )
  end

  # === Get binaries to the GPU ===
  d_binaries  = Vector{CuArray{Float32}}( ncbi );
  d_binariest = Vector{CuArray{Float32}}( ncbi );
  for j = 1:ncbi
    d_binaries[j]  = CuArray( binaries[j] )
    d_binariest[j] = CuArray( binaries_t[j] )
  end

  # Allocate space for temporary results
  bbs = Vector{Matrix{Cfloat}}(m-1);

  # Initialize the previous cost
  prevcost = Rayuela.veccost( RX, B, C );
  IDX = 1:n;

  # For codebook i, we have to condition on these codebooks
  to_look      = 1:m;
  to_condition = zeros(Int32, m-1, m);
  for j = 1:m
    tmp = collect(1:m);
    splice!( tmp, j );
    to_condition[:,j] = tmp;
  end

  # Loop for the number of requested ILS iterations
  for i = 1:maximum( ilsiters )

    @show i, ilsiters
    @time begin

    to_look_r      = to_look;
    to_condition_r = to_condition;

    # Compute the cost of the previous assignments
    # CudaUtilsModule.veccost(n, (1, d), d_RX, d_C, CuArray( convert(Matrix{Cuchar}, (B')-1) ), d_prevcost, Cint(m), Cint(n));
    CudaUtilsModule.veccost2(n, (1, d), d_RX, d_C, CuArray( convert(Matrix{Cuchar}, (B')-1) ), d_prevcost, Cint(d), Cint(m), Cint(n));
    CUDAdrv.synchronize(ctx)
    # prevcost = to_host( d_prevcost )
    prevcost = Array( d_prevcost )

    newB = copy(B)

    # Randomize the visit order in ICM
    if randord
      to_look_r      = randperm( m );
      to_condition_r = to_condition[:, to_look_r];
    end

    d_newB = CuArray( convert(Matrix{Cuchar}, newB-1 ) )

    # Perturn npert entries in each code
    CudaUtilsModule.perturb( n, (1,m), d_state, d_newB, Cint(n), Cint(m), Cint(npert) )

    newB = Array( d_newB )
    Bt   = newB';
    d_Bs = CuArray( Bt );

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
        d_bbs = CuArray( convert(Matrix{Cfloat}, cat(2,bbs...)) );
        # Sum binaries (condition) and minimize
        CudaUtilsModule.condition_icm3(
          n, (1, h),
          d_unaries[k], d_bbs, d_Bs, Cint(k-1), Cint(m), Cint(n) );
        CUDAdrv.synchronize(ctx)

        kidx = kidx + 1;
      end # for k = to_look_r
    end # for j = 1:icmiter

    newB = Array( d_Bs )
    newB = convert(Matrix{Int16}, newB')
    newB .+= 1

    # Keep only the codes that improved
    CudaUtilsModule.veccost2(n, (1, d), d_RX, d_C, d_Bs, d_newcost, Cint(d), Cint(m), Cint(n))
    CUDAdrv.synchronize(ctx)

    newcost = Array( d_newcost )

    areequal = newcost .== prevcost
    println("$(sum(areequal)) new codes are equal")

    arebetter = newcost .< prevcost
    println("$(sum(arebetter)) new codes are better")

    newB[:, .~arebetter] = B[:, .~arebetter]
    B = get_new_B( newB, m, n )
    end

    # Check if this # of iterations was requested
    if i in ilsiters

      ithidx = find( i .== ilsiters )[1]

      # Compute and save the objective
      obj = qerror( RX, B, C )
      @show obj
      objs[ ithidx ] = obj

      if qdbnorms
        dbnormsB = quantize_norms( B, C, cbnorms )
        # Save B
        B_with_norms = vcat( B, reshape(dbnormsB, 1, n))

        @show size( B_with_norms )
        Bs[ ithidx ] = B_with_norms;
      else
        @show size( B )
        Bs[ ithidx ] = B;
      end

    end # end if i in ilsiters
  end # end for i=1:max(ilsiters)

  # CUBLAS.cublasDestroy_v2( CUBLAS.cublashandle )
  CudaUtilsModule.finit()

  destroy!(ctx)
  # end # do devlist

  return Bs, objs

end

"Randomly perturbs the codes"
function perturb_codes!(
  B::Union{Matrix{Int16},SharedMatrix{Int16}}, # in/out. Codes to perturb
  npert::Integer,         # in. Number of entries to perturb in each code
  h::Integer,             # in. The number of codewords in each codebook
  IDX::UnitRange{Int64})  # in. Subset of codes in B to perturb

  m, _ = size(B)
  n    = length(IDX)

  # Sample random perturbation indices (places to perturb) in B
  pertidx  = Matrix{Integer}(npert, n)
  for i = 1:n
    # Sample npert unique values out of m
    sampleidx = Distributions.sample(1:m, npert, replace=false, ordered=true)

    # Save them in pert_idx
    for j = 1:npert
      pertidx[j, i] = sampleidx[j]
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


# Encode using iterated conditional modes
function encode_icm_fully!{T <: AbstractFloat}(
  B::Union{Matrix{Int16},SharedMatrix{Int16}},  # in/out. Initialization, and the place where the results are saved.
  X::Matrix{T},                 # in. d-by-n data to encode
  C::Vector{Matrix{T}},         # in. m-long vector with d-by-h codebooks
  binaries::Vector{Matrix{T}},  # in. The binary terms
  cbi::Matrix{Int32},           # in. 2-by-ncbi. indices for inter-codebook interactions
  niter::Integer,               # in. number of iterations for block-icm
  randord::Bool,                # in. whether to randomize search order
  npert::Integer,               # in. number of codes to perturb per iteration
  IDX::UnitRange{Int64},        # in. Index to save the result
  V::Bool)                      # in. whether to print progress

  # Compute unaries
  unaries = get_unaries( X, C, V )

  h, n = size( unaries[1] )
  m, _ = size( B )

  ncbi = length( binaries )

  # Create a transposed copy of the binaries
  binaries_t = similar( binaries )
  for i = 1:ncbi
    binaries_t[i] = binaries[i]'
  end

  # Create an index from codebook pairs to indices
  cbpair2binaryidx   = zeros(Int32, m, m)
  for i = 1:ncbi
    cbpair2binaryidx[ cbi[1,i], cbi[2,i] ] = i
  end

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
    to_look      = randperm( m )
    to_condition = to_condition[:, to_look]
  end

  # Preallocate some space
  bb = Matrix{T}( h, h )
  ub = Matrix{T}( h, n )

  # Perturb the codes
  B = perturb_codes!(B, npert, h, IDX)

  #Profile.@profile begin
  @inbounds for i=1:niter # Do the number of passed iterations

    # This is the codebook that we are updating (i.e., the node)
    jidx = 1;
    for j = to_look
      # Get the unaries that we will work on
      copy!(ub, unaries[j])

      # These are all the nodes that we are conditioning on (i.e., the edges to 'absorb')
      for k = to_condition[:, jidx]

        # Determine the pairwise interactions that we'll use (for cache-friendliness)
        if j < k
          binariidx = cbpair2binaryidx[ j, k ]
          bb = binaries[ binariidx ]
        else
          binariidx = cbpair2binaryidx[ k, j ]
          bb = binaries_t[ binariidx ]
        end

        # Traverse the unaries, absorbing the appropriate binaries
        for l=1:n
          codek = B[k, IDX[l]]
          #@simd for ll = 1:h
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
  end # for i=1:niter
  #end # profile

end


# Encode a full dataset
function encoding_icm{T <: AbstractFloat}(
  X::Matrix{T},         # d-by-n matrix. Data to encode
  oldB::Matrix{Int16},  # m-by-n matrix. Previous encoding
  C::Vector{Matrix{T}}, # m-long vector with d-by-h codebooks
  niter::Integer,       # number of ICM iterations
  randord::Bool,        # whether to use random order
  npert::Integer,       # the number of codes to perturb
  V::Bool=false)        # whether to print progress

  d, n =    size( X )
  m    =  length( C )
  _, h = size( C[1] )

  # Compute binaries between all codebook pairs
  binaries, cbi = get_binaries( C )
  _, ncbi       = size( cbi )

  # Compute the cost of the previous assignments
  prevcost = veccost( X, oldB, C )

  if nworkers() == 1
    B = zeros(Int16, m, n)
  else
    B = SharedArray{Int16}(m, n)
  end

  @inbounds @simd for i = 1:m*n
    B[i] = oldB[i]
  end

  if nworkers() == 1
    encode_icm_fully!( B, X, C, binaries, cbi, niter, randord, npert, 1:n, V )
    #encode_icm_cpp!( B, X, C, binaries, cbi, niter, randord, npert, 1:n, V );
  else
    paridx = splitarray( 1:n, nworkers() )
    @sync begin
      for (i,wpid) in enumerate(workers())
        @async begin
          Xw = X[:,paridx[i]]
          remotecall_wait(encode_icm_fully!, wpid, B, Xw, C, binaries, cbi, niter, randord, npert, paridx[i], V )
        end
      end
    end
  end
  B = sdata(B)

  # Keep only the codes that improved
  newcost = veccost( X, B, C )

  areequal = newcost .== prevcost
  if V println("$(sum(areequal)) new codes are equal"); end

  arebetter = newcost .< prevcost
  if V println("$(sum(arebetter)) new codes are better"); end

  B[:, .~arebetter] = oldB[:, .~arebetter]

  return B
end


function train_lsq{T <: AbstractFloat}(
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
  V::Bool=false)        # whether to print progress

  # if V
  println("**********************************************************************************************");
  println("Doing local search with $m codebooks, $npert perturbations, $icmiter icm iterations and random order = $randord");
  println("**********************************************************************************************");
  # end

  d, n = size( X )

  # Update RX
  RX = R' * X

  # Initialize C
  C = update_codebooks( RX, B, h, V, "lsqr" )

  # Apply the rotation to the codebooks
  for i = 1:m
    C[i] = R * C[i]
  end
  @printf("%3d %e \n", -2, qerror( X, B, C ))

  # Initialize B
  for i = 1:ilsiter
    B = encoding_icm( X, B, C, icmiter, randord, npert, V )
    @everywhere gc()
  end
  @printf("%3d %e \n", -1, qerror( X, B, C ))

  obj = zeros( Float32, niter )

  for iter = 1:niter

    obj[iter] = qerror( X, B, C )
    @printf("%3d %e \n", iter, obj[iter])

    # Update the codebooks
    C = update_codebooks( X, B, h, V, "lsqr" )

    # Update the codes with local search
    for i = 1:ilsiter
      B = encoding_icm( X, B, C, icmiter, randord, npert, V )
      @everywhere gc()
    end

  end

  # Get the codebook for norms
  CB = reconstruct(B, C)

  dbnorms = zeros(Float32, 1, n)
  for i = 1:n
     for j = 1:d
        dbnorms[i] += CB[j,i].^2
     end
  end

  # Quantize the norms with plain-old k-means
  dbnormsq = kmeans(dbnorms, h)
  cbnorms  = dbnormsq.centers

  # Add the dbnorms to the codes
  B_norms  = reshape( dbnormsq.assignments, 1, n )

  return C, B, cbnorms, B_norms, obj

end
