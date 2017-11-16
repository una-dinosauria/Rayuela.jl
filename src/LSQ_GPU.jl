
export encode_icm_cuda, train_lsq_cuda

"Encodes a database with ILS in cuda"
function encode_icm_cuda(
  RX::Matrix{Float32},         # in. The data to encode
  B::Matrix{Int16},            # in. Initial list of codes
  C::Vector{Matrix{Float32}},  # in. Codebooks
  ilsiters::Vector{Int64},     # in. ILS iterations to record Bs and obj function. Its max is the total number of iteration we will run.
  icmiter::Integer,            # in. Number of ICM iterations
  npert::Integer,              # in. Number of entries to perturb
  randord::Bool)               # in. Whether to randomize the order in which nodes are visited in ILS

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
    B = copy( newB )
    end

    # Check if this # of iterations was requested
    if i in ilsiters

      ithidx = find( i .== ilsiters )[1]

      # Compute and save the objective
      obj = qerror( RX, B, C )
      @show obj
      objs[ ithidx ] = obj
      @show size(B)
      Bs[ ithidx ] = B

    end # end if i in ilsiters
  end # end for i=1:max(ilsiters)

  # CUBLAS.cublasDestroy_v2( CUBLAS.cublashandle )
  CudaUtilsModule.finit()

  destroy!(ctx)
  # end # do devlist

  return Bs, objs

end

"LSQ training but some things happen in the GPU"
function train_lsq_cuda{T <: AbstractFloat}(
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
  B, _ = encode_icm_cuda(X, B, C, [ilsiter], icmiter, npert, randord)
  B    = B[end]
  @printf("%3d %e \n", -1, qerror( X, B, C ))

  obj = zeros( Float32, niter )

  for iter = 1:niter

    obj[iter] = qerror( X, B, C )
    @printf("%3d %e \n", iter, obj[iter])

    # Update the codebooks
    C = update_codebooks( X, B, h, V, "lsqr" )

    # Update the codes with local search
    B, _ = encode_icm_cuda(X, B, C, [ilsiter], icmiter, npert, randord)
    B    = B[end]
    @printf("%3d %e \n", -1, qerror( X, B, C ))

  end

  # Get the codebook for norms
  norms_codes, norms_codebook = get_norms_codebook(B, C)

  return C, B, norms_codebook, norms_codes, obj
end
