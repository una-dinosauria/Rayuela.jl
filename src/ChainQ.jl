
# function quantize_chainq
export train_chainq, quantize_chainq

function quantize_chainq!{T <: AbstractFloat}(
  CODES::SharedMatrix{Int16},  # out. Where to save the result
  X::Matrix{T},                # in. d-by-n matrix to encode
  C::Vector{Matrix{T}},        # in. m-long vector with d-by-h codebooks
  binaries::Vector{Matrix{T}}, # in. Binary terms
  IDX::UnitRange{Int64})       # in. Index to save the result

  # Get unaries
  unaries = get_unaries( X, C )

  h, n = size( unaries[1] )
  m    = length( binaries ) + 1

  # We need a matrix to keep track of the min and argmin
  mincost = zeros(T, h, m )
  minidx  = zeros(Integer, h, m )

  # Allocate memory for brute-forcing each pair
  cost = zeros( T, h )

  U = zeros(T, h, m)

  minv = typemax(T)
  mini = 1

  uidx = 1
  @inbounds for idx = IDX # Loop over the datapoints

    # Put all the unaries of this item together
    for i = 1:m
      ui = unaries[i]
      @simd for j = 1:h
        U[j,i] = ui[j,uidx]
      end
    end

    # Forward pass
    for i = 1:(m-1) # Loop over states

      # If this is not the first iteration, add the precomputed costs
      if i > 1
        @simd for j = 1:h
          U[j,i] += mincost[j,i-1]
        end
      end

      bb = binaries[i]
      for j = 1:h # Loop over the cost of going to j
        @simd for k = 1:h # Loop over the cost of coming from k
          ucost   =  U[k, i] # Pay the unary of coming from k
          bcost   = bb[k, j] # Pay the binary of going from j-k
          cost[k] = ucost + bcost
        end

        # Writing my own findmin
        minv = cost[1]
        mini = 1
        for k = 2:h
          costi = cost[k]
          if costi < minv #|| minv!=minv
            minv = costi
            mini = k
          end
        end

        mincost[j, i] = minv
         minidx[j, i] = mini
      end
    end

    @simd for j = 1:h
      U[j,m] += mincost[j,m-1]
    end

    _, mini = findmin( U[:,m] )

    # Backward trace
    backpath = [ mini ]
    for i = (m-1):-1:1
      push!( backpath, minidx[ backpath[end], i ] )
    end

    # Save the inferred code
    CODES[:, idx] = reverse( backpath )

    uidx = uidx + 1;
  end # for idx = IDX
end


"Function to call that encodes a dataset using dynamic programming"
function quantize_chainq(
  X::Matrix{Float32},         # d-by-n matrix. Data to encode
  C::Vector{Matrix{Float32}}) # m-long vector with d-by-h codebooks

  d, n = size( X );
  m    = length( C );

  # Compute binary tables
  binaries = Vector{Matrix{Float32}}(m-1)
  for i = 1:(m-1)
    binaries[i] = 2 * C[i]' * C[i+1]
  end
  CODES = SharedArray{Int16,2}((m, n))

  if nworkers() == 1
    quantize_chainq!( CODES, X, C, binaries, 1:n )
  else
    paridx = splitarray( 1:n, nworkers() )
    @sync begin
      for (i,wpid) in enumerate(workers())
        @async begin
          Xw = X[:,paridx[i]]
          remotecall_wait(quantize_chainq!, wpid, CODES, Xw, C, binaries, paridx[i])
        end
      end
    end
  end

  return sdata(CODES)
end

# Train a chain quantizer with viterbi decoding
function train_chainq{T <: AbstractFloat}(
  X::Matrix{T},             # d-by-n matrix of data points to train on.
  m::Integer,               # number of codebooks
  h::Integer,               # number of entries per codebook
  R::Matrix{T},             # Init rotation matrix
  B::Matrix{Int16},         # Init codes
  C::Vector{Matrix{T}},     # Init codebooks
  niter::Integer,           # number of optimization iterations
  V::Bool=false)            # whether to print progress

  if V; @printf("Training a chain quantizer\n"); end

  d, n = size( X )
  obj  = zeros(Float32, niter+1)

  CB = zeros( Float32, size(X) )
  RX = R' * X

  # Initialize C
  C = update_codebooks_chain( RX, B, h, V )
  if V; @printf("%3d %e\n", -2, qerror( RX, B, C )); end

  # Initialize B
  B   = quantize_chainq( RX, C )
  if V; @printf("%3d %e\n", -1, qerror( RX, B, C )); end

  for iter = 0:niter
    if V; tic(); end # Take time if asked to

    obj[iter+1] = qerror( RX, B, C  )
    if V; @printf("%3d %e... ", iter, obj[iter+1]); end

    # update CB
    CB[:] = 0
    for i = 1:m; CB += C[i][:, vec(B[i,:]) ]; end

    # update R
    U, S, VV = svd(X * CB', thin=true)
    R = U * VV'

    # update R*X
    RX = R' * X

    # Update the codebooks #
    C = update_codebooks_chain( RX, B, h, V )

    # Update the codes with lattice search
    B = quantize_chainq( RX, C )
      if V; @printf("done in %.2f secs\n", toq()); end
  end

  return C, B, R, obj
end
