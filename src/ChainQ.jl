
# function quantize_chainq
export train_chainq, quantize_chainq

function quantize_chainq_cpp!{T <: AbstractFloat}(
  CODES::Matrix{Int16},  # out. Where to save the result
  X::Matrix{T},                # in. d-by-n matrix to encode
  C::Vector{Matrix{T}},        # in. m-long vector with d-by-h codebooks
  binaries::Vector{Matrix{T}}, # in. Binary terms
  IDX::UnitRange{Int64})       # in. Index to save the result

  # Get unaries
  unaries = get_unaries( X, C )

  h, n = size( unaries[1] )
  m    = length( binaries ) + 1

  # We need a matrix to keep track of the min and argmin
  mincost = zeros(T, h)
  minidx  = zeros(Int32, h, m )

  # Allocate memory for brute-forcing each pair
  cost = zeros( T, h )
  U = zeros(T, h, m)

  CODES2 = zeros(Cuchar,m,n)
  unaries2, binaries2 = vcat(unaries...), hcat(binaries...)
  U2 = similar(U)
  backpath2 = zeros(Int32, m)

  ccall(("viterbi_encoding", encode_icm_so), Void,
    (Ptr{Int16}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},
    Ptr{Cint}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint),
    CODES2, unaries2, binaries2, mincost, U, minidx, cost, backpath2, n, m)

  CODES2 = convert(Matrix{Int16}, CODES2+1)

  CODES[:] = CODES2[:]
end

function my_findmin(cost::Vector{T}, h) where T <: Real
  minv = cost[1]
  mini = 1
  for k = 2:h
    costi = cost[k]
    if costi < minv
      minv = costi
      mini = k
    end
  end
  return minv, mini
end

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
  mincost = zeros(T, h)
  minidx  = zeros(Int32, h, m )

  # Allocate memory for brute-forcing each pair
  cost = zeros( T, h )

  U = zeros(T, h, m)

  minv = typemax(T)
  mini = 1

  # unaries = vcat(unaries...)
  # @show typeof(unaries)

  uidx = 1
  @inbounds for idx = IDX # Loop over the datapoints

    # Put all the unaries of this item together
    # U[:] = unaries[:,idx]
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
          U[j,i] += mincost[j]
        end
      end

      bb = binaries[i]
      for j = 1:h # Loop over the cost of going to j
        @simd for k = 1:h # Loop over the cost of coming from k
          ucost   =  U[k, i] # Pay the unary of coming from k
          bcost   = bb[k, j] # Pay the binary of going from j-k
          cost[k] = ucost + bcost
        end

        # findmin -- julia's is too slow
        minv = cost[1]
        mini = 1
        for k = 2:h
          costi = cost[k]
          if costi < minv
            minv = costi
            mini = k
          end
        end

        mincost[j] = minv
         minidx[j, i] = mini
      end
    end

    # @show mincost, minidx

    @simd for j = 1:h
      U[j,m] += mincost[j]
    end

    _, mini = findmin( U[:,m] )

    # Backward trace
    backpath = [ mini ]
    for i = (m-1):-1:1
      push!(backpath, minidx[ backpath[end], i ])
    end

    # Save the inferred code
    CODES[:, idx] = reverse( backpath )
    uidx = uidx + 1;
  end # for idx = IDX
end

function quantize_chainq_batched!{T <: AbstractFloat}(
  CODES::SharedMatrix{Int16},  # out. Where to save the result
  X::Matrix{T},                # in. d-by-n matrix to encode
  C::Vector{Matrix{T}},        # in. m-long vector with d-by-h codebooks
  binaries::Vector{Matrix{T}}, # in. Binary terms
  IDX::UnitRange{Int64})       # in. Index to save the result

  # Get unaries
  # @Profile.profile begin
  unaries = get_unaries( X, C )

  h, n = size( unaries[1] )
  m    = length( binaries ) + 1

  # We need a matrix to keep track of the min and argmin
  mincost = zeros(T, h, n)
  minidx  = zeros(Int32, h, m, n)

  # Allocate memory for brute-forcing each pair
  cost = zeros(T, h, n)

  minv = typemax(T)
  mini = 1

  # Put all the unaries of this item together
  # unaries = vcat(unaries...)

  # Forward pass
  @inbounds for i = 1:(m-1) # Loop over states

    if i > 1; unaries[i] .+= mincost; end

    bb = binaries[i]
    for j = 1:h # Loop over the cost of going to j
      ucost = unaries[i]
      bcost = bb[:,j]
      # cost  = ucost .+ bcost

      for kk = IDX
        @simd for k=1:h
          cost[k,kk] = ucost[k,kk] + bcost[k]
          # ui[k,kk] += bcost[k]
        end
      end

      # Findmin
      minv, mini = findmin(cost,1)
      mincost[j,:]  = minv
      minidx[j,i,:] = rem.(mini-1,h) + 1
    end
  end

  unaries[m] .+= mincost
  _, mini = findmin(unaries[m],1)
  mini = rem.(mini-1,h) + 1

  @inbounds for idx = IDX # Loop over the datapoints
    # Backward trace
    backpath = [ mini[idx] ]
    for i = (m-1):-1:1
      push!(backpath, minidx[ backpath[end], i, idx ])
    end

    # Save the inferred code
    CODES[:, idx] = reverse( backpath )
  end # for idx = IDX
  # end
end

"Function to call that encodes a dataset using dynamic programming"
function quantize_chainq(
  X::Matrix{Float32},         # d-by-n matrix. Data to encode
  C::Vector{Matrix{Float32}}, # m-long vector with d-by-h codebooks
  use_cpp::Bool=false)

  tic()
  d, n = size( X )
  m    = length( C )

  # Compute binary tables
  binaries = Vector{Matrix{Float32}}(m-1)
  for i = 1:(m-1)
    binaries[i] = 2 * C[i]' * C[i+1]
  end
  CODES = SharedArray{Int16,2}(m, n)
  CODES[:] = 0

  if nworkers() == 1
    if use_cpp
      quantize_chainq_cpp!( sdata(CODES), X, C, binaries, 1:n )
    else
      quantize_chainq!( CODES, X, C, binaries, 1:n )
      # quantize_chainq_batched!( CODES, X, C, binaries, 1:n )
    end
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

  return sdata(CODES), toq()
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
  C, Ctime = update_codebooks_chain( RX, B, h )
  if V; @printf("%3d %e... %.2f secs updating C\n", -2, qerror( RX, B, C ), Ctime); end

  # Initialize B
  B, Btime = quantize_chainq( RX, C )
  if V; @printf("%3d %e... %.2f secs updating B\n", -1, qerror( RX, B, C ), Btime); end

  for iter = 0:niter
    if V; tic(); end # Take time if asked to

    obj[iter+1] = qerror( RX, B, C )
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
    C, Ctime = update_codebooks_chain( RX, B, h )

    # Update the codes with lattice search
    B, Btime = quantize_chainq( RX, C )
    if V; @printf("done in %.2f secs. %.2f secs updating B. %.2f secs updating C\n", toq(), Btime, Ctime); end

  end

  return C, B, R, obj
end
