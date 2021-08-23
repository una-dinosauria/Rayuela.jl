
export train_chainq, quantize_chainq, experiment_chainq

"Quantize using Vitebi algorithm implemented in c++"
function quantize_chainq_cpp!(
  CODES::Matrix{Int16},  # out. Where to save the result
  X::Matrix{T},                # in. d-by-n matrix to encode
  C::Vector{Matrix{T}},        # in. m-long vector with d-by-h codebooks
  binaries::Vector{Matrix{T}}, # in. Binary terms
  IDX::UnitRange{Int64}) where T <: AbstractFloat # in. Index to save the result

  # Get unaries
  unaries = get_unaries( X, C )

  h, n = size(unaries[1])
  m    = length(C)

  if h != 256
    error("The C++ implementation of chain quantization encoding only supports
    codebooks with 256 entries")
  end

  CODES2 = zeros(Cuchar, m, n)
  unaries2, binaries2 = vcat(unaries...), hcat(binaries...)

  ccall(("viterbi_encoding", encode_icm_so), Nothing,
    (Ptr{Int16}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint),
    CODES2, unaries2, binaries2, n, m)

  CODES2 = convert(Matrix{Int16}, CODES2) .+ one(Int16)
  CODES[:] = CODES2[:]
end


"Quantize using Vitebi algorithm implemented in julia"
function quantize_chainq!(
  CODES::SharedMatrix{Int16},  # out. Where to save the result
  X::Matrix{T},                # in. d-by-n matrix to encode
  C::Vector{Matrix{T}},        # in. m-long vector with d-by-h codebooks
  binaries::Vector{Matrix{T}}, # in. Binary terms
  IDX::UnitRange{Int64}) where T <: AbstractFloat # in. Index to save the result

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


"Quantize using Vitebi algorithm implemented in julia. This algorithm is batched and thus may be faster, but in practice
it's about as fast as the sequential version."
function quantize_chainq_batched!(
  CODES::SharedMatrix{Int16},  # out. Where to save the result
  X::Matrix{T},                # in. d-by-n matrix to encode
  C::Vector{Matrix{T}},        # in. m-long vector with d-by-h codebooks
  binaries::Vector{Matrix{T}}, # in. Binary terms
  IDX::UnitRange{Int64}) where T <: AbstractFloat # in. Index to save the result

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
    ucost = unaries[i]

    bb = binaries[i]
    for j = 1:h # Loop over the cost of going to j
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


"Quantize using Vitebi algorithm implemented in CUDA"
function quantize_chainq_cuda!(
  CODES::Matrix{Int16},  # out. Where to save the result
  X::Matrix{T},                # in. d-by-n matrix to encode
  C::Vector{Matrix{T}},        # in. m-long vector with d-by-h codebooks
  binaries::Vector{Matrix{T}}, # in. Binary terms
  IDX::UnitRange{Int64}) where T <: AbstractFloat # in. Index to save the result

  d, n = size( X )
  _, h = size( C[1] )
  m    = length( binaries ) + 1

  # We need a matrix to keep track of the min and argmin
  mincost = zeros(T, h, n)
  # minidx  = zeros(Int32, h, m, n)
  minidx = convert(Array{Int32}, rand(1:h, h, m, n))

  # Allocate memory for brute-forcing each pair
  cost = zeros(T, h, n)

  minv = zeros(T, n)
  mini = zeros(Int32, n)

  # Setup GPU stuff
  gpuid = 0
  CudaUtilsModule.init(gpuid, cudautilsptx)

  # Make space for unaries
  d_X = CuArrays.CuArray(X)
  d_unaries   = Vector{CuArrays.CuArray{T}}(undef, m)
  d_codebooks = Vector{CuArrays.CuArray{T}}(undef, m)
  for j = 1:m
    d_codebooks[j] = CuArrays.CuArray(C[j])

    # -2 * C' * X
    # d_unaries[j] = -2.0f0 * d_codebooks[j]' * d_RX <-- thus runs out of memory real fast
    d_unaries[j] = CuArrays.BLAS.gemm('T', 'N', -2.0f0, d_codebooks[j], d_X)

    # Add self-codebook interactions || C_{i,i} ||^2
    CudaUtilsModule.vec_add(n, (1,h), d_unaries[j].buf, CuArrays.CuArray(sum(C[j].^2, dims=1)).buf, Cint(n), Cint(h))
  end

  d_binaries = Vector{CuArrays.CuArray{T}}(undef, length(binaries))
  for i = 1:length(binaries)
    d_binaries[i] = CuArrays.CuArray(binaries[i])
  end

  d_mincost = CuArrays.CuArray(mincost)
  d_mini = CuArrays.CuArray(mini)

  # Forward pass
  @inbounds for i = 1:(m-1) # Loop over states
    if i > 1; CuArrays.BLAS.axpy!(n * h, 1.0f0, d_mincost, 1, d_unaries[i], 1); end

    for j = 1:h # Loop over the cost of going to j
      CudaUtilsModule.viterbi_forward(n, (1, h), d_unaries[i].buf, d_binaries[i].buf, d_mincost.buf, d_mini.buf, Cint(n), Cint(j-1))

      Mem.download!(mini, d_mini.buf)
      minidx[j,i,:] .= mini .+ one(eltype(mini))
    end
  end

  CuArrays.BLAS.axpy!(n * h, 1.0f0, d_mincost, 1, d_unaries[m], 1)
  CudaUtilsModule.viterbi_forward(n, (1, h), d_unaries[m].buf, CuArrays.CuArray(zeros(Float32, h, h)).buf, d_mincost.buf, d_mini.buf, Cint(n), Cint(0))
  Mem.download!(mini, d_mini.buf)
  mini .+= one(eltype(mini))

  # Backward trace
  @inbounds for idx = IDX # Loop over the datapoints

    backpath = [ mini[idx] ]
    for i = (m-1):-1:1
      push!(backpath, minidx[ backpath[end], i, idx ])
    end

    # Save the inferred code
    CODES[:, idx] .= reverse!( backpath )
  end

  CudaUtilsModule.finit()

  nothing
end


"""
    quantize_chainq(X, C, use_cuda=false, use_cpp=false) -> B, ellapsed

Given data and chain codebooks, find codes using the Viterbi algorithm chain quantizer.

# Arguments
- `X::Matrix{T}`: `d`-by-`n` data to quantize
- `C::Vector{Matrix{T}}`: `m`-long vector with `d`-by-`h` matrices. Each matrix is a pretrained codebook of size approximately `d`-by-`h`.
- `use_cuda::Bool`: whether to use a CUDA implementation
- `use_cpp::Bool`: whether to use a c++ implementation

If both `use_cuda` and `use_cpp` are `true`, the CUDA implementation is used.

# Returns
- `B::Matrix{Int16}`: `m`-by-`n` matrix with the codes
- `ellapsed::Float64`: The time spent encoding
"""
function quantize_chainq(
  X::Matrix{Float32},         # d-by-n matrix. Data to encode
  C::Vector{Matrix{Float32}}, # m-long vector with d-by-h codebooks
  use_cuda::Bool=false,
  use_cpp::Bool=false)

  start_time = time_ns()
  d, n = size(X)
  m    = length(C)

  # Compute binary tables
  binaries = Vector{Matrix{Float32}}(undef, m-1)
  for i = 1:(m-1)
    binaries[i] = 2 * C[i]' * C[i+1]
  end

  CODES = SharedArray{Int16,2}(m, n)
  CODES .= zero(Int16)

  if use_cuda # CUDA version
    quantize_chainq_cuda!(sdata(CODES), X, C, binaries, 1:n)
    GC.gc()
  elseif nworkers() == 1
    if use_cpp # C++ parallel version
      quantize_chainq_cpp!(sdata(CODES), X, C, binaries, 1:n)
      # quantize_chainq_batched!(CODES, X, C, binaries, 1:n)
    else # Julia single-threaded version
      quantize_chainq!(CODES, X, C, binaries, 1:n)
    end
  else # Julia-paralell version
    paridx = splitarray(1:n, nworkers())
    @sync begin
      for (i,wpid) in enumerate(workers())
        @async begin
          Xw = X[:,paridx[i]]
          remotecall_wait(quantize_chainq!, wpid, CODES, Xw, C, binaries, paridx[i])
        end
      end
    end
  end

  B = sdata(CODES)
  B, (time_ns() - start_time)/1e9
end


"""
    train_chainq(X, m, h, R, B, C, niter, V=false) -> C, B, R, error

Train a chain quantizer.
This method is typically initialized by [Optimized product quantization (OPQ)](@ref)

# Arguments
- `X::Matrix{T}`: `d`-by-`n` data to quantize
- `m::Integer`: Number of codebooks
- `h::Integer`: Number of entries in each codebook (typically 256)
- `R::Matrix{T}`: `d`-by-`d` rotation matrix for initialization
- `B::Matrix{Int16}`: `m`-by-`n` matrix with pre-trained codes for initialization
- `C::Vector{Matrix{T}}`: `m`-long vector with `d`-by-`h` matrices. Each matrix is a pretrained codebook of size approximately `d`-by-`h`.
- `niter::Integer`: Number of iterations to use
- `V::Bool`: Whether to print progress

# Returns
- `C::Vector{Matrix{T}}`: `m`-long vector with `d`-by-`h` matrix entries. Each matrix is a codebook of size approximately `d`-by-`h`.
- `B::Matrix{Int16}`: `m`-by-`n` matrix with the codes
- `R::Matrix{T}`: `d`-by-`d` optimized rotation matrix
- `error::T`: The quantization error after training
"""
function train_chainq(
  X::Matrix{T},             # d-by-n matrix of data points to train on.
  m::Integer,               # number of codebooks
  h::Integer,               # number of entries per codebook
  R::Matrix{T},             # Init rotation matrix
  B::Matrix{Int16},         # Init codes
  C::Vector{Matrix{T}},     # Init codebooks
  niter::Integer,           # number of optimization iterations
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  if V; @printf("Training a chain quantizer\n"); end

  d, n = size(X)
  obj  = zeros(T, niter+1)

  # TODO expose these
  use_cuda = false
  use_cpp = false

  CB = zeros(T, size(X))
  RX = R' * X

  # Initialize C
  # C, Ctime = update_codebooks_chain(RX, B, h)
  C, Ctime = update_codebooks_chain_bin(RX, B, h)
  if V; @printf("%3d %e... %.2f secs updating C\n", -2, qerror(RX, B, C), Ctime); end

  # Initialize B
  B, Btime = quantize_chainq(RX, C, use_cuda, use_cpp)
  if V; @printf("%3d %e... %.2f secs updating B\n", -1, qerror(RX, B, C), Btime); end

  for iter = 0:niter
    if V; start_time = time_ns(); end # Take time if asked to

    obj[iter+1] = qerror(RX, B, C)
    if V; @printf("%3d %e... ", iter, obj[iter+1]); end

    # update CB
    CB .= zero(T)
    for i = 1:m; CB .+= C[i][:, vec(B[i,:])]; end

    # update R
    U, S, VV = svd(X * CB', full=false)
    R = U * VV'

    # update R*X
    RX = R' * X

    # Update the codebooks #
    # C, Ctime = update_codebooks_chain(RX, B, h)
    C, Ctime = update_codebooks_chain_bin(RX, B, h)

    # Update the codes with lattice search
    B, Btime = quantize_chainq(RX, C, use_cuda, use_cpp)
    if V; @printf("done in %.2f secs. %.2f secs updating B with %d workers. %.2f secs updating C\n", (time_ns()-start_time)/1e9, Btime, nworkers(), Ctime); end
  end

  return C, B, R, obj
end
