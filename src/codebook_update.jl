
### Methods for updating codebooks in ChainQ, LSQ, and LSQ++
### Julieta

# Update a dimension of a codebook using conjugate gradient either LSQR or LSMR
function updatecb!(
  K::SharedMatrix{Float32},
  C::SparseMatrixCSC{Int32,Int32},
  X::Matrix{Float32},
  IDX::UnitRange{Int64},
  codebook_upd_method::AbstractString="lsqr")   # choose the codebook update method out of lsqr or lsmr

  if codebook_upd_method == "lsqr"
    for i = IDX
      K[i,:] = IterativeSolvers.lsqr( C, vec(X[i, :]) )
    end
  elseif codebook_upd_method == "lsmr"
    for i = IDX
      K[i,:] = IterativeSolvers.lsmr( C, vec(X[i, :]) )
    end
  else
    error("Codebook update method unknown: ", codebook_upd_method)
  end
end

# Same as above but sparse matrix is parameterized differently
function updatecb!(
  K::SharedMatrix{Float32},
  C::SparseMatrixCSC{Float32,Int32},
  X::Matrix{Float32},
  IDX::UnitRange{Int64},
  codebook_upd_method::AbstractString="lsqr")   # choose the codebook update method out of lsqr or lsmr

  if codebook_upd_method == "lsqr"
    for i = IDX
      K[i,:] = IterativeSolvers.lsqr( C, vec(X[i, :]) )
    end
  elseif codebook_upd_method == "lsmr"
    for i = IDX
      K[i,:] = IterativeSolvers.lsmr( C, vec(X[i, :]) )
    end
  else
    error("Codebook update method unknown: ", codebook_upd_method)
  end
end

# Naive codebook update method. Super slow and nobody should use it
function update_codebooks_naive(
  X::Matrix{Float32}, # d-by-n matrix to update codebooks on.
  B::Matrix{Int16},   # m-by-n matrix. X encoded.
  h::Integer,         # number of entries per codebook.
  V::Bool=false)      # whether to print progress

  if V print("Doing Naive codebook update... "); st=time(); end

  m, _ = size( B )
  B_ok = sparsify_codes(B, h)
  C = convert( Matrix{Float32}, full(B_ok) \ X')
  if V @printf("done in %.3f seconds.\n", time()-st); end
  return K2vec( C', m, h )
end

# Naive implementation of codebook update based on Cholesky decomposition
function update_codebooks_fast(
  X::Matrix{Float32}, # d-by-n matrix to update codebooks on.
  B::Matrix{Int16},   # m-by-n matrix. X encoded.
  h::Integer,         # number of entries per codebook.
  V::Bool=false,      # whether to print progress
  rho::AbstractFloat=1e-4) # regularization

  if V print("Doing fast codebook update... "); st=time(); end

  m, _ = size( B )

  # size( B_ok ) = n x mh
  B_ok = sparsify_codes(B, h)

  BTB = B_ok'*B_ok
  BTB = convert( Matrix{Float32}, full( BTB ))

  BTXT = (X*B_ok)'

  # Solve sub-problem to solve C
  A = BTB+rho*I
  b = BTXT

  # size( C ) = mh x d
  C = convert( Matrix{Float32}, A \ b )

  if V @printf("done in %.3f seconds.\n", time()-st); end

  return K2vec( C', m, h )
end

# Same as above but matrix multiplications are accelerated by taking advantage
# of the binary structure of B
function update_codebooks_fast_bin(
  X::Matrix{Float32}, # d-by-n matrix to update codebooks on.
  B::Matrix{Int16},   # m-by-n matrix. X encoded.
  h::Integer,         # number of entries per codebook.
  V::Bool=false,      # whether to print progress
  rho::Float64=1e-4) # regularization

  if V print("Doing fast bin codebook update... "); st=time(); end

  m, n = size( B )
  d, n = size( X )

  Bm = Vector{Vector{Int16}}(m)
  for i = 1:m
    Bm[i] = B[i,:]
  end

  BTB = Matrix{Matrix{Float64}}(m,m)

  # Diagonals are easy
  hi = zeros( Float32, h )
  @inbounds for i = 1:m
    @simd for j = 1:h; hi[j]=0; end
    for j = 1:n; hi[ B[i,j] ] += 1; end
    BTB[i,i] = diagm(hi)
  end

  # Loop on off-diagonal quadrants
  @inbounds for i = 1:m
    Bmi = Bm[i]

    for j = (i+1):m

      Bmj = Bm[j]
      cij = zeros(Float32, h, h)

      for k=1:n
        cij[ Bmj[k], Bmi[k] ] += 1
      end

      # BTB is symmetric
      BTB[i,j] = cij
      BTB[j,i] = cij'
    end
  end

  # Concatenate into a mh x mh matrix
  BTB = hvcat(m, BTB...)

  BXT = Vector{Matrix{Float64}}(m)
  @inbounds for i=1:m
    BXTi = zeros( Float32, d, h)
    Bmi = Bm[i]
    for j=1:n
      Bmij = Bmi[j]
      @simd for k=1:d
        BXTi[ k, Bmij ] += X[k,j]
      end
    end
    BXT[i] = BXTi
  end

  BXT = hcat( BXT... )'

  # Solve with Cholesky
  A = BTB+rho*I
  b = BXT

  # Slightly more naive way to do it
  # C = A \ b

  # More low-level (avoids checks) and less readable but more efficient.
  # See https://software.intel.com/en-us/mkl-developer-reference-c-getrf
  lpt = LAPACK.getrf!(A)
  # See https://software.intel.com/en-us/mkl-developer-reference-c-getrs
  C = LAPACK.getrs!('N', lpt[1], lpt[2], b)

  # size( C ) = mh x d
  C = convert(Matrix{Float32}, C)

  if V @printf("done in %.3f seconds.\n", time()-st); end

  return K2vec( C', m, h )
end

###############################################
## Codebook update for a fully-connected MRF ##
###############################################

function update_codebooks(
  X::Matrix{Float32}, # d-by-n matrix to update codebooks on.
  B::Matrix{Int16},   # m-by-n matrix. X encoded.
  h::Integer,         # number of entries per codebook.
  V::Bool=false,      # whether to print progress
  method::AbstractString="fastbin")   # choose the codebook update method out of lsqr or lsmr

  if !(method in ["fast", "fastbin", "lsmr", "lsqr", "naive"]); error("Codebook update method unknown"); end

  if method == "fast"
    return update_codebooks_fast(X, B, h, V)
  elseif method == "fastbin"
    return update_codebooks_fast_bin(X, B, h, V)
  elseif method == "naive"
    return update_codebooks_naive(X, B, h, V)
  else # (method in ["lsqr", "lsmr"])

    if V print("Doing " * uppercase(method) * " codebook update... "); st=time(); end

    d, n   = size(X)
    m, _   = size(B)
    C = sparsify_codes( B, h )

    K = SharedArray{Float32}(d, size(C,2))
    if nworkers() == 1
      updatecb!( K, C, X, 1:d )
    else
      paridx = splitarray( 1:d, nworkers() )
      @sync begin
        for (i, wpid) in enumerate( workers() )
          @async begin
            remotecall_wait( updatecb!, wpid, K, C, X, paridx[i], method )
          end #@async
        end #for
      end #@sync
    end

    new_C = K2vec( K, m, h )

    if V @printf("done in %.3f seconds.\n", time()-st); end

    return new_C
  end
end # function update_codebooks

function get_cbdims_chain(
  d::Integer, # The number of dimensions
  m::Integer) # The number of codebooks

  subdims = splitarray(1:d, m-1)
  odims = Vector{UnitRange{Integer}}(m)

  odims[1] =  subdims[1]
  for i = 2:m-1
    odims[i] = subdims[i-1][1]:subdims[i][end]
  end
  odims[end] = subdims[end]

  return odims
end

# Update a dimension of a codebook using LSQR
function updatecb_struct!(
  K::SharedMatrix{Float32},
  C::SparseMatrixCSC{Float32,Int32},
  X::Matrix{Float32},
  dim2C, #::Vector{Bool},
  subcbs,
  IDX::UnitRange{Int64} )

  for i = IDX
    rcbs      = cat( 1, subcbs[find(dim2C[i,:])]... )
    K[i,rcbs] = IterativeSolvers.lsqr( C[:,rcbs], vec(X[i, :]) )
  end
end

function update_codebooks_generic(
  X::Matrix{Float32},  # d-by-n. The data that was encoded.
  B::Union{Matrix{Int16},SharedArray{Int16,2}}, # d-by-m. X encoded.
  h::Integer,          # number of entries per codebooks
  odimsfunc::Function, # Function that says which dimensions each codebook has
  V::Bool=false)       # whether to print progress

  if V print("Doing LSQR codebook update... "); st=time(); end

  d, n   = size( X )
  m, _   = size( B )
  C = sparsify_codes( B, h )

  odims = odimsfunc(d, m)

  # Make a map of dimensions to codebooks
  dim2C = zeros(Bool, d, m)
  for i = 1:m; dim2C[ odims[i], i ] = true; end

  K = SharedArray{Float32,2}((d, h*m))
  subcbs = splitarray(1:(h*m), m)

  if nworkers() == 1
    updatecb_struct!( K, C, X, dim2C, subcbs, 1:d )
  else
    paridx = splitarray( 1:d, nworkers() )
    @sync begin
      for (i, wpid) in enumerate( workers() )
        @async begin
          remotecall_wait(updatecb_struct!, wpid, K, C, X, dim2C, subcbs, paridx[i] )
        end #@async
      end #for
    end #@sync
  end

  new_C = K2vec( K, m, h )
  if V @printf("done in %.3f seconds.\n", time()-st) end

  return new_C
end

# Update codebooks in a chain
function update_codebooks_chain(
  X::Matrix{Float32}, # d-by-n. The data that was encoded.
  B::Union{Matrix{Int16},SharedArray{Int16,2}}, # d-by-m. X encoded.
  h::Integer,         # number of entries per codebook.
  V::Bool=false)      # whether to print progress

  tic()
  C = update_codebooks_generic(X, B, h, get_cbdims_chain, V)

  return C, toq()
end
