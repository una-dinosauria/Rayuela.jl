export quantize_norms, splitarray, K2vec, quantize_norms, get_norms_codebook

"Get the codebook of the norms with k-means"
function get_norms_codebook(
  B::Matrix{T1},
  C::Vector{Matrix{T2}}) where {T1<:Integer, T2<:AbstractFloat}

  m, n = size(B)
  d, h = size(C[1])

  # Make sure there are m codebooks
  @assert m == length(C)

  # Reconstruct the approximation and compute its norms
  CB      = reconstruct(B, C)
  dbnorms = sum(CB.^2, dims=1)

  # Quantize the norms with k-means
  dbnormsq = Clustering.kmeans(dbnorms, h)

  norms_codes     = vec(reshape(dbnormsq.assignments, 1, n)[:])
  norms_codebook  = vec(dbnormsq.centers)

  # Add the dbnorms to the codes
  return norms_codes, norms_codebook
end

"Quantize the norms of an encoding"
function quantize_norms(
  B::Matrix{T1},         # m-by-n. Codes
  C::Vector{Matrix{T2}}, # m-long h-by-d each. Codebooks
  cbnorms::Vector{T2}) where {T1<:Integer, T2<:AbstractFloat} # h-long. norms codebooks

  CB = reconstruct(B,C)
  d, n = size( CB )
  _, h = size( C[1] )

  dbnormsB   = Vector{T1}(undef, n)
  dbnormsX   = Vector{T2}(undef, n)
  dists2norm = Vector{T2}(undef, h)

  @inbounds for i = 1:n
    ithnorm::T2 = zero(T2)

    @simd for j = 1:d
      ithnorm += CB[j,i].^2
    end
    dbnormsX[i] = ithnorm

    for j = 1:h
      dists2norm[j] = (ithnorm - cbnorms[j]).^2
    end

    # Store of the index of the minimum distance
    _, dbnormsB[i] = findmin( dists2norm )
  end

  return dbnormsB, dbnormsX
end

# Compute dot products between codebooks
function comp_prod_C(C::Vector{Matrix{T}}) where T <: AbstractFloat  # Codebooks
  m   = length(C)
  dps = Matrix{Matrix{T}}(m, m) # dot products
  for i = 1:m, j = (i+1):m
    dps[i,j] = 2 * C[i]' * C[j]
  end
  return dps
end

############################################
## Functions that support codebook update ##
############################################

# Creates a sparse matrix out of codes
function sparsify_codes(
  B::Matrix{T}, # m-by-n matrix. Codes to sparsify
  h::Integer) where T <: Integer # Number of entries per codebook

  m, n   = size(B)
  ncodes = length(B)

  # Storing the indices of the sparse matrix.
  I = zeros(Int32, ncodes) # row indices
  J = zeros(Int32, ncodes) # column indices

  for i = 1:m
    I[ (i-1)*n+1 : i*n ] = 1:n
    J[ (i-1)*n+1 : i*n ] = vec(B[i,:]) .+ (i-1) * h
  end

  C = sparse(I, J, ones(Float32, ncodes), n, m * h)

  return C
end

# Transform the output of LSQR/SPGL1 into a vector of matrices again
function K2vec(
  K::Union{SharedMatrix{Float32}, Matrix{Float32}}, # d-by-(h*m) matrix with the new codebooks
  m::Integer, # Number of codebooks
  h::Integer) # Number of elements in each codebook

  @assert size(K,2) == m*h

  C = Vector{Matrix{Float32}}(undef, m)

  subdims = splitarray(1:(h*m), m)
  for i = 1:m
    C[i] = sdata(K[ :, subdims[i] ])
  end

  return C
end

###########################################
## Functions to get unaries and binaries ##
###########################################

# Get unaries terms
function get_unaries(
  X::Matrix{T},         # data to get unaries from
  C::Vector{Matrix{T}}, # codebooks
  V::Bool = false) where T <: AbstractFloat

  d, n = size( X )
  m    = length(C)
  _, h = size( C[1] )

  unaries = Vector{Matrix{T}}(undef, m)
  #if V print("Computing unaries... "); st=time(); end


  @inbounds for i = 1:m
    unaries[i] = -2 * C[i]' * X
    sci        = diag( C[i]' * C[i] )
    #unaries[i] = broadcast(+, sci, unaries[i])

    ui = unaries[i]
    for j = 1:n
      @simd for k = 1:h
        ui[k, j] += sci[k]
      end
    end
  end

  #if V @printf("done in %.3f seconds.\n", time()-st); end
  return unaries
end

# Computes all the binaries for a set of codebooks
function get_binaries(
  C::Vector{Matrix{T}}) where T <: AbstractFloat # codebooks

  m = length( C )

  ncbi     = sum(1:(m-1)) # The number of pairwise interactions
  binaries = Vector{Matrix{T}}(undef, ncbi) # We'll store the binaries here
  cbi      = zeros(Int32, 2, ncbi) # Store the pairwise indices here

  idx=1
  for i = 1:m
    for j = (i+1):m
      binaries[idx] = 2 * C[i]' * C[j]
      cbi[:,idx]    = [i;j]
      idx = idx + 1
    end
  end

  return binaries, cbi
end

# Split an array in equal parts
# if the number of x is a multiple of nparts, then each parts has the same number of
# elements.
# if that's not the case, then the n rest elements, e.t. the rest of the division, will be
# added to the first n parts. So the first n parts contain each 1 item more than the the
# parts from n + 1 to nparts
function splitarray(
  x::Union{UnitRange, Vector}, # Vector to split in equal parts
  nparts::Integer)             # Number of parts to split the vector on

  n       = length(x)
  perpart = div(n, nparts)
  xtra    = mod(n, nparts)
  out     = Array{Any}(undef, nparts)

  #fills the parts, which have 1 item more than the other parts,
  #so these parts have size perpart + 1
  glidx = 1
  for i = 1:xtra
    out[i] = x[glidx : (glidx+perpart)]
    glidx  = glidx+perpart+1
  end

  #fills the parts with the normal size
  for i = (xtra+1):nparts
    out[i] = x[glidx : (glidx+perpart-1)]
    glidx  = glidx+perpart
  end

  return out
end
