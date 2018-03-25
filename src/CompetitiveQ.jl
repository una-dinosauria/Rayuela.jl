
using Clustering # For k-meas

# # Compute quantization error
# function qerror(
#   X::Matrix{Float32},
#   C::Vector{Matrix{Float32}},
#   B::Matrix{Int16})
#
#   m, n = size(B)
#   @assert length(C) == m
#
#   # Reconstruct the approximation
#   Xrec = zeros( Float32, size(X) )
#   for i = 1:m
#     Bi = B[i,:]
#     Xrec = Xrec + C[i][:,Bi]
#   end
#
#   return mean( sum((X - Xrec).^2,1) )
# end
#
# # Compute quantization error
# function qerror(
#   X::Vector{Float32},
#   C::Vector{Matrix{Float32}},
#   B::Vector{Int16})
#
#   m = size(B)
#   m = m[1]
#   @assert length(C) == m
#
#   # Reconstruct the approximation
#   Xrec = zeros( Float32, size(X) )
#   for i = 1:m
#     Bi = B[i]
#     Xrec = Xrec + C[i][:,Bi]
#   end
#
#   return mean( sum((X - Xrec).^2,1) )
# end

# Codebook update with gradient descent
function update_codebooks(
  C::Vector{Matrix{Float32}},
  m::Integer,
  xr::Vector{Float32},
  b::Vector{Int16},
  lr::Vector{Float32}) # learning rates

  # Update all the codebooks jointly
  for i = 1:m
    bi = b[i];
    C[i][:,bi] = C[i][:,bi] + 2*lr[i]*xr
  end

  return C
end

function vec_minus_mat!(
  res::Matrix{T}, # in/out. d-by-h
  x::Vector{T}, # in. d-long
  C::Matrix{T}, # in. d-by-h
  d::Integer, h::Integer) where T <: AbstractFloat

  @inbounds @simd for i=1:h
    for j=1:d
      res[j,i] = x[j] - C[j,i]
    end
  end

end

# Encoding
function encode(
  x::Vector{T},
  C::Vector{Matrix{T}},
  new_res::Vector{Matrix{T}}, # Buffer for residuals
  m::Integer,
  h::Integer,
  d::Integer,
  H::Integer) where T <: AbstractFloat

  # Get the first H candidates
  # xrs      = broadcast( -, x, C[1] ) # Get all h residuals
  # xrs = zeros(T, d, h)
  xrs = x .- C[1]
  qerrs    = vec( sum( xrs.^2, 1 ) ) # Compute the qerrors
  sort_idx = sortperm( qerrs )[1:H]  # Sort and get the top H indices
  xrs      = xrs[:, sort_idx]        # The top H residuals

  # Structure for residuals at lth level
  # new_res   = Vector{Matrix{T}}(H)
  # for i = 1:H; new_res[i] = zeros(T,d,h); end

  new_qerrs = Vector{Vector{T}}(H)
  new_bs    = zeros( Int16, H*h, m )
  # intiialize the code candidates
  for i = 1:H
    new_bs[ (i-1)*h+1 : i*h, 1 ] = sort_idx[i]
  end

  for i = 2:m

    # Compute new residuals and costs
    Ci = C[i]

    for j = 1:H
      # new_res[j]   = broadcast(-, xrs[:,j], Ci )
      # new_res[j]   = xrs[:,j] .- Ci
      vec_minus_mat!(new_res[j], xrs[:,j], Ci, d, h)
      new_qerrs[j] = vec( sum( new_res[j].^2, 1 ) )
      new_bs[ (j-1)*h+1 : j*h, i ] = 1:h
    end

    # Find the top H candidates
    all_qerrs = vcat( new_qerrs... )
    sort_idx = sortperm( all_qerrs )[1:H]  # Sort and get the top H indices

    all_res = hcat( new_res... )
    xrs     = all_res[ :,  sort_idx]
    top_bs  = deepcopy( new_bs[sort_idx, 1:i ] )

    # Reset the candidate bs
    for j = 1:H
      for k = 1:h
        new_bs[ (j-1)*h+k, 1:i ] = top_bs[j, :]
      end
    end

  end

  return new_bs[1,:], xrs[:,1]

end

# Train competitive quantization
function train_competitiveq(
  X::Matrix{T},         # d-by-n training dataset
  C::Vector{Matrix{T}}, # Initial codebook
  n_its::Integer, # Number of optimization iterations
  H::Integer,
  B::Matrix{Int16},
  init_lr_total::T) where T <: AbstractFloat# Depth of pseudo-beam search for encoding

  d, n = size(X)
  m    = length(C)
  h    = size(C[1],2)

  # Compute the learning rate for each layer (Eq. 26)
  lrs = Vector{Float32}(m)

  lr = init_lr_total

  for i = 1:m
    lrs[i] = (1  ./ ( log2(i) +  1 )) * lr
  end
  lrs = lrs ./ sum(lrs)
  lrs = lrs .* lr
  @show lrs
  @show sum(lrs)

  # Structure for residuals at lth level
  new_res   = Vector{Matrix{T}}(H)
  for i = 1:H; new_res[i] = zeros(T,d,h); end

  # @profile begin
  for i = 1:n_its

    # Encode each vector
    tic()
    for j = 1:n

      x  = X[:,j]
      bj = B[:,j]

      code_before = B[:,j]
      #qbefore = qerror(x, C, bj)

      bj, xr = encode(x, C, new_res, m, h, d, H);
      # bj, xr = encode(x, C, new_res, m, h, d, H)

      B[:,j] = bj; # Update the code

      code_after = bj


      #@show code_before
      #@show code_after

      if j % 100 == 0
        # qafter = qerror(x, bj, bj)
        #@printf( "%d, before:%.2f, after:%.2f, diff:%.2f\n", j, qbefore, qafter, qbefore-qafter )
        # @printf( "%d, after:%.2f\n", j, qafter )
        qerr = qerror(X, B, C)
        print( "Error after $i iterations / $j samples is $qerr\n" )
        print( "$(toq()) seconds since last tic\n"); tic()
      end

      # Update the codebooks
      C = update_codebooks(C, m, xr, bj, lrs)
    end

    # Compute overall quantization error
    qerr = qerror(X, B, C)
    print( "Error after $i iterations is $qerr\n" )

    # Decrease the learning rate by 1%
    lr = 0.99f0 * lr

    for i = 1:m
      lrs[i] = (1  ./ ( log2(i) +  1 )) * lr
    end
    lrs = lrs ./ sum(lrs)
    lrs = lrs .* lr
    @show lrs
    @show sum(lrs)

  end
  # end #profile


end


# === Main
Xt = fvecs_read(Int(1e4), "../data/sift/sift_learn.fvecs")
m  = 8   # Number of codebooks
h  = 256 # Number of entries in each codebook
lr = 0.5f0 # learning rate
H  = 32 # depth for search during encoding

n_its = 250

C, B = train_rvq(Xt, m, h, 25, true)
qerr = qerror(Xt, B, C)
@show B[:, 1]
print( "Error after initialization is $qerr\n" )

train_competitiveq(Xt, C, n_its, H, B, lr )
