
export train_ervq, quantize_ervq

"""
    quantize_ervq(X::Matrix{T}, C::Vector{Matrix{T}}, V::Bool=false) where T <: AbstractFloat

Quantize using a residual quantizer
"""
function quantize_ervq(
  X::Matrix{T},         # d-by-n. Data to encode
  C::Vector{Matrix{T}}, # codebooks
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  # The quantization method is the same as RVQ
  quantize_rvq(X,C,V)
end

"""
    train_ervq(X::Matrix{T}, m::Integer, h::Integer, V::Bool=false) where T <: AbstractFloat

Trains a residual quantizer.
"""
function train_ervq(
  X::Matrix{T},  # d-by-n. Data to learn codebooks from
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of k-means iterations for training
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  # ERVQ starts with RVQ
  C, B, error = train_rvq(X,m,h,niter,V)
  B = convert(Matrix{Int64}, B)

  # Then we do the fine-tuning part of https://arxiv.org/abs/1411.2173
  if V print("Error after init is $error \n"); end

  for i = 1:niter
    if V print("=== Iteration $i / $niter ===\n"); end

    # Dummy singletons
    singletons = Vector{Vector{Int}}(2)
    singletons[1] = Vector{Int}()
    singletons[2] = Vector{Int}()

    Xr = copy(X)
    Xd = X .- reconstruct(B[2:end,:],C[2:end])

    for j=1:m
      if V print("Updating codebook $j... "); end

      if j == m
        Xd = Xr .- reconstruct(B[j-1,:],C[j-1])
      elseif j > 1
        Xd = Xr .- reconstruct( vcat(B[j-1,:]', B[j+1:end,:]), [C[j-1],C[j+1:end]...] )
      end

      # Update the codebook C[j]
      to_update = zeros(Bool,h)
      to_update[B[j,:]] = true

      # Check if some centres are unasigned
      Clustering.update_centers!(Xd, nothing, B[j,:], to_update, C[j], zeros(T,h))

      # TODO Assert that the number of singletons is enough
      if sum(to_update) < h
        ii = 1
        for idx in find(.!to_update)
          C[j][:,idx] = singletons[2][:,ii]
          ii = ii+1
        end
      end
      if V print("done.\n"); end

      # Update the residual
      if j > 1
        Xr .-= reconstruct(B[j-1,:], C[j-1])
      end

      if V print("Updating codes... "); end
      B[j:end,:], singletons = quantize_ervq(Xr, C[j:end], V)
      if V print("done. "); end
      error = qerror(X, B, C)
      print("Qerror is $error.\n")

    end # End loop over codebooks

    if V
      error = qerror(X, B, C)
      print("Iteration $i / $niter done. Qerror is $error.\n");
    end

  end
  return C, B, 0.0
end
