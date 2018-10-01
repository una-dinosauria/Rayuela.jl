# Enhanced RVQ, aka Stacked Quantizers: https://arxiv.org/abs/1411.2173
export train_ervq, quantize_ervq, experiment_ervq

"""
    quantize_ervq(X::Matrix{T}, C::Vector{Matrix{T}}, V::Bool=false) where T <: AbstractFloat

Quantize using a residual quantizer
"""
function quantize_ervq(
  X::Matrix{T},         # d-by-n. Data to encode
  C::Vector{Matrix{T}}, # codebooks
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  # The quantization method is the same as RVQ
  quantize_rvq(X, C, V)
end

"""
    train_ervq(X::Matrix{T}, m::Integer, h::Integer, V::Bool=false) where T <: AbstractFloat

Trains an enhanced residual quantizer.
"""
function train_ervq(
  X::Matrix{T},  # d-by-n. Data to learn codebooks from
  B::Matrix{T2}, # codes
  C::Vector{Matrix{T}}, # codebooks
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of iterations for training
  V::Bool=false) where {T <: AbstractFloat, T2 <: Integer} # whether to print progress

  B = convert(Matrix{Int}, B)

  # Then we do the fine-tuning part of https://arxiv.org/abs/1411.2173
  # TODO make sure that C is full-dimensional
  error = qerror(X, B, C)
  if V print("Error after init is $error \n"); end

  for i = 1:niter
    if V print("=== Iteration $i / $niter ===\n"); start_time = time_ns(); end

    # Dummy singletons
    singletons = Vector{Matrix{T}}(undef, m)

    Xr = copy(X)
    Xd = X .- reconstruct(B[2:end,:], C[2:end])

    for j=1:m
      if V print("Updating codebook $j... "); end

      if j == m
        Xd = Xr .- reconstruct(B[j-1,:], C[j-1])
      elseif j > 1
        Xd = Xr .- reconstruct(vcat(B[j-1,:]', B[j+1:end,:]), [C[j-1],C[j+1:end]...])
      end

      # Update the codebook C[j]
      weights = nothing  # use unweighted version of update_centers!
      to_update = zeros(Bool, h)
      to_update[B[j,:]] .= true # In. Whether a codebook entry needs update
      cweights = zeros(T, h)   # Out. Cluster weights. We do not use this.
      Clustering.update_centers!(Xd, weights, B[j,:], to_update, C[j], cweights)

      # Check if some centres are unasigned
      if sum(to_update) < h && j == 1
        # If this happens with the first codebook, we have no precomputed
        # singletons, so we have to compute them ourselves.
        # Empirically this is an edge case but it has happened to me so.

        unused = .!to_update # In. Unused centres are centres that were not updated.
        costs = sum((X .- Xd).^2, 1) # In. The cost of assigning the ith vector
        Clustering.repick_unused_centers(Xd, costs, C[j], unused)

      elseif sum(to_update) < h
        # In other cases we already have the precomputed singletons
        ii = 1
        for idx in findall(.!to_update)
          C[j][:,idx] = singletons[2][:, ii]
          ii = ii + 1
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
      time_spent = (time_ns() - start_time) / 1e9
      print("Iteration $i / $niter done in $time_spent seconds. Qerror is $error.\n");
    end

  end
  return C, B, qerror(X, B, C)
end

function train_ervq(
  X::Matrix{T},  # d-by-n. Data to learn codebooks from
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of k-means iterations for training
  V::Bool=false) where T <: AbstractFloat# whether to print progress

  # Initialize with RVQ
  C, B, _ = train_rvq(X, m, h, niter, V)
  train_ervq(X, B, C, m, h, niter, V)
end


function experiment_ervq(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  B::Matrix{T2}, # codes
  C::Vector{Matrix{T}}, # codebooks
  Xb::Matrix{T}, # d-by-n. Base set
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of k-means iterations for training
  knn::Integer=1000,
  V::Bool=false) where {T <: AbstractFloat, T2 <: Integer} # whether to print progress

  # === ERVQ train ===
  d, _ = size(Xt)
  @time C, B, train_error = Rayuela.train_ervq(Xt, B, C, m, h, niter, V)
  norms_B, norms_C = get_norms_codebook(B, C)

  # === Encode the base set ===
  B_base, _ = Rayuela.quantize_ervq(Xb, C, V)
  base_error = qerror(Xb, B_base, C)
  if V; @printf("Error in base is %e\n", base_error); end

  # Compute and quantize the database norms
  B_base_norms, _ = quantize_norms( B_base, C, norms_C )
  db_norms        = vec( norms_C[ B_base_norms ] )

  if V; print("Querying m=$m ... "); end
  @time dists, idx = linscan_lsq(B_base, Xq, C, db_norms, Matrix{Float32}(1.0I, d, d), knn)
  if V; println("done"); end

  recall = eval_recall(gt, idx, knn)
  return C, B, train_error, B_base, recall
end

function experiment_ervq_query_base(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  B::Matrix{T2}, # codes
  C::Vector{Matrix{T}}, # codebooks
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of k-means iterations for training
  knn::Integer=1000,
  V::Bool=false) where {T <: AbstractFloat, T2 <: Integer} # whether to print progress

  # === ERVQ train ===
  d, _ = size(Xt)
  C, B, train_error = Rayuela.train_ervq(Xt, B, C, m, h, niter, V)
  norms_B, norms_C = get_norms_codebook(B, C)
  db_norms     = vec( norms_C[ norms_B ] )

  if V; print("Querying m=$m ... "); end
  @time dists, idx = linscan_lsq(B, Xq, C, db_norms, eye(Float32, d), knn)
  if V; println("done"); end

  recall = eval_recall(gt, idx, knn)
  return C, B, train_error, recall
end

function experiment_ervq(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  Xb::Matrix{T}, # d-by-n. Base set
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of k-means iterations for training
  knn::Integer=1000,
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  C, B, _ = Rayuela.train_rvq(Xt, m, h, niter, V)
  experiment_ervq(Xt, B, C, Xb, Xq, gt, m, h, niter, knn, V)
end

function experiment_ervq_query_base(
  Xt::Matrix{T}, # d-by-n. Data to learn codebooks from
  Xq::Matrix{T}, # d-by-n. Queries
  gt::Vector{UInt32}, # ground truth
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  niter::Integer=25, # Number of k-means iterations for training
  knn::Integer=1000,
  V::Bool=false) where T <: AbstractFloat # whether to print progress

  C, B, _ = Rayuela.train_rvq(Xt, m, h, niter, V)
  experiment_ervq_query_base(Xt, B, C, Xq, gt, m, h, niter, knn, V)
end
