
export linscan_pq, linscan_opq, linscan_lsq, eval_recall, linscan_cq

# Linear scan using PQ codebooks no rotation
function linscan_pq(
  B::Matrix{UInt8},          # m-by-n. The database, encoded
  X::Matrix{Cfloat},         # d-by-nq. The queries.
  C::Vector{Matrix{Cfloat}}, # The cluster centers
  b:: Int,                   # Number of bits per code -- log2(h) * m
  k:: Int = 10000)           # Number of knn results to return

  m, n  = size( B )
  d, nq = size( X )

  @show k, nq
  dists = zeros( Cfloat, k, nq )
  res   = zeros(  Cuint, k, nq )

  ccall(("linscan_aqd_query", linscan_aqd), Nothing,
    (Ptr{Cfloat}, Ptr{Cuint}, Ptr{Cuchar}, Ptr{Cfloat},
    Ptr{Cfloat}, Cint, Cuint, Cint, Cint, Cint, Cint, Cint),
    dists, res, B, cat(3,C...), X, Cint(n), Cuint(nq),
    Cint(b), Cint(k), Cint(m), Cint(d), Cint(d/m) )

  return dists, (res.+=1)
end

function linscan_pq(
  B::Matrix{T},          # m-by-n. The database, encoded
  X::Matrix{Cfloat},         # d-by-nq. The queries.
  C::Vector{Matrix{Cfloat}}, # The cluster centers
  b:: Int,                   # Number of bits per code -- log2(h) * m
  k:: Int = 10000)  where T <: Integer

  B_uint8 = convert(Matrix{UInt8},B-1)
  return linscan_pq(B_uint8, X, C, b, k)
end

"My attempt to implement linscan in julia. Very slow because sortperm is slow and no multithreading"
function linscan_pq_julia(
  B::Matrix{Int16},          # m-by-n. The database, encoded
  X::Matrix{Cfloat},         # d-by-nq. The queries.
  C::Vector{Matrix{Cfloat}}, # The cluster centers
  knn:: Int = 10000)           # Number of knn results to return

  m, n  = size( B )
  d, nq = size( X )
  subdims = splitarray( 1:d, m )

  @show knn, nq
  dists = zeros( Cfloat, knn, nq )
  idx   = zeros(  Cuint, knn, nq )
  Bt = B'

  # Compute distance tables between queries and codebooks
  tables = Vector{Matrix{Cfloat}}(m)
  for i = 1:m
    # tables[i] = Distances.pairwise(Distances.SqEuclidean(), X[subdims[i],:], C[i])
    tables[i] = Distances.pairwise(Distances.SqEuclidean(), C[i], X[subdims[i],:])
  end

  # Compute approximate distances and sort
  @inbounds for i = 1:nq # Loop over each query

    # println(i)
    xq_dists  = zeros(Cfloat, n)
    p         = zeros(Cuint, n)

     for j = 1:m # Loop over each codebook
      t = tables[j][:,i]

      for k = 1:n # Loop over each code
        xq_dists[k] += t[ Bt[k,j] ]
      end
    end

    # p = sortperm(xq_dists; alg=PartialQuickSort(knn))
    sortperm!(p, xq_dists; alg=PartialQuickSort(knn))
    # @simd for j=1:n; p[j]=j; end
    # sortperm!(p, xq_dists; alg=PartialQuickSort(knn), initialized=true)
    # sort(xq_dists; alg=PartialQuickSort(knn))

    dists[:,i] = xq_dists[ p[1:knn] ]
    idx[:,i]   = p[1:knn]

  end # @inbounds

  return dists, idx

end

# Linear scan using OPQ
function linscan_opq(
  B::Matrix{UInt8},           # m-by-n. The database, encoded
  X::Matrix{Cfloat},         # d-by-nq. The queries.
  C::Vector{Matrix{Cfloat}}, # The cluster centers
  b::Int,                     # Number of bits per code -- log2(h) * m
  R::Matrix{Cfloat},         # Rotation matrix
  k::Int = 10000)             # Number of knn results to return

  # Rotate and call the function as usual
  return linscan_pq( B, R'*X, C, b, k )
end

function linscan_opq(
  B::Matrix{T},           # m-by-n. The database, encoded
  X::Matrix{Cfloat},         # d-by-nq. The queries.
  C::Vector{Matrix{Cfloat}}, # The cluster centers
  b::Int,                     # Number of bits per code -- log2(h) * m
  R::Matrix{Cfloat},         # Rotation matrix
  k::Int = 10000) where T <: Integer # Number of knn results to return

  B_uint8 = convert(Matrix{UInt8},B-1)
  return linscan_opq( B_uint8, X, C, b, R, k )
end

# Linear scan using LSQ, with dbnorms not encoded.
function linscan_lsq(
  B::Matrix{UInt8},           # m-by-n. The database, encoded
  X::Matrix{Cfloat},         # d-by-nq. The queries.
  C::Vector{Matrix{Cfloat}}, # (m-1)-long. The cluster centers
  dbnorms::Vector{Cfloat},   # h-long. Table with database norms
  R::Matrix{Cfloat},         # Rotation matrix
  k::Int = 10000)             # Number of knn results to return

  RX = R' * X;

  m, n  = size( B );
  d, nq = size( RX );
  _, h  = size( C[1] );

  dists = zeros( Cfloat, k, nq );
  res   = zeros(  Cuint, k, nq  );

  ccall(("linscan_aqd_query_extra_byte", linscan_aqd_pairwise_byte), Nothing,
    (Ptr{Cfloat}, Ptr{Cint},
    Ptr{Cuchar}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},
    Cuint, Cint, Cint, Cint, Cint, Cint),
    dists, res,
    B, RX, hcat(C...), dbnorms,
    Cint(nq), Cint(n), Cint(m), Cint(h), Cint(d), Cint(k) );

  return dists, res
end

# Linear scan using LSQ, with dbnorms not encoded.
function linscan_lsq(
  B::Matrix{T},           # m-by-n. The database, encoded
  X::Matrix{Cfloat},         # d-by-nq. The queries.
  C::Vector{Matrix{Cfloat}}, # (m-1)-long. The cluster centers
  dbnorms::Vector{Cfloat},   # h-long. Table with database norms
  R::Matrix{Cfloat},         # Rotation matrix
  k::Int = 10000) where T <: Integer # Number of knn results to return

  B_uint8 = convert(Matrix{UInt8},B-1)
  return linscan_lsq(B_uint8, X, C, dbnorms, R, k)
end

# Linear scan using Composite Quantization (Zhang et al, ICML 14)
function linscan_cq(
  B::Matrix{UInt8},           # m-by-n. The database, encoded
  X::Matrix{Cfloat},         # d-by-nq. The queries.
  C::Vector{Matrix{Cfloat}}, # (m-1)-long. The cluster centers
  k::Int = 10000)             # Number of knn results to return

  m, n  = size( B );
  d, nq = size( X );
  _, h  = size( C[1] );

  dists = zeros( Cfloat, k, nq );
  res   = zeros(  Cuint, k, nq  );

  ccall(("linscan_aqd_cq_query_extra_byte", linscan_aqd_pairwise_byte), Nothing,
    (Ptr{Cfloat}, Ptr{Cint},
    Ptr{Cuchar}, Ptr{Cfloat}, Ptr{Cfloat}, #Ptr{Cfloat},
    Cuint, Cint, Cint, Cint, Cint, Cint),
    dists, res,
    B, X, hcat(C...), #dbnorms,
    Cint(nq), Cint(n), Cint(m), Cint(h), Cint(d), Cint(k) );

  return dists, res
end

# Linear scan using Composite Quantization (Zhang et al, ICML 14)
function linscan_cq(
  B::Matrix{T},           # m-by-n. The database, encoded
  X::Matrix{Cfloat},         # d-by-nq. The queries.
  C::Vector{Matrix{Cfloat}}, # (m-1)-long. The cluster centers
  k::Int = 10000) where T <: Integer # Number of knn results to return

  B_uint8 = convert(Matrix{UInt8},B-1)
  return linscan_cq(B_uint8, X, C, k)
end

# Evaluate ANN search vs ground truth. Produces a recall@N curve.
function eval_recall(ids_gnd::Vector{T}, ids_predicted::Matrix{T}, k::Integer) where T <: Integer

  # Modified from Hervé Jégou's test_compute_stats.m matlab code.

  nquery = size( ids_predicted, 2 );
  assert( nquery == length( ids_gnd) );

  nn_ranks = zeros( nquery );
  hist_pqc = zeros( k+1 );

  for i = 1:nquery
    gnd_ids = ids_gnd[i];
    nn_pos = find( ids_predicted[:,i] .== gnd_ids );

    if length(nn_pos) == 1
      nn_ranks[i] = nn_pos[1];
    else
      nn_ranks[i] = k+1;
    end
  end

  nn_ranks = sort( nn_ranks );

  recall_at_i = zeros( k );

  for i = [1 2 5 10 20 50 100 200 500 1000 2000 5000 10000]
    if i <= k
      recall_at_i[i] = length( find( (nn_ranks .<= i) .& (nn_ranks .<= k) )) ./ nquery * 100;
      println("r@$(i) = $(recall_at_i[i])");
    end
  end

  for i = 1:k
    recall_at_i[i] = length(find( (nn_ranks .<= i) .& (nn_ranks .<= k) )) ./ nquery;
  end

  return recall_at_i

end
