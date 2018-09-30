export qerror, qerror_pq, qerror_opq

# === Functions that compute quantization error ===

"Reconstruct an encoding"
function reconstruct(
  B::Matrix{T1},
  C::Vector{Matrix{T2}}) where {T1<:Integer, T2<:AbstractFloat}

  m, n = size( B )
  d, _ = size( C[1] )
  CB   = zeros( T2, d, n )

  @inbounds for i = 1:m
    mcb = C[i] # Pick the mth codebook
    for j = 1:n
      codej = B[i, j] # Choose the code
      @simd for k = 1:d     # And copy it all
        CB[k, j] += mcb[k, codej]
      end
    end
  end

  return CB
end

function reconstruct(
  B::Vector{T1},
  C::Matrix{T2}) where {T1<:Integer, T2<:AbstractFloat}

  CB = C[:,B[:]]
  return CB
end

"Compute cost of encoding in each vector"
function veccost(
  X::Matrix{T1},
  B::Matrix{T2},
  C::Vector{Matrix{T1}}) where {T1 <: AbstractFloat, T2 <: Integer}

  # This implementation is devectorized and minimizes extra memory allocated

  d, n = size( X )
  m,_ = size(B)

  cost = zeros(T1, n)
  CB   = zeros(T1, d)

  @inbounds for i = 1:n # Loop over vectors in X
    for k = 1:m # Loop over codebooks
      Ci = C[k]
      code = B[k, i] # Choose the code
      @simd for j = 1:d # add the codebook entry
        CB[j] += Ci[j, code]
      end
    end

    # Compute the squared error
    @simd for j = 1:d
      cost[i] += (CB[j] - X[j,i]).^2
      CB[j] = 0.0
    end
  end

  cost
end

"Get the total cost of encoding"
function qerror(
  X::Matrix{T1},
  B::Matrix{T2},
  C::Vector{Matrix{T1}}) where {T1 <: AbstractFloat, T2 <: Integer}
  mean(veccost(X,B,C))
end

"Get the total error of an OPQ encoding"
function qerror_opq(
  X::Matrix{T1},
  B::Matrix{T2},
  C::Vector{Matrix{T1}},
  R::Matrix{T1}) where {T1 <: AbstractFloat, T2 <: Integer}

  CB = similar( X )
  subdims = splitarray( 1:size(X,1), length(C) )
  for i=1:length(C)
    CB[subdims[i], :] = C[i][:, vec(B[i,:]) ]
  end

  mean(sum((R*CB - X).^2, dims=1))
end

"Get the total error of a PQ encoding"
function qerror_pq(
  X::Matrix{T1},
  B::Matrix{T2},
  C::Vector{Matrix{T1}}) where {T1 <: AbstractFloat, T2 <: Integer}
  d = size(X, 1)
  R = Matrix{T1}(1.0I, d, d)
  qerror_opq(X, B, C, R)
end
