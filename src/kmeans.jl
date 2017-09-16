
#
#  Update centers based on updated assignments
#
#  (specific to the case where samples are not weighted)
#
function update_centers!{T<:AbstractFloat}(
  x::Matrix{T},                   # in: sample matrix (d x n)
  assignments::Vector{Int},       # in: assignments (n)
  cweights::Vector{T},            # out: updated cluster weights (k)
  k::Integer)

  d::Int = size(x, 1)
  n::Int = size(x, 2)

  centers = zeros(T, d, k)

  # initialize center weights
  for i = 1:k
    cweights[i] = 0
  end

  for i = 1:n
    ci = assignments[i]
    cweights[ci] += 1
  end

  # @show sum( cweights .== 0 )

  # accumulate columns
  @inbounds for j = 1:n
    cj = assignments[j]
    1 <= cj <= k || error("assignment out of boundary.")

    for i = 1:d
      centers[i, cj] += x[i, j]
    end
  end

  # sum ==> mean
  for j = 1:k
    if cweights[j] != 0
      @inbounds cj::T = 1 / cweights[j]
      vj = view(centers,:,j)
      for i = 1:d
        @inbounds vj[i] *= cj
      end
    end
  end

  return centers

end
