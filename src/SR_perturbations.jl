# Methods that add perturbations

# Scales the standard deviation according to the passed schedule
function apply_schedule(
  stdev::Vector{Float32},     # Standard deviation
  iter::Integer,              # Iteration number
  niter::Integer,             # Total number of iterations
  p::Float32)

  stdev = stdev * (1 - (iter/niter)).^p

  return stdev
end

# Perturb the codebooks according to the SR-D algorithm. Assumes d-dim codebooks
function SR_D_perturb(
  C::Vector{Matrix{Float32}}, # The codebooks
  iter::Integer,              # Iteration number
  niter::Integer,             # Total number of iterations
  p::Float32=0.5)             # Power parameter in equation (18)

  m = length( C )
  d, h = size( C[1] )

  # Compute the standard deviation
  stdc = std( cat(2, C...), 2 ) ./ m
  stdc = apply_schedule( stdc[:], iter, niter, p )

  for i = 1:m # Loop through each codebook
    for j = 1:d # Loop through each dimension
      noise = randn(h,1)*(stdc[j])
      C[i][j,:] = C[i][j,:] + noise
    end
  end

  return C
end

# Perturb the data according to the SR-C algorithm
function SR_C_perturb(
  X::Matrix{Float32},         # d-by-n matrix of data points to train on.
  iter::Integer,              # Iteration number
  niter::Integer,             # Total number of iterations
  p::Float32=0.5)             # Power parameter in equation (18)

  d, n = size( X )

  # Compute the standard deviation
  stdx = std( X, 2 )
  stdx = apply_schedule( stdx[:], iter, niter, p )

  Y = zeros(Float32,size(X))

  for i = 1:d # Loop through each dimension
      noise = randn(n,1)*(stdx[i])
      Y[i,:] = X[i,:] + noise
  end

  return Y
end
