
# Methods that add perturbations

# TODO leave just one schedule
# Scales the standard deviation according to the passed schedule
function apply_schedule(
  stdev::Vector{Float32},
  schedule::Integer,
  iter::Integer,              # Iteration number
  niter::Integer,             # Total number of iterations
  p::Float32,
  alpha::Float32)

  if schedule == 1 # Schedule 1
    stdev = stdev * (1 - (iter/niter)).^p # -- deep1b hack on
  elseif schedule == 2 # Schedule 2
    stdev = stdev / ((1 + iter).^p)
  elseif schedule == 3 # Schedule 3
    stdev = stdev * alpha^(iter/2)
  else
    error("Schedule unknown: ", schedule )
  end

  return stdev
end

# Perturb the codebooks according to the SR-D algorithm. Assumes d-dim codebooks
function SR_D_perturb(
  C::Vector{Matrix{Float32}}, # The codebooks
  iter::Integer,              # Iteration number
  niter::Integer,             # Total number of iterations
  schedule::Integer,   # Choose the schedule expression
  p::Float32=0.5,              # Power parameter in equation (18)
  alpha::Float32=0.5 )        # Scaling factor in the third schedule

  m = length( C )
  d, h = size( C[1] )

  # Compute the standard deviation
  stdc = std( cat(2, C...), 2 ) ./ m # For sift1b
  stdc = apply_schedule( stdc[:], schedule, iter, niter, p, alpha )

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
  schedule::Integer,          # Choose the schedule expression
  p::Float32=0.5,              # Power parameter in equation (18)
  alpha::Float32=0.5 )        # Scaling factor in the third schedule

  d, n = size( X )

  # Compute the standard deviation
  stdx = std( X, 2 )
  stdx = apply_schedule( stdx[:], schedule, iter, niter, p, alpha )

  Y = zeros(Float32,size(X))

  for i = 1:d # Loop through each dimension
      noise = randn(n,1)*(stdx[i])
      Y[i,:] = X[i,:] + noise
  end

  return Y
end
