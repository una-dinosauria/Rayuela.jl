
### Stochastic relaxations for LSQ

function train_sr(
  X::Matrix,                  # d-by-n matrix of data points to train on.
  m::Integer,                 # number of codebooks
  h::Integer,                 # number of entries per codebook
  R::Matrix{Float32},         # init rotation
  B::Matrix{Int16},     # init codes
  C::Vector{Matrix{Float32}}, # init codebooks
  niter::Integer,             # number of optimization iterations
  ilsiter::Integer,           # number of ILS iterations to use during encoding
  icmiter::Integer,           # number of iterations in local search
  randord::Bool,              # whether to use random order
  npert::Integer,             # The number of codes to perturb
  method::AbstractString,     # The SR method to use
  schedule::Integer,          # The schedule to choose
  p::Float32,                 # SR-D power parameter
  alpha::Float32,             # SR-D Scaling parameter in third schedule
  V::Bool=false)              # whether to print progress

  # if V
  println("**********************************************************************************************");
  println("Doing local search with $m codebooks, $npert perturbations, $icmiter icm iterations and random order = $randord");
  println("**********************************************************************************************");
  # end

  if !(method in ["SR_C", "SR_D"]); error("SR method unknown"); end

  d, n = size( X )

  # RX = zeros( Float32, size(X) )
  # B  = zeros( Int16, m, n )
  # C  = Vector{Matrix{Float32}}(m)
  # CB = zeros( Float32, size(X) )

  # [[ Random initialization ]]
  RX = R' * X
  # B = randinit(n, m, h)
  # C = update_codebooks_fast_bin( RX, B, h, V )
  @printf("Random error: %e\n", qerror( RX, B, C ))

  # Add noise to X
  if method == "SR_C"
    RX_noisy = SR_C_perturb( RX, 0, niter, schedule, p, alpha )
    C = update_codebooks_fast_bin( RX_noisy, B, h, V )
  else
    C = update_codebooks_fast_bin( RX, B, h, V )

    obj = qerror( RX, B, C )
    @printf("%3d %e \n", -1, obj)

    if method == "SR_D"
      #C = SR_D_perturb( C, 0, niter, schedule, p, alpha )
      C = SR_D_perturb( C, 1, niter, schedule, p, alpha )
    end
  end

  obj = qerror( RX, B, C );
  @printf("%3d %e \n", -1, obj);

  # Initialize B
  @time B = encoding_icm( RX, B, C, ilsiter, icmiter, randord, npert, true, V )

  obj = qerror( RX, B, C )

  @printf("%3d %e \n", -1, obj)

  obj     = Inf;
  objlast = Inf;
  objarray = zeros( Float32, niter+1 )

  for iter = 1:niter

    objlast = obj
    obj = qerror( RX, B, C  )
    objarray[iter] = obj
    @printf("%3d %e (%e better) \n", iter, obj, objlast - obj)

    # Add noise to X
    if method == "SR_C"
      RX_noisy = SR_C_perturb( RX, iter, niter, schedule, p, alpha )
      C = update_codebooks_fast_bin( RX_noisy, B, h, V )
    else
      C = update_codebooks_fast_bin( RX, B, h, V )
      if method == "SR_D"
        C = SR_D_perturb( C, iter, niter, schedule, p, alpha )
      end
    end

    # Update the codes with local search
    # B = randinit(n, m, h)
    @time B = encoding_icm( RX, B, C, ilsiter, icmiter, randord, npert, true, V )

  end

  objarray[niter+1] = qerror( RX, B, C  )

  return C, B, objarray

end
