
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
  method::AbstractString,     # The SR method to use. Either SR_C or SR_D
  p::Float32,                 # SR-D power parameter
  cpp::Bool=true,             # whether to use ICM's cpp implementation
  V::Bool=false)              # whether to print progress

  # if V
  println("**********************************************************************************************");
  println("Doing local search with $m codebooks, $npert perturbations, $icmiter icm iterations and random order = $randord");
  println("**********************************************************************************************");
  # end

  if !(method in ["SR_C", "SR_D"]); error("SR method unknown"); end
  d, n = size( X )

  RX = R' * X
  @printf("Random error: %e\n", qerror( RX, B, C ))

  if method == "SR_C"
    # In SR-C we add noise to X
    RX_noisy = SR_C_perturb( RX, 0, niter, p )
    C = update_codebooks_fast_bin( RX_noisy, B, h, V )
  else
    # In SR-D we add noise to C
    C = update_codebooks_fast_bin( RX, B, h, V )
    obj = qerror( RX, B, C )
    @printf("%3d %e \n", -1, obj)
    C = SR_D_perturb( C, 1, niter, p )
  end

  obj = qerror( RX, B, C );
  @printf("%3d %e \n", -1, obj);

  # Initialize B
  @time B = encoding_icm( RX, B, C, ilsiter, icmiter, randord, npert, cpp, V )
  obj = qerror( RX, B, C )
  @printf("%3d %e \n", -1, obj)

  obj     = Inf
  objlast = Inf
  objarray = zeros( Float32, niter+1 )

  for iter = 1:niter

    objlast = obj
    obj = qerror( RX, B, C  )
    objarray[iter] = obj
    @printf("%3d %e (%e better) \n", iter, obj, objlast - obj)

    if method == "SR_C"
      # In SR-C we add noise to X
      RX_noisy = SR_C_perturb( RX, iter, niter, p )
      C = update_codebooks_fast_bin( RX_noisy, B, h, V )
    else
      C = update_codebooks_fast_bin( RX, B, h, V )
      C = SR_D_perturb( C, iter, niter, p )
    end

    # Update the codes with local search
    @time B = encoding_icm( RX, B, C, ilsiter, icmiter, randord, npert, cpp, V )

  end

  objarray[niter+1] = qerror( RX, B, C  )

  return C, B, objarray

end
