using Rayuela
using HDF5
using Printf

# === Saving functions ===
function save_results_pq_query_base(
  bpath::String, trial::Integer, C::Vector{Matrix{Float32}}, B, train_error, recall)
  for i = 1:length(C)
    h5write(bpath, "$(trial)/C_$i", C[i])
  end
  h5write(bpath, "$(trial)/B", convert(Matrix{UInt8}, B.-1))
  h5write(bpath, "$(trial)/train_error", train_error)
  h5write(bpath, "$(trial)/recall", recall)
end

function save_results_pq(
  bpath::String, trial::Integer, C::Vector{Matrix{Float32}}, B, train_error, B_base, recall)
  h5write(bpath, "$(trial)/B_base", convert(Matrix{UInt8}, B_base.-1))
  save_results_pq_query_base(bpath, trial, C, B, train_error, recall)
end

function save_results_opq_query_base(
  bpath::String, trial::Integer, C::Vector{Matrix{Float32}}, B, R::Matrix{Float32}, train_error, recall)
  h5write(bpath, "$(trial)/R", R)
  save_results_pq_query_base(bpath, trial, C, B, train_error, recall)
end

function save_results_opq(
  bpath::String, trial::Integer, C::Vector{Matrix{Float32}}, B, R::Matrix{Float32}, train_error, B_base, recall)
  h5write(bpath, "$(trial)/B_base", convert(Matrix{UInt8}, B_base.-1))
  save_results_opq_query_base(bpath, trial, C, B, R, train_error, recall)
end

function save_results_lsq_query_base(
  bpath::String, trial::Integer, C::Vector{Matrix{Float32}}, B, R::Matrix{Float32}, train_error, opq_error, recall)
  h5write(bpath, "$(trial)/opq_base", opq_error)
  save_results_opq_query_base(bpath, trial, C, B, R, train_error, recall)
end

function save_results_lsq(
  bpath::String, trial::Integer, C::Vector{Matrix{Float32}}, B, R::Matrix{Float32}, train_error, opq_error, B_base, recall)
  h5write(bpath, "$(trial)/B_base", convert(Matrix{UInt8}, B_base.-1))
  save_results_lsq_query_base(bpath, trial, C, B, R, train_error, opq_error, recall)
end

function load_chainq(fname::String, m::Integer, trial::Integer)
  B = h5read(fname, "$trial/B"); B = convert(Matrix{Int16}, B); B.+=1
  R = h5read(fname, "$trial/R")
  chainq_error = h5read(fname, "$trial/train_error")
  C = Vector{Matrix{Float32}}(undef, m)
  for i=1:(m); C[i] = h5read(fname, "$trial/C_$i"); end
  return C, B, R, chainq_error
end

function load_rvq(fname::String, m::Integer, trial::Integer)
  B = h5read(fname, "$trial/B"); B = convert(Matrix{Int16}, B); B.+=1
  rvq_error = h5read(fname, "$trial/train_error")
  C = Vector{Matrix{Float32}}(m)
  for i=1:(m); C[i] = h5read(fname, "$trial/C_$i"); end
  return C, B, rvq_error
end

# === Data loading function ===
function load_experiment_data(
  dataset_name::String,
  ntrain::Integer, nbase::Integer, nquery::Integer, V::Bool=false)

  Xt = Rayuela.read_dataset(dataset_name, ntrain, V)
  Xb = Rayuela.read_dataset(dataset_name * "_base", nbase, V)
  Xq = Rayuela.read_dataset(dataset_name * "_query", nquery, V)[:,1:nquery]
  gt = Rayuela.read_dataset(dataset_name * "_groundtruth", nquery, V)
  if dataset_name == "SIFT1M" || dataset_name == "GIST1M"
    gt = gt .+ 1
  end

  if dataset_name != "Deep1M"
    gt = convert( Vector{UInt32}, gt[1,1:nquery] )
  else
    gt = convert( Vector{UInt32}, gt[1:nquery] )
  end

  return Xt, Xb, Xq, gt
end


# === experiment functions ===
function run_demos(
  dataset_name="SIFT1M",
  ntrain::Integer=Int(1e5),
  m::Integer=8, h::Integer=256, niter::Integer=25)

  nquery, nbase, knn = 0, 0, 0
  if dataset_name == "SIFT1M" || dataset_name == "Deep1M" || dataset_name == "Convnet1M"
    nquery, nbase, knn = Int(1e4), Int(1e6), Int(1e3)
  else
    error("dataset unknown")
  end

  verbose = true

  # Load data
  Xt, Xb, Xq, gt = load_experiment_data(dataset_name, ntrain, nbase, nquery, verbose)
  d, _    = size( Xt )

  ntrials = 10
  # (Semi-)orthogonal methods: PQ, OPQ, ChainQ
  for trial = 1:ntrials
    # C, B, train_error, B_base, recall = Rayuela.experiment_pq(Xt, Xb, Xq, gt, m, h, niter, knn, verbose)
    # save_results_pq("./results/$(lowercase(dataset_name))/pq_m$(m)_it$(niter).h5", trial, C, B, train_error, B_base, recall)
  end
  for trial = 1:ntrials
    init = "natural"
    # C, B, R, train_error, B_base, recall = Rayuela.experiment_opq(Xt, Xb, Xq, gt, m, h, init, niter, knn, verbose)
    # save_results_opq("./results/$(lowercase(dataset_name))/opq_m$(m)_it$(niter).h5", trial, C, B, R, train_error, B_base, recall)
  end

  # Cheap non-orthogonal methods: RVQ, ERVQ
  for trial = 1:ntrials
    # C, B, train_error, B_base, recall = Rayuela.experiment_rvq( Xt,       Xb, Xq, gt, m-1, h, niter, knn, verbose)
    # save_results_pq("./results/$(lowercase(dataset_name))/rvq_m$(m-1)_it$(niter).h5",  trial, C, B, train_error, B_base, recall)
  end
  for trial = 1:ntrials
    # C, B, train_error = load_rvq("./results/$(lowercase(dataset_name))/rvq_m$(m-1)_it$(niter).h5", m-1, trial)
    # C, B, train_error, B_base, recall = Rayuela.experiment_ervq(Xt, B, C, Xb, Xq, gt, m-1, h, niter, knn, verbose)
    # save_results_pq("./results/$(lowercase(dataset_name))/ervq_m$(m-1)_it$(niter).h5", trial, C, B, train_error, B_base, recall)
  end

  # Precompute init for LSQ/SR
  for trial = 1:ntrials
    # C, B, R, train_error = Rayuela.train_opq(Xt, m-1, h, niter, "natural", verbose)
    # save_results_opq("./results/$(lowercase(dataset_name))/opq_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, ones(UInt16,1,1), [0f0])
  end
  for trial = 1:ntrials
    # C, B, R, _ = load_chainq("./results/$(lowercase(dataset_name))/opq_m$(m-1)_it$(niter).h5", m-1, trial)
    # C, B, R, chainq_error = train_chainq(    Xt, m-1, h, R, B, C, niter, verbose)
    # save_results_opq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", trial, C, B, R, chainq_error, ones(UInt16,1,1), [0f0])
  end

  nsplits_train =  m == 8 ? 1 : 1
  nsplits_base  =  m == 8 ? 2 : 4
  @show nsplits_train, nsplits_base

  # for trial = 1:ntrials
  #
  #   # C, B, R, _ = load_chainq("./results/$(lowercase(dataset_name))/lsq_m$(m-1)_it$(25).h5", m-1, trial)
  #   # B_base = h5read("./results/$(lowercase(dataset_name))/lsq_m$(m-1)_it$(25).h5", "$trial/B_base")
  #   # B_base = convert(Matrix{Int16}, B_base) .+ 1
  #   # @show qerror(Xt, B, C), qerror(Xb, B_base, C)
  #
  #   C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
  #   C, B, R, train_error, B_base, recall = Rayuela.experiment_lsq(Xt, B, C, R, Xb, Xq, gt, m-1, h, niter, knn, verbose)
  #   # C, B, R, train_error, B_base, recall = Rayuela.experiment_lsq_cuda(Xt, B, C, R, Xb, Xq, gt, m-1, h, niter, knn, nsplits_train, nsplits_base, verbose)
  #   save_results_lsq("./results/$(lowercase(dataset_name))/lsq_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, B_base, recall)
  # end
  #
  # for trial = 1:ntrials
  #   sr_method = "SR_D"
  #   C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
  #   C, B, R, train_error, B_base, recall = Rayuela.experiment_sr_cuda(Xt, B, C, R, Xb, Xq, gt, m-1, h, niter, knn, nsplits_train, nsplits_base, sr_method, verbose)
  #   save_results_lsq("./results/$(lowercase(dataset_name))/srd_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, B_base, recall)
  # end
  #
  # for trial = 1:ntrials
  #   sr_method = "SR_C"
  #   C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
  #   C, B, R, train_error, B_base, recall = Rayuela.experiment_sr_cuda(Xt, B, C, R, Xb, Xq, gt, m-1, h, niter, knn, nsplits_train, nsplits_base, sr_method, verbose)
  #   save_results_lsq("./results/$(lowercase(dataset_name))/src_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, B_base, recall)
  # end

  # GPU methods
  # nsplits_train = 1
  # nsplits_base  = 2
  # Rayuela.experiment_lsq_cuda(Xt, Xb, Xq, gt, m, h, niter, knn, nsplits_train, nsplits_base, verbose)
  # Rayuela.experiment_sr_cuda(Xt, Xb, Xq, gt, m, h, niter, knn, nsplits_train, nsplits_base, verbose)


  # GPU methods with random inputs
  # B = convert(Matrix{Int16}, rand(1:h, m, size(Xt,2)))
  # C = Vector{Matrix{Float32}}(m); for i=1:m; C[i]=zeros(Float32,d,h); end
  # R = eye(Float32, d)
  # Rayuela.experiment_lsq_cuda(Xt, B, C, R, Xb, Xq, gt, m, h, niter, knn, verbose)
  # Rayuela.experiment_sr_cuda( Xt, B, C, R, Xb, Xq, gt, m, h, niter, knn, verbose)

  # return C, B, train_error, B_base, recall
  # return C, B, R, train_error, B_base, recall
end


# === demos query base ===
function run_demos_query_base(
  dataset_name="labelme",
  ntrain::Integer=Int(2e3),
  m::Integer=8, h::Integer=256, niter::Integer=25, ntrials::Integer=10)

  nquery, nbase, knn = 0, 0, 0
  if dataset_name == "MNIST"
    nquery, nbase, knn = Int(10e3), Int(60e3), Int(1e3)
  elseif dataset_name == "labelme"
    nquery, nbase, knn = Int(2e3), Int(20019), Int(1e3)
  else
    error("dataset unknown")
  end

  verbose = true

  Xt, Xb, Xq, gt = load_experiment_data(dataset_name, ntrain, nbase, nquery, verbose)
  d, _ = size( Xt )

  # # (Semi-)orthogonal methods: PQ, OPQ, ChainQ
  # for trial = 1:ntrials
  #   C, B, train_error, recall = Rayuela.experiment_pq_query_base(Xt, Xq, gt, m, h, niter, knn, verbose)
  #   save_results_pq_query_base("./results/$(lowercase(dataset_name))/pq_m$(m)_it$(niter).h5", trial, C, B, train_error, recall)
  # end
  # for trial = 1:ntrials
  #   C, B, R, train_error, recall = Rayuela.experiment_opq_query_base(Xt, Xq, gt, m, h, niter, knn, verbose)
  #   save_results_opq_query_base("./results/$(lowercase(dataset_name))/opq_m$(m)_it$(niter).h5", trial, C, B, R, train_error, recall)
  # end

  # # Cheap non-orthogonal methods: RVQ, ERVQ
  # for trial = 1:ntrials
  #   C, B, train_error, recall = Rayuela.experiment_rvq_query_base(Xt,        Xq, gt, m-1, h, niter, knn, verbose)
  #   save_results_pq_query_base("./results/$(lowercase(dataset_name))/rvq_m$(m-1)_it$(niter).h5",  trial, C, B, train_error, recall)
  # end
  # for trial = 1:ntrials
  #   C, B, _ = load_rvq("./results/$(lowercase(dataset_name))/rvq_m$(m-1)_it$(niter).h5", m-1, trial)
  #   C, B, train_error, recall = Rayuela.experiment_ervq_query_base(Xt, B, C, Xq, gt, m-1, h, niter, knn, verbose)
  #   save_results_pq_query_base("./results/$(lowercase(dataset_name))/ervq_m$(m-1)_it$(niter).h5", trial, C, B, train_error, recall)
  # end

    # More expensive non-orthogonal methods
    # Rayuela.experiment_chainq(Xt, Xb, Xq, gt, m, h, niter, knn, verbose)
    # Rayuela.experiment_lsq(Xt, Xb, Xq, gt, m, h, niter, knn, verbose)
    # Rayuela.experiment_sr(Xt, Xb, Xq, gt, m, h, niter, knn, verbose)


  # # Precompute init for LSQ/SR
  # for trial = 1:ntrials
  #   C, B, R, train_error = Rayuela.train_opq(Xt, m-1, h, niter, "natural", verbose)
  #   save_results_opq_query_base("./results/$(lowercase(dataset_name))/opq_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, [0f0])
  # end
  for trial = 1:ntrials
    C, B, R, _ = load_chainq("./results/$(lowercase(dataset_name))/opq_m$(m-1)_it$(niter).h5", m-1, trial)
    C, B, R, chainq_error = Rayuela.train_chainq(Xt, m-1, h, R, B, C, niter, verbose)
    save_results_opq_query_base("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", trial, C, B, R, chainq_error, [0f0])
  end

  # nsplits_train = 1
  # for trial = 1:ntrials
  #   C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
  #   # C, B, R, train_error, recall = Rayuela.experiment_lsq_cuda_query_base(Xt, B, C, R, Xq, gt, m-1, h, niter, knn, nsplits_train, verbose)
  #   C, B, R, train_error, recall = Rayuela.experiment_lsq_query_base(Xt, B, C, R, Xq, gt, m-1, h, niter, knn, verbose)
  #   save_results_lsq_query_base("./results/$(lowercase(dataset_name))/lsq_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, recall)
  # end

  # for trial = 1:ntrials
  #   sr_method = "SR_D"
  #   C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
  #   C, B, R, train_error, recall = Rayuela.experiment_sr_cuda_query_base( Xt, B, C, R, Xq, gt, m-1, h, niter, knn, nsplits_train, sr_method, verbose)
  #   save_results_lsq_query_base("./results/$(lowercase(dataset_name))/srd_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, recall)
  # end
  #
  # for trial = 1:ntrials
  #   sr_method = "SR_C"
  #   C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
  #   C, B, R, train_error, recall = Rayuela.experiment_sr_cuda_query_base( Xt, B, C, R, Xq, gt, m-1, h, niter, knn, nsplits_train, sr_method, verbose)
  #   save_results_lsq_query_base("./results/$(lowercase(dataset_name))/src_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, recall)
  # end


  # GPU methods
  # nsplits_train = 1
  # (C, B, R, train_error, recall), opq_error = Rayuela.experiment_lsq_cuda_query_base(Xt, Xq, gt, m-1, h, niter, knn, nsplits_train, verbose)
  # save_results_lsq_query_base("./results/$(lowercase(dataset_name))/lsq_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, opq_error, recall)

  # sr_method = "SR_D"
  # (C, B, R, train_error, recall), opq_error = Rayuela.experiment_sr_cuda_query_base( Xt, Xq, gt, m-1, h, niter, knn, nsplits_train, sr_method, verbose)
  # save_results_lsq_query_base("./results/$(lowercase(dataset_name))/srd_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, opq_error, recall)
  #
  # sr_method = "SR_C"
  # (C, B, R, train_error, recall), opq_error = Rayuela.experiment_sr_cuda_query_base( Xt, Xq, gt, m-1, h, niter, knn, nsplits_train, sr_method, verbose)
  # save_results_lsq_query_base("./results/$(lowercase(dataset_name))/src_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, opq_error, recall)


  # GPU methods with random inputs
  # B = convert(Matrix{Int16}, rand(1:h, m, size(Xt,2)))
  # C = Vector{Matrix{Float32}}(m); for i=1:m; C[i]=zeros(Float32,d,h); end
  # R = eye(Float32, d)
  # Rayuela.experiment_lsq_cuda(Xt, B, C, R, Xb, Xq, gt, m, h, niter, knn, verbose)
  # Rayuela.experiment_sr_cuda( Xt, B, C, R, Xb, Xq, gt, m, h, niter, knn, verbose)

end

function high_recall_experiments(
  dataset_name="SIFT1M",
  method::String="lsq",
  ntrain::Integer=Int(1e5),
  m::Integer=8,
  h::Integer=256,
  niter::Integer=25)

  if !(method in ["lsq", "src", "srd"])
    error("Method $method unkonwn")
  end

  nquery, nbase, knn = Int(1e4), Int(1e6), Int(1e3)
  verbose = V = true
  Xt, Xb, Xq, gt = load_experiment_data(dataset_name, ntrain, nbase, nquery, verbose)
  d, _    = size( Xt )

  ilsiters = [1, 2, 4, 8, 16, 32, 64, 128, 256]
  icmiter = 4
  randord = true
  npert   = 4
  p       = 0.5f0
  nsplits_base = m <= 8 ? 2 : 4

  ntrials = 10
  for trial = 1:ntrials

    @show trial

    fname = "./results/$(lowercase(dataset_name))/$(method)_m$(m)_it$(niter).h5"
    C, B, R, chainq_error = load_chainq(fname, m, trial)
    norms_B, norms_C = get_norms_codebook(B, C)

    # === Encode the base set ===
    B_base = convert(Matrix{Int16}, rand(1:h, m, size(Xb,2)))

    # recalls = zeros(Float32, length(ilsiters), knn)
    Bs_base, _ = encode_icm_cuda(Xb, B_base, C, ilsiters, icmiter, npert, randord, nsplits_base, V)

    for (idx, ilsiter) in enumerate(ilsiters)
      B_base = Bs_base[idx]
      base_error = qerror(Xb, B_base, C)
      if V; @printf("Error in base is %e\n", base_error); end

      # Compute and quantize the database norms
      B_base_norms, db_norms_X = quantize_norms( B_base, C, norms_C )
      db_norms = vec( norms_C[ B_base_norms ] )

      if V; print("Querying m=$m, ilsiter=$ilsiter ... "); end
      @time dists, idx = linscan_lsq(B_base, Xq, C, db_norms, eye(Float32, d), knn)
      if V; println("done"); end

      recall = eval_recall(gt, idx, knn)

      println("Saving to $fname...")
      h5write(fname, "$trial/recall_$(ilsiter)", recall)
      # h5write(bpath, "$trial/B_Base_$(ilsiter)", convert(Matrix{UInt8}, B_base.-1))
    end
  end

end

# high_recall_experiments("Convnet1M", "lsq", Int(1e5), 15, 256, 25)
# for method in ["lsq", "src", "srd"], niter = [50, 100], m = [15]
for method in ["srd"], niter = [25], m = [15]
  # high_recall_experiments("SIFT1M", method, Int(1e5), m, 256, niter)
  # high_recall_experiments("Deep1M", method, Int(1e5), m, 256, niter)
  # high_recall_experiments("Convnet1M", method, Int(1e5), m, 256, niter)
end


# run_demos("SIFT1M", Int(1e5),  8, 256, 25)
for niter = [25]
  run_demos("SIFT1M", Int(1e4),  8, 256, niter)
  # run_demos("SIFT1M", Int(1e5), 16, 256, niter)
  # run_demos("Convnet1M", Int(1e5),   8, 256, niter)
  # run_demos("Convnet1M", Int(1e5),  16, 256, niter)
  # run_demos("Deep1M", Int(1e5),  8, 256, niter)
  # run_demos("Deep1M", Int(1e5), 16, 256, niter)
end

# run_demos_query_base("MNIST",   Int(60e3), 8,  256, 5, 1)
# run_demos_query_base("MNIST",   Int(60e3), 16, 256, 5, 1)
# run_demos_query_base("labelme", Int(20e3), 8,  256, 5, 1)
# run_demos_query_base("labelme", Int(20e3), 16, 256, 5, 1)

# run_demos
# for niter = [100]
#   # run_demos_query_base("labelme", Int(20e3), 8, 256, niter)
#   run_demos_query_base("labelme", Int(20e3), 16, 256, niter)
#   # run_demos_query_base("MNIST",   Int(60e3), 8, 256, niter)
#   run_demos_query_base("MNIST",   Int(60e3), 16, 256, niter)
# end
