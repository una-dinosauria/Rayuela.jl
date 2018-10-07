using Rayuela
using Printf

# Load utilities for saving experimental results
include("experiment_utils.jl")


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
    C, B, train_error, B_base, recall = Rayuela.experiment_pq(Xt, Xb, Xq, gt, m, h, niter, knn, verbose)
    save_results_pq("./results/$(lowercase(dataset_name))/pq_m$(m)_it$(niter).h5", trial, C, B, train_error, B_base, recall)
  end
  for trial = 1:ntrials
    init = "natural"
    C, B, R, train_error, B_base, recall = Rayuela.experiment_opq(Xt, Xb, Xq, gt, m, h, init, niter, knn, verbose)
    save_results_opq("./results/$(lowercase(dataset_name))/opq_m$(m)_it$(niter).h5", trial, C, B, R, train_error, B_base, recall)
  end

  # Cheap non-orthogonal methods: RVQ, ERVQ
  for trial = 1:ntrials
    C, B, train_error, B_base, recall = Rayuela.experiment_rvq( Xt,       Xb, Xq, gt, m-1, h, niter, knn, verbose)
    save_results_pq("./results/$(lowercase(dataset_name))/rvq_m$(m-1)_it$(niter).h5",  trial, C, B, train_error, B_base, recall)
  end
  for trial = 1:ntrials
    C, B, train_error = load_rvq("./results/$(lowercase(dataset_name))/rvq_m$(m-1)_it$(niter).h5", m-1, trial)
    C, B, train_error, B_base, recall = Rayuela.experiment_ervq(Xt, B, C, Xb, Xq, gt, m-1, h, niter, knn, verbose)
    save_results_pq("./results/$(lowercase(dataset_name))/ervq_m$(m-1)_it$(niter).h5", trial, C, B, train_error, B_base, recall)
  end

  # Precompute init for LSQ/SR
  for trial = 1:ntrials
    C, B, R, train_error = Rayuela.train_opq(Xt, m-1, h, niter, "natural", verbose)
    save_results_opq("./results/$(lowercase(dataset_name))/opq_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, ones(UInt16,1,1), [0f0])
  end
  for trial = 1:ntrials
    C, B, R, _ = load_chainq("./results/$(lowercase(dataset_name))/opq_m$(m-1)_it$(niter).h5", m-1, trial)
    C, B, R, chainq_error = train_chainq(    Xt, m-1, h, R, B, C, niter, verbose)
    save_results_opq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", trial, C, B, R, chainq_error, ones(UInt16,1,1), [0f0])
  end

  nsplits_train =  m <= 8 ? 1 : 1
  nsplits_base  =  m <= 8 ? 2 : 4

  ilsiter = 8
  icmiter = 4
  randord = true
  npert   = 4

  # === GPU LSQ ===
  for trial = 1:ntrials
    C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
    C, B, R, train_error, B_base, recall = Rayuela.experiment_lsq_cuda(Xt, B, C, R, Xb, Xq, gt, m-1, h,
      niter, ilsiter, icmiter, randord, npert, knn, nsplits_train, nsplits_base, verbose)
    save_results_lsq("./results/$(lowercase(dataset_name))/lsq_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, B_base, recall)
  end

  schedule = 1
  p = 0.5

  # === GPU LSQ++, SR-D and SR-C ===
  for trial = 1:ntrials
    sr_method = "SR_D"
    C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
    C, B, R, train_error, B_base, recall = Rayuela.experiment_sr_cuda(Xt, B, C, R, Xb, Xq, gt, m-1, h,
      niter, ilsiter, icmiter, randord, npert, knn, nsplits_train, nsplits_base, sr_method, schedule, p, verbose)
    save_results_lsq("./results/$(lowercase(dataset_name))/srd_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, B_base, recall)
  end

  for trial = 1:ntrials
    sr_method = "SR_C"
    C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
    C, B, R, train_error, B_base, recall = Rayuela.experiment_sr_cuda(Xt, B, C, R, Xb, Xq, gt, m-1, h,
      niter, ilsiter, icmiter, randord, npert, knn, nsplits_train, nsplits_base, sr_method, schedule, p, verbose)
    save_results_lsq("./results/$(lowercase(dataset_name))/src_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, B_base, recall)
  end
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
# for method in ["srd"], niter = [25], m = [15]
#   high_recall_experiments("SIFT1M", method, Int(1e5), m, 256, niter)
#   high_recall_experiments("Deep1M", method, Int(1e5), m, 256, niter)
#   high_recall_experiments("Convnet1M", method, Int(1e5), m, 256, niter)
# end


# run_demos("SIFT1M", Int(1e5),  8, 256, 25)
for niter = [25]
  run_demos("SIFT1M", Int(1e5),  8, 256, niter)
  # run_demos("SIFT1M", Int(1e5), 16, 256, niter)
  # run_demos("Convnet1M", Int(1e5),   8, 256, niter)
  # run_demos("Convnet1M", Int(1e5),  16, 256, niter)
  # run_demos("Deep1M", Int(1e5),  8, 256, niter)
  # run_demos("Deep1M", Int(1e5), 16, 256, niter)
end
