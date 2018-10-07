using Rayuela
using Printf

# Load utilities for saving experimental results
include("experiment_utils.jl")


# === demos query base ===
function run_demos_query_base(
  dataset_name="labelme",     # Dataset to work on
  ntrain::Integer=Int(20e3),  # Number of training/base examples
  m::Integer=8,               # Number of codebooks (-1 will be substracted for non-orthogonal methods)
  h::Integer=256,             # Number of Entries per codebook
  niter::Integer=25,          # Number of training iterations
  ntrials::Integer=10)        # Number of times to repeat the experiment

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

  # (Semi-)orthogonal methods: PQ, OPQ, ChainQ
  for trial = 1:ntrials
    C, B, train_error, recall = Rayuela.experiment_pq_query_base(Xt, Xq, gt, m, h, niter, knn, verbose)
    save_results_pq_query_base("./results/$(lowercase(dataset_name))/pq_m$(m)_it$(niter).h5", trial, C, B, train_error, recall)
  end
  for trial = 1:ntrials
    init = "natural"
    C, B, R, train_error, recall = Rayuela.experiment_opq_query_base(Xt, Xq, gt, m, h, init, niter, knn, verbose)
    save_results_opq_query_base("./results/$(lowercase(dataset_name))/opq_m$(m)_it$(niter).h5", trial, C, B, R, train_error, recall)
  end

  # Cheap non-orthogonal methods: RVQ, ERVQ
  for trial = 1:ntrials
    C, B, train_error, recall = Rayuela.experiment_rvq_query_base(Xt,        Xq, gt, m-1, h, niter, knn, verbose)
    save_results_pq_query_base("./results/$(lowercase(dataset_name))/rvq_m$(m-1)_it$(niter).h5",  trial, C, B, train_error, recall)
  end
  for trial = 1:ntrials
    C, B, _ = load_rvq("./results/$(lowercase(dataset_name))/rvq_m$(m-1)_it$(niter).h5", m-1, trial)
    C, B, train_error, recall = Rayuela.experiment_ervq_query_base(Xt, B, C, Xq, gt, m-1, h, niter, knn, verbose)
    save_results_pq_query_base("./results/$(lowercase(dataset_name))/ervq_m$(m-1)_it$(niter).h5", trial, C, B, train_error, recall)
  end

  # Precompute init for LSQ/SR
  for trial = 1:ntrials
    C, B, R, train_error = Rayuela.train_opq(Xt, m-1, h, niter, "natural", verbose)
    save_results_opq_query_base("./results/$(lowercase(dataset_name))/opq_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, [0f0])
  end
  for trial = 1:ntrials
    C, B, R, _ = load_chainq("./results/$(lowercase(dataset_name))/opq_m$(m-1)_it$(niter).h5", m-1, trial)
    C, B, R, chainq_error = Rayuela.train_chainq(Xt, m-1, h, R, B, C, niter, verbose)
    save_results_opq_query_base("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", trial, C, B, R, chainq_error, [0f0])
  end

  ilsiter = 8
  icmiter = 4
  randord = true
  npert   = 4
  nsplits_train = 1

  for trial = 1:ntrials
    C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
    C, B, R, train_error, recall = Rayuela.experiment_lsq_cuda_query_base(Xt, B, C, R, Xq, gt, m-1, h,
      niter, ilsiter, icmiter, randord, npert, knn, nsplits_train, verbose)
    save_results_lsq_query_base("./results/$(lowercase(dataset_name))/lsq_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, recall)
  end

  schedule = 1
  p = 0.5

  for trial = 1:ntrials
    sr_method = "SR_D"
    C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
    C, B, R, train_error, recall = Rayuela.experiment_sr_cuda_query_base( Xt, B, C, R, Xq, gt, m-1, h,
      niter, ilsiter, icmiter, randord, npert, knn, nsplits_train, sr_method, schedule, p, verbose)
    save_results_lsq_query_base("./results/$(lowercase(dataset_name))/srd_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, recall)
  end

  for trial = 1:ntrials
    sr_method = "SR_C"
    C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
    C, B, R, train_error, recall = Rayuela.experiment_sr_cuda_query_base( Xt, B, C, R, Xq, gt, m-1, h,
      niter, ilsiter, icmiter, randord, npert, knn, nsplits_train, sr_method, schedule, p, verbose)
    save_results_lsq_query_base("./results/$(lowercase(dataset_name))/src_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, recall)
  end

end

for niter = [25]
  run_demos_query_base("labelme", Int(20e3), 8, 256, niter)
  # run_demos_query_base("labelme", Int(20e3), 16, 256, niter)
  # run_demos_query_base("MNIST",   Int(60e3), 8, 256, niter)
  # run_demos_query_base("MNIST",   Int(60e3), 16, 256, niter)
end
