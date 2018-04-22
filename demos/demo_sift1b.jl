include("demos.jl")

dataset_name="SIFT1B"


h=256
niter=100

ntrain, nquery, nbase, knn = Int(1e4), Int(1e4), Int(1e6), Int(1e3)

V = verbose = true

Xt = read_dataset(dataset_name, ntrain, V)
# Xb = read_dataset(dataset_name * "_base", nbase, V)
Xq = read_dataset(dataset_name * "_query", nquery, V)[:,1:nquery]
gt = read_dataset(dataset_name * "_groundtruth_10M", nquery, V)
gt .+= 1
gt = convert( Vector{UInt32}, gt[1,1:nquery] )

println("Done loading dataset")

idx = []
train_error = []

ntrials = 1
nsplits = nbase == Int(1e9) ? 1000 : 2

for m = [8, 16]
  # (Semi-)orthogonal methods: PQ, OPQ, ChainQ
  # for trial = 1:ntrials
  #   C, B, train_error = Rayuela.train_pq(Xt, m, h, niter, verbose)
  #   B_base = zeros(UInt8, m, nbase)
  #   base_error = 0.
  #   for (i, xt_range) in enumerate(Rayuela.splitarray(1:nbase, nsplits))
  #     println("$i / $nsplits: ", xt_range)
  #     Xb = read_dataset(dataset_name * "_base", xt_range, V)
  #     B_base_xt = Rayuela.quantize_pq(Xb, C, V)
  #     base_error += qerror_pq(Xb, B_base_xt, C)
  #     B_base[:, xt_range] = convert(Matrix{UInt8}, B_base_xt .- 1)
  #   end
  #   if V; @printf("Error in base is %e\n", base_error / nsplits); end
  #
  #   if V; println("Querying m=$m ... "); end
  #   @time dists, idx = linscan_pq(B_base, Xq, C, Int(log2(h) * m), knn)
  #   if V; println("done"); end
  #   recall = eval_recall( gt, idx, knn )
  #
  #   B_base = convert(Matrix{Int16}, B_base)
  #   save_results_pq("/hdd/results/sift10m/pq_m$(m)_it$(niter).h5", trial, C, B, train_error, B_base, recall)
  # end

  for trial = 1:ntrials
    C, B, R, train_error = Rayuela.train_opq(Xt, m, h, niter, "natural", verbose)
    B_base = zeros(UInt8, m, nbase)
    base_error = 0.
    for (i, xt_range) in enumerate(Rayuela.splitarray(1:nbase, nsplits))
      println("$i / $nsplits: ", xt_range)
      Xb = read_dataset(dataset_name * "_base", xt_range, V)
      B_base_xt = Rayuela.quantize_opq(Xb, R, C, V)
      base_error += qerror_opq(Xb, B_base_xt, C, R)
      B_base[:, xt_range] = convert(Matrix{UInt8}, B_base_xt .- 1)
    end
    if V; @printf("Error in base is %e\n", base_error / nsplits); end

    if V; println("Querying m=$m ... "); end
    @time dists, idx = linscan_pq(B_base, Xq, C, Int(log2(h) * m), knn)
    if V; println("done"); end
    recall = eval_recall( gt, idx, knn )

    B_base = convert(Matrix{Int16}, B_base)
    # save_results_opq("/hdd/results/sift10m/opq_m$(m)_it$(niter).h5", trial, C, B, R, train_error, B_base, recall)
  end

  # # Cheap non-orthogonal methods: RVQ, ERVQ
  # for trial = 1:ntrials
  #   C, B, train_error, B_base, recall = Rayuela.experiment_rvq( Xt,       Xb, Xq, gt, m-1, h, niter, knn, verbose)
  #   save_results_pq("/hdd/results/$(lowercase(dataset_name))/rvq_m$(m-1)_it$(niter).h5",  trial, C, B, train_error, B_base, recall)
  # end
  # for trial = 1:ntrials
  #   C, B, train_error = load_rvq("./results/$(lowercase(dataset_name))/rvq_m$(m-1)_it$(niter).h5", m-1, trial)
  #   C, B, train_error, B_base, recall = Rayuela.experiment_ervq(Xt, B, C, Xb, Xq, gt, m-1, h, niter, knn, verbose)
  #   save_results_pq("/hdd/results/$(lowercase(dataset_name))/ervq_m$(m-1)_it$(niter).h5", trial, C, B, train_error, B_base, recall)
  # end
  #
  # # # Precompute init for LSQ/SR
  # for trial = 1:ntrials
  #   C, B, R, train_error = Rayuela.train_opq(Xt, m-1, h, niter, "natural", verbose)
  #   save_results_opq("/hdd/results/$(lowercase(dataset_name))/opq_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, ones(UInt16,1,1), [0f0])
  # end
  # for trial = 1:ntrials
  #   C, B, R, _ = load_chainq("./results/$(lowercase(dataset_name))/opq_m$(m-1)_it$(niter).h5", m-1, trial)
  #   C, B, R, chainq_error = train_chainq(    Xt, m-1, h, R, B, C, niter, verbose)
  #   save_results_opq("/hdd/results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", trial, C, B, R, chainq_error, ones(UInt16,1,1), [0f0])
  # end
  #
  #
  # nsplits_train =  m == 8 ? 3 : 6
  # nsplits_base  =  m == 8 ? 3 : 6
  # @show nsplits_train, nsplits_base
  #
  # for trial = 1:ntrials
  #
  #   # C, B, R, _ = load_chainq("./results/$(lowercase(dataset_name))/lsq_m$(m-1)_it$(25).h5", m-1, trial)
  #   # B_base = h5read("./results/$(lowercase(dataset_name))/lsq_m$(m-1)_it$(25).h5", "$trial/B_base")
  #   # B_base = convert(Matrix{Int16}, B_base) .+ 1
  #   # @show qerror(Xt, B, C), qerror(Xb, B_base, C)
  #
  #   C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
  #   # C, B, R, train_error, B_base, recall = Rayuela.experiment_lsq(Xt, B, C, R, Xb, Xq, gt, m-1, h, niter, knn, verbose)
  #   C, B, R, train_error, B_base, recall = Rayuela.experiment_lsq_cuda(Xt, B, C, R, Xb, Xq, gt, m-1, h, niter, knn, nsplits_train, nsplits_base, verbose)
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

end
