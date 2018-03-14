using Rayuela
using HDF5


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


# === Data loading function ===
function load_experiment_data(
  dataset_name::String,
  ntrain::Integer, nbase::Integer, nquery::Integer, V::Bool=false)

  Xt = read_dataset(dataset_name, ntrain, V)
  Xb = read_dataset(dataset_name * "_base", nbase, V)
  Xq = read_dataset(dataset_name * "_query", nquery, V)[:,1:nquery]
  gt = read_dataset(dataset_name * "_groundtruth", nquery, V)
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

function load_chainq(fname::String, m::Integer, trial::Integer)

  B = h5read(fname, "$trial/B"); B = convert(Matrix{Int16}, B); B.+=1
  R = h5read(fname, "$trial/R")
  chainq_error = h5read(fname, "$trial/train_error")
  C = Vector{Matrix{Float32}}(m)
  for i=1:(m); C[i] = h5read(fname, "$trial/C_$i"); end

  return C, B, R, chainq_error
end

# === experiment functions ===
function run_demos(
  dataset_name="SIFT1M",
  ntrain::Integer=Int(1e5),
  m::Integer=8, h::Integer=256, niter::Integer=25)

  nquery, nbase, knn = 0, 0, 0
  if dataset_name == "SIFT1M" || dataset_name == "Deep1M"
    nquery, nbase, knn = Int(1e4), Int(1e6), Int(1e3)
  else
    error("dataset unknown")
  end

  verbose = true

  # Load data
  Xt, Xb, Xq, gt = load_experiment_data(dataset_name, ntrain, nbase, nquery, verbose)
  d, _    = size( Xt )

  ntrials = 10
  for trial = 1:ntrials

    # (Semi-)orthogonal methods: PQ, OPQ, ChainQ
    # C, B, train_error, B_base, recall = Rayuela.experiment_pq( Xt, Xb, Xq, gt, m, h, niter, knn, verbose)
    # save_results_pq("./results/$(lowercase(dataset_name))/pq_m$(m)_it$(niter).h5", trial, C, B, train_error, B_base, recall)
    #
    # C, B, R, train_error, B_base, recall = Rayuela.experiment_opq(Xt, Xb, Xq, gt, m, h, niter, knn, verbose)
    # save_results_opq("./results/$(lowercase(dataset_name))/opq_m$(m)_it$(niter).h5", trial, C, B, R, train_error, B_base, recall)

    # Cheap non-orthogonal methods: RVQ, ERVQ
    # C, B, train_error, B_base, recall = Rayuela.experiment_rvq( Xt,       Xb, Xq, gt, m-1, h, niter, knn, verbose)
    # save_results_pq("./results/$(lowercase(dataset_name))/rvq_m$(m-1)_it$(niter).h5",  trial, C, B, train_error, B_base, recall)
    #
    # C, B, train_error, B_base, recall = Rayuela.experiment_ervq(Xt, B, C, Xb, Xq, gt, m-1, h, niter, knn, verbose)
    # save_results_pq("./results/$(lowercase(dataset_name))/ervq_m$(m-1)_it$(niter).h5", trial, C, B, train_error, B_base, recall)

    # More expensive non-orthogonal methods
    # Rayuela.experiment_chainq(Xt, Xb, Xq, gt, m, h, niter, knn, verbose)
    # Rayuela.experiment_lsq(Xt, Xb, Xq, gt, m, h, niter, knn, verbose)
    # Rayuela.experiment_sr(Xt, Xb, Xq, gt, m, h, niter, knn, verbose)

    # Precompute init for LSQ/SR
    # C, B, R, train_error = Rayuela.train_opq(Xt, m-1, h, niter, "natural", verbose)
    # save_results_opq("./results/$(lowercase(dataset_name))/opq_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, ones(UInt16,1,1), [0f0])

    # Load OPQ
    # fname = "./results/$(lowercase(dataset_name))/opq_m$(m-1)_it$(niter).h5"

    # C, B, R, chainq_error = train_chainq(    Xt, m-1, h, R, B, C, niter, verbose)
    # save_results_opq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", trial, C, B, R, chainq_error, ones(UInt16,1,1), [0f0])

    @show trial
    nsplits_train = 1
    nsplits_base  = 3

    # Load ChainQ
    C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
    C, B, R, train_error, B_base, recall = Rayuela.experiment_lsq(Xt, B, C, R, Xb, Xq, gt, m-1, h, niter, knn, verbose)
    # C, B, R, train_error, B_base, recall = Rayuela.experiment_lsq_cuda(Xt, B, C, R, Xb, Xq, gt, m-1, h, niter, knn, nsplits_train, nsplits_base, verbose)
    save_results_lsq("./results/$(lowercase(dataset_name))/lsq_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, B_base, recall)

    # sr_method = "SR_D"
    # C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
    # C, B, R, train_error, B_base, recall = Rayuela.experiment_sr_cuda(Xt, B, C, R, Xb, Xq, gt, m-1, h, niter, knn, nsplits_train, nsplits_base, sr_method, verbose)
    # save_results_lsq("./results/$(lowercase(dataset_name))/srd_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, B_base, recall)
    #
    # sr_method = "SR_C"
    # C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
    # C, B, R, train_error, B_base, recall = Rayuela.experiment_sr_cuda(Xt, B, C, R, Xb, Xq, gt, m-1, h, niter, knn, nsplits_train, nsplits_base, sr_method, verbose)
    # save_results_lsq("./results/$(lowercase(dataset_name))/src_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, B_base, recall)

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

  end

  # return C, B, train_error, B_base, recall
  # return C, B, R, train_error, B_base, recall
end


function run_demos_query_base(
  dataset_name="labelme",
  ntrain::Integer=Int(2e3),
  m::Integer=8, h::Integer=256, niter::Integer=25)

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
  d, _    = size( Xt )

  ntrials = 10
  for trial = 1:ntrials
  # for trial = 1:1

    # (Semi-)orthogonal methods: PQ, OPQ, ChainQ
    # C, B, train_error, recall = Rayuela.experiment_pq_query_base(Xt, Xq, gt, m, h, niter, knn, verbose)
    # save_results_pq_query_base("./results/$(lowercase(dataset_name))/pq_m$(m)_it$(niter).h5", trial, C, B, train_error, recall)

    # C, B, R, train_error, recall = Rayuela.experiment_opq_query_base(Xt, Xq, gt, m, h, niter, knn, verbose)
    # save_results_opq_query_base("./results/$(lowercase(dataset_name))/opq_m$(m)_it$(niter).h5", trial, C, B, R, train_error, recall)

    # Cheap non-orthogonal methods: RVQ, ERVQ
    # C, B, train_error, recall = Rayuela.experiment_rvq_query_base(Xt,        Xq, gt, m-1, h, niter, knn, verbose)
    # save_results_pq_query_base("./results/$(lowercase(dataset_name))/rvq_m$(m-1)_it$(niter).h5",  trial, C, B, train_error, recall)

    # C, B, train_error, recall = Rayuela.experiment_ervq_query_base(Xt, B, C, Xq, gt, m-1, h, niter, knn, verbose)
    # save_results_pq_query_base("./results/$(lowercase(dataset_name))/ervq_m$(m-1)_it$(niter).h5", trial, C, B, train_error, recall)

    # More expensive non-orthogonal methods
    # Rayuela.experiment_chainq(Xt, Xb, Xq, gt, m, h, niter, knn, verbose)
    # Rayuela.experiment_lsq(Xt, Xb, Xq, gt, m, h, niter, knn, verbose)
    # Rayuela.experiment_sr(Xt, Xb, Xq, gt, m, h, niter, knn, verbose)


    # Precompute init for LSQ/SR
    # C, B, R, train_error = Rayuela.train_opq(Xt, m-1, h, niter, "natural", verbose)
    # save_results_opq_query_base("./results/$(lowercase(dataset_name))/opq_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, [0f0])
    #
    # C, B, R, chainq_error = Rayuela.train_chainq(    Xt, m-1, h, R, B, C, niter, verbose)
    # save_results_opq_query_base("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", trial, C, B, R, chainq_error, [0f0])

    # Load ChainQ

    @show trial

    nsplits_train = 1

    C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
    C, B, R, train_error, recall = Rayuela.experiment_lsq_cuda_query_base(Xt, B, C, R, Xq, gt, m-1, h, niter, knn, nsplits_train, verbose)
    save_results_lsq_query_base("./results/$(lowercase(dataset_name))/lsq_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, recall)

    sr_method = "SR_D"
    C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
    C, B, R, train_error, recall = Rayuela.experiment_sr_cuda_query_base( Xt, B, C, R, Xq, gt, m-1, h, niter, knn, nsplits_train, sr_method, verbose)
    save_results_lsq_query_base("./results/$(lowercase(dataset_name))/srd_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, recall)

    sr_method = "SR_C"
    C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
    C, B, R, train_error, recall = Rayuela.experiment_sr_cuda_query_base( Xt, B, C, R, Xq, gt, m-1, h, niter, knn, nsplits_train, sr_method, verbose)
    save_results_lsq_query_base("./results/$(lowercase(dataset_name))/src_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, recall)


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

  # return C, B, train_error, B_base, recall
  # return C, B, R, train_error, B_base, recall
end

for niter = [25]#, 50, 100]
  # run_demos("SIFT1M", Int(1e5),  8, 256, niter)
  # run_demos("Deep1M", Int(1e5),  8, 256, niter)
  # run_demos("SIFT1M", Int(1e5), 16, 256, niter)
  run_demos("Deep1M", Int(1e5), 16, 256, niter)
end

# run_demos_query_base("labelme", Int(20e3), 8,  256, 25)
# run_demos_query_base("labelme", Int(20e3), 16, 256, 25)
# run_demos_query_base("MNIST",   Int(60e3), 8,  256, 25)
# run_demos_query_base("MNIST",   Int(60e3), 16, 256, 25)

# run_demos
# for niter = [25, 50, 100]
#   run_demos_query_base("labelme", Int(20e3), 8, 256, niter)
#   run_demos_query_base("MNIST",   Int(60e3), 8, 256, niter)
#   run_demos_query_base("labelme", Int(20e3), 16, 256, niter)
#   run_demos_query_base("MNIST",   Int(60e3), 16, 256, niter)
# end
