module smac_util

using Rayuela
using HDF5

export run_demos_query_base, run_demos_train_query_base

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
  C = Vector{Matrix{Float32}}(m)
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


# === demos query base ===
function run_demos_query_base(
  dataset_name="labelme",
  m::Integer=8,
  h::Integer=256,
  niter::Integer=5,
  sr_method::String="SR_D",
  ilsiter::Integer=8,
  icmiter::Integer=4,
  randord::Bool=true,
  npert::Integer=4,
  schedule::Integer=1,
  p::Float64=0.5)

  if !(sr_method in ["LSQ", "SR_D", "SR_C"])
    error("Unknown sr_method $sr_method")
  end

  nquery, nbase, knn = 0, 0, 0
  if dataset_name == "MNIST"
    ntrain, nquery, nbase, knn = Int(60e3), Int(10e3), Int(60e3), Int(1e3)
  elseif dataset_name == "labelme"
    ntrain, nquery, nbase, knn = Int(20e3), Int(2e3), Int(20019), Int(1e3)
    # ntrain, nquery, nbase, knn = Int(20019), Int(2e3), Int(20019), Int(1e3)
  else
    error("dataset unknown")
  end
  # @show ntrain, nquery, nbase, knn

  verbose = true

  Xt, Xb, Xq, gt = load_experiment_data(dataset_name, ntrain, nbase, nquery, verbose)
  @show size(Xt), size(Xb), size(Xq), size(gt)
  d, _ = size(Xt)

  nsplits_train = 1

  recall = [0]
  # trial = rand(1:10, 1)[1]
  for trial = 1:10
    if sr_method == "LSQ"
      C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
      C, B, R, train_error, recall = Rayuela.experiment_lsq_cuda_query_base(Xt, B, C, R, Xq, gt, m-1, h, niter, ilsiter,
          icmiter, randord, npert, knn, nsplits_train, verbose)
      save_results_lsq_query_base("./results/smac/$(lowercase(dataset_name))/lsq_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, recall)
    elseif sr_method == "SR_D"
      C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
      C, B, R, train_error, recall = Rayuela.experiment_sr_cuda_query_base(Xt, B, C, R, Xq, gt, m-1, h, niter, ilsiter,
          icmiter, randord, npert, knn, nsplits_train, sr_method, schedule, p, verbose)
        save_results_lsq_query_base("./results/smac/$(lowercase(dataset_name))/srd_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, recall)
    elseif sr_method == "SR_C"
      C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
      C, B, R, train_error, recall = Rayuela.experiment_sr_cuda_query_base(Xt, B, C, R, Xq, gt, m-1, h, niter, ilsiter,
          icmiter, randord, npert, knn, nsplits_train, sr_method, schedule, p, verbose)
      save_results_lsq_query_base("./results/smac/$(lowercase(dataset_name))/src_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, recall)
    end
  end

end

function run_demos_train_query_base(
  dataset_name="SIFT1M",
  m::Integer=8,
  h::Integer=256,
  niter::Integer=5,
  sr_method::String="SR_D",
  ilsiter::Integer=8,
  icmiter::Integer=4,
  randord::Bool=true,
  npert::Integer=4,
  schedule::Integer=1,
  p::Float64=0.5)

  if !(sr_method in ["LSQ", "SR_D", "SR_C"])
    error("Unknown sr_method $sr_method")
  end

  nquery, nbase, knn = 0, 0, 0
  if dataset_name == "SIFT1M" || dataset_name == "Deep1M" || dataset_name == "Convnet1M"
    ntrain, nquery, nbase, knn = Int(1e5), Int(1e4), Int(1e6), Int(1e3)
  else
    error("dataset unknown")
  end
  # @show ntrain, nquery, nbase, knn

  verbose = true

  Xt, Xb, Xq, gt = load_experiment_data(dataset_name, ntrain, nbase, nquery, verbose)
  d, _ = size(Xt)

  nsplits_train = 1
  nsplits_base = 4

  recall = [0]
  # trial = rand(1:10, 1)[1]
  for trial = 1:10
    if sr_method == "LSQ"
      C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
      C, B, R, train_error, B_base, recall = Rayuela.experiment_lsq_cuda(Xt, B, C, R, Xb, Xq, gt, m-1, h, niter, ilsiter,
          icmiter, randord, npert, knn, nsplits_train, nsplits_base, verbose)
      save_results_lsq("./results/smac/$(lowercase(dataset_name))/lsq_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, B_base, recall)

    elseif sr_method == "SR_D"
      C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
      C, B, R, train_error, B_base, recall = Rayuela.experiment_sr_cuda(Xt, B, C, R, Xb, Xq, gt, m-1, h, niter, ilsiter,
          icmiter, randord, npert, knn, nsplits_train, nsplits_base, sr_method, schedule, p, verbose)
      save_results_lsq("./results/smac/$(lowercase(dataset_name))/srd_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, B_base, recall)

    elseif sr_method == "SR_C"
      C, B, R, chainq_error = load_chainq("./results/$(lowercase(dataset_name))/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
      C, B, R, train_error, B_base, recall = Rayuela.experiment_sr_cuda(Xt, B, C, R, Xb, Xq, gt, m-1, h, niter, ilsiter,
          icmiter, randord, npert, knn, nsplits_train, nsplits_base, sr_method, schedule, p, verbose)
      save_results_lsq("./results/smac/$(lowercase(dataset_name))/src_m$(m-1)_it$(niter).h5", trial, C, B, R, train_error, chainq_error, B_base, recall)
    end
  end

end

# Query/base datasets
for niter = [100]
  # run_demos_query_base("labelme", 8,  256, niter, "SR_D", 9, 3, true,  1, 1, 0.43098784299895454)
  # # run_demos_query_base("labelme", 16, 256, niter, "SR_D", 8, 4, true,  4, 1, 0.5)  # No change here
  # run_demos_query_base("MNIST",   8,  256, niter, "SR_D", 9, 3, false, 5, 1, 0.18979255389609623)
  # run_demos_query_base("MNIST",   16, 256, niter, "SR_D", 8, 4, false, 4, 1, 0.8282107865533627)
end

# Train/Query/base datasets
# TODO change the function so it savez the results
for niter = [25, 100]
  run_demos_train_query_base("SIFT1M",    8,  256, niter, "SR_D", 8,  4, true,  4, 1, 0.6458745069743886)
  run_demos_train_query_base("SIFT1M",    16, 256, niter, "SR_D", 7,  4, true,  2, 1, 0.18722222602931293)

  # run_demos_train_query_base("Deep1M",    8,  256, niter, "SR_D", 8,  4, true,  4, 1, 0.5) # No change here
  run_demos_train_query_base("Deep1M",    16, 256, niter, "SR_C", 15, 2, true,  2, 1, 0.9534092523209057)

  run_demos_train_query_base("Convnet1M", 8,  256, niter, "SR_C", 8,  4, true,  4, 1, 0.7134116312190524)
  run_demos_train_query_base("Convnet1M", 16, 256, niter, "SR_C", 10, 3, false, 5, 1, 0.937363908221641)
end

end # module smac_util
