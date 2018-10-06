using Rayuela
using HDF5

# === Saving functions to keep track of experimental results ===
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
  C = Vector{Matrix{Float32}}(undef, m)
  for i=1:(m); C[i] = h5read(fname, "$trial/C_$i"); end
  return C, B, rvq_error
end

# === Data loading function ===
function load_experiment_data(
  dataset_name::String,
  ntrain::Integer, nbase::Integer, nquery::Integer, V::Bool=false)

  # Load training, base, queries and ground truth
  Xt = Rayuela.read_dataset(dataset_name, ntrain, V)
  Xb = Rayuela.read_dataset(dataset_name * "_base", nbase, V)
  Xq = Rayuela.read_dataset(dataset_name * "_query", nquery, V)[:,1:nquery]
  gt = Rayuela.read_dataset(dataset_name * "_groundtruth", nquery, V)

  # SIFT1M and GIST1M have zero-based indexing for their ground truth
  if dataset_name == "SIFT1M" || dataset_name == "GIST1M"
    gt = gt .+ 1
  end

  # Keep only the top neighbour in the ground truth
  if dataset_name != "Deep1M"
    gt = convert( Vector{UInt32}, gt[1,1:nquery] )
  else
    gt = convert( Vector{UInt32}, gt[1:nquery] )
  end

  return Xt, Xb, Xq, gt
end
