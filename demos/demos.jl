using Rayuela

function encode_base_and_test(
  x_query::Matrix{T1},    # Queries
  B_base::Matrix{T2},     # encoded base, 1-based still
  C::Vector{Matrix{T1}},  # Learned codebooks
  gt::Vector{UInt32},     # Ground truth
  knn::Integer,           # Largers N in recall@N
  V::Bool) where {T1 <: AbstractFloat, T2 <: Integer}

  m, h = length(C), size(C[1],2)
  b = Int(log2(h) * m)

  # === Compute recall ===
  if V; println("Querying m=$m ... "); end
  @time dists, idx = linscan_pq(convert(Matrix{UInt8}, B_base-1), x_query, C, b, knn)
  if V; println("done"); end

  recall_at_n = eval_recall( gt, idx, knn )
  return recall_at_n
end


function run_demos(
  dataset_name="SIFT1M",
  ntrain::Integer=Int(1e5)) # Increase this to 1e5 to use the full dataset

  # Experiment params
  m, h = 8, 256
  nquery, nbase, knn = Int(1e4), Int(1e6), Int(1e3)
  niter, verbose = 25, true
  b       = Int(log2(h) * m)

  # Load data
  Xt = read_dataset(dataset_name, ntrain)
  Xb = read_dataset(dataset_name * "_base", nbase)
  Xq = read_dataset(dataset_name * "_query", nquery, verbose)[:,1:nquery]
  gt = read_dataset(dataset_name * "_groundtruth", nquery, verbose )
  if dataset_name == "SIFT1M" || dataset_name == "GIST1M"
    gt = gt .+ 1
  end
  gt = convert( Vector{UInt32}, gt[1,1:nquery] )
  d, _    = size( Xt )

  # ==========================
  # === Orthogonal methods ===
  # ==========================

  # PQ & OPQ
  # Rayuela.experiment_pq( Xt, Xb, Xq, gt, m, h, niter, knn, verbose)
  Rayuela.experiment_opq(Xt, Xb, Xq, gt, m, h, niter, knn, verbose)

  # ==============================
  # === Non-orthogonal methods ===
  # ==============================
  m = m - 1
  # Rayuela.experiment_rvq(Xt, Xb, Xq, gt, m, h, niter, knn, verbose)
  # ERVQ

end

# run_demos
run_demos()
