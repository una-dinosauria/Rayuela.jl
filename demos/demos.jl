using Rayuela


function run_demos(
  # dataset_name="SIFT1M",
  # ntrain::Integer=Int(1e5)) # Increase this to 1e5 to use the full dataset
  dataset_name="MNIST",
  ntrain::Integer=Int(60e3)) # Increase this to 1e5 to use the full dataset
  # dataset_name="GIST1M",
  # ntrain::Integer=Int(5e5)) # Increase this to 1e5 to use the full dataset

  # Experiment params
  m, h = 8, 256
  nquery, nbase, knn = Int(1e4), Int(1e6), Int(1e3)
  # nquery, nbase, knn = Int(1e3), Int(1e6), Int(1e3)
  niter, verbose = 25, true
  b       = Int(log2(h) * m)

  # Load data
  Xt = read_dataset(dataset_name, ntrain)
  Xb = read_dataset(dataset_name * "_base", nbase)
  Xq = read_dataset(dataset_name * "_query", nquery, verbose)[:,1:nquery]
  gt = read_dataset(dataset_name * "_groundtruth", nquery, verbose)
  if dataset_name == "SIFT1M" || dataset_name == "GIST1M"
    gt = gt .+ 1
  end
  gt = convert( Vector{UInt32}, gt[1,1:nquery] )
  d, _    = size( Xt )

  # (Semi-)orthogonal methods: PQ, OPQ, ChainQ
  # Rayuela.experiment_pq( Xt, Xb, Xq, gt, m, h, niter, knn, verbose)
  # C, B, R = Rayuela.experiment_opq(Xt, Xb, Xq, gt, m, h, niter, knn, verbose)

  # Cheap non-orthogonal methods: RVQ, ERVQ
  # m = m - 1

  # C, B = Rayuela.experiment_rvq( Xt, Xb, Xq, gt, m, h, niter, knn, verbose)
  # Rayuela.experiment_ervq(Xt, B, C, Xb, Xq, gt, m, h, niter, knn, verbose)

  # More expensive non-orthogonal methods
  # Rayuela.experiment_chainq(Xt, Xb, Xq, gt, m, h, niter, knn, verbose)
  # Rayuela.experiment_lsq(Xt, Xb, Xq, gt, m, h, niter, knn, verbose)
  # Rayuela.experiment_sr(Xt, Xb, Xq, gt, m, h, niter, knn, verbose)

  # GPU methods
  nsplits_train = 1
  nsplits_base  = 2
  Rayuela.experiment_lsq_cuda(Xt, Xb, Xq, gt, m, h, niter, knn, nsplits_train, nsplits_base, verbose)
  # Rayuela.experiment_sr_cuda(Xt, Xb, Xq, gt, m, h, niter, knn, nsplits_train, nsplits_base, verbose)


  # GPU methods with random inputs
  # B = convert(Matrix{Int16}, rand(1:h, m, size(Xt,2)))
  # C = Vector{Matrix{Float32}}(m); for i=1:m; C[i]=zeros(Float32,d,h); end
  # R = eye(Float32, d)
  # Rayuela.experiment_lsq_cuda(Xt, B, C, R, Xb, Xq, gt, m, h, niter, knn, verbose)
  # Rayuela.experiment_sr_cuda( Xt, B, C, R, Xb, Xq, gt, m, h, niter, knn, verbose)

end

# run_demos
run_demos()
