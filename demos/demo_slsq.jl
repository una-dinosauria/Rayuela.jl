using Rayuela, HDF5

# Compute recalls
ntrials = 10
bname = "/home/julieta/.julia/v0.6/Rayuela/results/sift1m"
dataset_name = "SIFT1M"
m, h = 7, 256
knn = 1000

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
  gt = convert( Vector{UInt32}, gt[1,1:nquery] )
  return Xt, Xb, Xq, gt
end

# Load data
V = true
 Xt, Xb, Xq, gt = load_experiment_data(dataset_name, Int(1e5), Int(1e6), Int(1e4), V)

# for trial = 1:ntrials, fname = ["SIFT1M_sc1.h5", "SIFT1M_sc2.h5"]
for trial = 1:ntrials, fname = ["SIFT1M_sc2.h5"]

  # Load codes and codebooks
  B = h5read(joinpath(bname, fname), "trial$trial/B")
  C = Vector{Matrix{Float32}}(m)
  for i = 1:m; C[i] = h5read(joinpath(bname, fname), "trial$trial/C_$i"); end
  non_zeros = 0
  for i = 1:m; non_zeros += sum((C[i] .!= 0)); end
  @show fname, non_zeros, trial, qerror(Xt, B, C)
  norms_B, norms_C = get_norms_codebook(B, C)

  # === Encode the base set ===
  icmiter = 4
  randord = true
  npert   = 4
  nsplits_base = 2

  ilsiters = [16, 32]
  B_base = convert(Matrix{Int16}, rand(1:h, m, size(Xb,2)))
  Bs_base, _ = Rayuela.encode_icm_cuda(Xb, B_base, C, ilsiters, icmiter, npert, randord, nsplits_base, V)

  for (idx, ilsiter) in enumerate(ilsiters)
    B_base = Bs_base[idx]
    base_error = qerror(Xb, B_base, C)
    if V; @printf("Error in base is %e\n", base_error); end

    # Compute and quantize the database norms
    B_base_norms, db_norms_X = quantize_norms(B_base, C, norms_C)
    db_norms     = vec( norms_C[ B_base_norms ] )

    if V; print("Querying m=$m ... "); end
    @time dists, idx = linscan_lsq(B_base, Xq, C, db_norms, eye(Float32, size(Xb,1)), knn)
    if V; println("done"); end

    recall = eval_recall(gt, idx, knn)
    h5write(joinpath(bname, fname), "trial$trial/recall_$ilsiter", recall)
    # bpath = "./results/large_recalls/srd_m$(m)_it$(niter).h5"
    # h5write(bpath, "$trial/recall_$(ilsiter)", recall)
    # h5write(bpath, "$trial/B_Base_$(ilsiter)", convert(Matrix{UInt8}, B_base.-1))
  end

end
