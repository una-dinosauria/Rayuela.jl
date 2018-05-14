using Rayuela
using HDF5
include("demos.jl")

dataset_name="SIFT1B"

h=256
niter=100

ntrain, nbase = Int(1e6), Int(1e9)
nquery, knn   = Int(1e4), Int(1e3)

V = verbose = true

# Load training data
Xt = read_dataset(dataset_name, ntrain, V)
# Xb = read_dataset(dataset_name * "_base", nbase, V)
Xq = read_dataset(dataset_name * "_query", nquery, V)[:,1:nquery]

gt = []
if nbase == Int(1e9)
  gt = read_dataset(dataset_name * "_groundtruth"), nquery, V)
else
  gt = read_dataset(dataset_name * "_groundtruth_10M"), nquery, V)
end
gt .+= 1
gt = convert( Vector{UInt32}, gt[1,1:nquery] )

println("Done loading dataset")

idx = []
train_error = []

ntrials = 1

# Specific function for codebooks
function save(bpath::String, trial::Integer, hname::String, C::Vector{Matrix{Float32}})
  @assert hname == "C"
  for i = 1:length(C)
    h5write(bpath, "$(trial)/C_$i", C[i])
  end
end

# Spefific function for codes
function save(bpath::String, trial::Integer, hname::String, B::Matrix{Int16})
  h5write(bpath, "$(trial)/$hname", convert(Matrix{UInt8}, B.-1))
end


# Generic fallback function
function save(bpath::String, trial::Integer, hname::String, X)
  h5write(bpath, "$(trial)/$hname", X)
end

# Call this to save all the outputs
function save_all(bpath::String, trial::Integer, hnames::Vector{String}, Xs::Vector)
  @assert length(hnames) == length(Xs)
  for (hname, X) in zip(hnames, Xs)
    save(bpath, trial, hname, X)
  end
end

function load(bpath::String, trial::Integer, m::Integer, hname::String)
  @show bpath, trial, m, hname
  if hname == "C"
    X = Vector{Matrix{Float32}}(m)
    for i = 1:m; X[i] = h5read(bpath, "$trial/C_$i"); end
  else
    X = h5read(bpath, "$trial/$hname")
  end
  X
end

function load_all(bpath::String, trial::Integer, m::Integer, hnames::Vector{String})
  Xs = []
  for hname in hnames
    push!(Xs, load(bpath, trial, m, hname))
  end
  Xs
end

# Training part
m = 0
B = []
C = []
for m = [8, 16]
  # # (Semi-)orthogonal methods: PQ, OPQ, ChainQ
  # for trial = 1:ntrials
  #   C, B, train_error = Rayuela.train_pq(Xt, m, h, niter, verbose)
  #   fname = "/hdd/results/sift10m/pq_m$(m)_it$(niter).h5"
  #   save_all(fname, trial, ["C", "B", "train_error"], [C, B, train_error])
  # end
  # for trial = 1:ntrials
  #   C, B, R, train_error = Rayuela.train_opq(Xt, m, h, niter, "natural", verbose)
  #   fname = "/hdd/results/sift10m/opq_m$(m)_it$(niter).h5"
  #   save_all(fname, trial,  ["C", "B", "R", "train_error"], [C, B, R, train_error])
  # end
  #
  # # Cheap non-orthogonal methods: RVQ, ERVQ
  # for trial = 1:ntrials
  #   C, B, train_error = Rayuela.train_rvq(Xt, m-1, h, niter, verbose)
  #   fname = "/hdd/results/sift10m/rvq_m$(m-1)_it$(niter).h5"
  #   save_all(fname, trial, ["C", "B", "train_error"], [C, B, train_error])
  #
  #   # ERVQ
  #   C, B, train_error = Rayuela.train_ervq(Xt, B, C, m-1, h, niter, verbose)
  #   fname = "/hdd/results/sift10m/ervq_m$(m-1)_it$(niter).h5"
  #   save_all(fname, trial, ["C", "B", "train_error"], [C, B, train_error])
  # end
  #
  # # Precompute init for LSQ/SR
  # for trial = 1:ntrials
  #   C, B, R, train_error = Rayuela.train_opq(Xt, m-1, h, niter, "natural", verbose)
  #   fname = "/hdd/results/sift10m/opq_m$(m-1)_it$(niter).h5"
  #   save_all(fname, trial, ["C", "B", "R", "train_error"], [C, B, R, train_error])
  # end
  # for trial = 1:ntrials
  #   C, B, R, _ = load_chainq("/hdd/results/sift10m/opq_m$(m-1)_it$(niter).h5", m-1, trial)
  #   C, B, R, chainq_error = Rayuela.train_chainq(Xt, m-1, h, R, B, C, niter, verbose)
  #   fname = "/hdd/results/sift10m/chainq_m$(m-1)_it$(niter).h5"
  #   save_all(fname, trial, ["C", "B", "R", "train_error"], [C, B, R, chainq_error])
  # end

  # nsplits_train =  m == 8 ? 1 : 2
  #
  # for trial = 1:ntrials
  #   C, B, R, chainq_error = load_chainq("/hdd/results/sift10m/chainq_m$(m-1)_it$(niter).h5", m-1, trial)
  #
  #   ilsiter = 8
  #   icmiter = 4
  #   randord = true
  #   npert   = 4
  #   cpp     = true
  #
  #   C, B, train_error = Rayuela.train_lsq_cuda(Xt, m-1, h, R, B, C, niter, ilsiter, icmiter, randord, npert, nsplits_train, V)
  #
  #   fname = "/hdd/results/sift10m/lsq_m$(m-1)_it$(niter).h5"
  #   save_all(fname, trial, ["C", "B", "R", "train_error"], [C, B, R, train_error])
  # end

end

nsplits = nbase == Int(1e9) ? 1000 : 20
dname = "sift1b"

C, B, R = [], [], []

# methods = [:pq, :opq, :rvq:, :ervq, :lsq]
# methods = [:rvq, :ervq, :lsq]
methods = [:pq, :opq]
base_error_xt = 0.f0

B_base, base_error, recall = [], [], []

for method = methods, m = [8, 16]
  # (Semi-)orthogonal methods: PQ, OPQ, ChainQ
  for trial = 1:ntrials

    @show method, m

    loadname, fname = "", ""
    if method == :pq || method == :opq
      loadname = "/hdd/results/sift10m/$(method)_m$(m)_it$(niter).h5"
      fname = "/hdd/results/$dname/$(method)_m$(m)_it$(niter).h5"
    else
      loadname = "/hdd/results/sift10m/$(method)_m$(m-1)_it$(niter).h5"
      fname = "/hdd/results/$dname/$(method)_m$(m-1)_it$(niter).h5"
    end
    @show fname

    if method == :pq
      C, B, train_error = load_all(loadname, trial, m, ["C", "B", "train_error"])
    elseif method == :rvq || method == :ervq || method == :lsq
      C, B, train_error = load_all(loadname, trial, m-1, ["C", "B", "train_error"])
    elseif method == :opq
      C, B, R, train_error = load_all(fname, trial, m, ["C", "B", "R", "train_error"])
    end

    B_base = zeros(UInt8, m, nbase)
    if method == :rvq || method == :ervq || method == :lsq
      B_base = zeros(UInt8, m-1, nbase)
    end
    base_error = 0.

    for (i, xt_range) in enumerate(Rayuela.splitarray(1:nbase, nsplits))
      print("$i/$nsplits ", xt_range, " ")
      Xb = read_dataset(dataset_name * "_base", xt_range, V)

      # Quantize and compute error
      if method == :pq
        B_base_xt = Rayuela.quantize_pq(Xb, C, V)
        base_error_xt = qerror_pq(Xb, B_base_xt, C)
      elseif method == :opq
        B_base_xt = Rayuela.quantize_opq(Xb, R, C, V)
        base_error_xt = qerror_opq(Xb, B_base_xt, C, R)
      elseif method == :rvq || method == :ervq
        B_base_xt, _ = quantize_rvq(Xb, C, V)
        base_error_xt = qerror(Xb, B_base_xt, C)
      elseif method == :lsq
        # TODO compute dbnorms. Do it for 16, 32 iterations
        B_base_xt, _ = quantize_rvq(Xb, C, V)
        base_error_xt = qerror(Xb, B_base_xt, C)
      end

      @printf("Error on this partition is %e\n", base_error_xt)
      base_error += base_error_xt

      # Save to main B_base
      B_base[:, xt_range] = convert(Matrix{UInt8}, B_base_xt .- 1)
    end
    if V; @printf("Error in base is %e\n", base_error / nsplits); end

    if V; println("Querying m=$m ... "); end
    if method == :pq
      @time dists, idx = linscan_pq(B_base, Xq, C, Int(log2(h) * m), knn)
    elseif method == :opq
      @time dists, idx = linscan_opq(B_base, Xq, C, Int(log2(h) * m), R, knn)
    else
      @time dists, idx = linscan_lsq(B_base, Xq, C, dbnorms, R, knn)
    end
    if V; println("done"); end
    recall = eval_recall(gt, idx, knn)

    save_all(fname, trial, ["B_base", "base_error", "recall"], [B_base, base_error, recall])
  end
end
