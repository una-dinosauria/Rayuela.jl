

bpath = "/home/julieta/Desktop/CQ/build/temp/"

Cpath = bpath * "D"
Bpath = bpath * "B"

# Read float binary file produced by CQ
function read_cq_fvecs(fname::AbstractString)

  dim, count = zero(Cint), zero(Cint)
  vectors = Matrix{Cfloat}

  open(fname, "r") do fid
    count = read(fid, Cint, 1)[1]
    dim   = read(fid, Cint, 1)[1]
    vectors = read(fid, Cfloat, dim, count)
  end

  return vectors
end

# Read int binary file produced by CQ
function read_cq_bvecs(fname::AbstractString)

  dim, count = zero(Cint), zero(Cint)
  codes = Matrix{Cint}

  open(fname, "r") do fid
    count = read(fid, Cint, 1)[1]
    dim   = read(fid, Cint, 1)[1]
    codes = read(fid, Cint, dim, count)
  end

  return codes
end

function inner_terms(C, B)
  m, n = size(B)
  d, h = size(C[1])

  # Compute squared codebook norms
  Cn = Vector{Vector{Float32}}(m)
  for i = 1:m
    Cn[i] = vec(sum(C[i].^2, 1))
  end

  # make space
  it = zeros(Float32, n)
  for i = 1:m
    it += Cn[i][B[i,:]]
  end

  return it
end


K = read_cq_fvecs(Cpath)
B = read_cq_bvecs(Bpath)
B = convert(Matrix{Int16}, B)
B .+= 1

C = Vector{Matrix{Cfloat}}(m)
indices = splitarray(1:(m*h), m)
for i = 1:m
  C[i] = K[:, indices[i]]
end

dataset_name = "SIFT1M"
nquery = Int(1e4)
verbose = true
Xt = Rayuela.read_dataset(dataset_name, Int(1e5))
Xb = Rayuela.read_dataset(dataset_name * "_base", Int(1e6))
Xq = Rayuela.read_dataset(dataset_name * "_query", nquery, verbose)[:,1:nquery]
gt = Rayuela.read_dataset(dataset_name * "_groundtruth", nquery, verbose)
if dataset_name == "SIFT1M" || dataset_name == "GIST1M"
  gt = gt .+ 1
end
gt = convert( Vector{UInt32}, gt[1,1:nquery] )

qerr = qerror(Xb, B, C)
@printf("%e\n", qerr )

# # Encode the base
# B_base = ones(Int16, 8, Int(1e6))
# ilsiter, icmiter, randord, npert, cpp, V = 1, 3, false, 0, true, true
# B_base = Rayuela.encoding_icm(Xb, B_base, C, ilsiter, icmiter, randord, npert, cpp, V)
#
# qerr = qerror(Xb, B_base, C)
# @printf("%e\n", qerr )

# Compute and quantize the database norms
# norms_B, norms_C = get_norms_codebook(B, C)
# B_base_norms = quantize_norms( B_base, C, norms_C )
# db_norms1     = vec( norms_C[ B_base_norms ] )

# Compute database norms. No need to quantize them?

knn = Int(1e3)

# Brute-force approach to search gives R@1 of 30
# idx = zeros(UInt32, knn, nquery)
# @time dists = Distances.pairwise(Distances.SqEuclidean(), Xq, CB)
# #Threads.@threads
# for i = 1:nquery
#   if i % 100 == 0
#     @show i
#   end
#   idx[:,i] = sortperm(dists[i,:]; alg=PartialQuickSort(knn))[1:knn]
# end

# Fast search
db_norms = zeros(Float32, Int(1e6))
CB      = Rayuela.reconstruct(B, C)
R = eye(Float32, d)
dbnorms = zeros(Float32, Int(1e6))
# dbnorms = vec(sum(CB.^2, 1))
# it = inner_terms(C,B)
# dbnorms -= it


if verbose; print("Querying m=$m ... "); end
# @time dists, idx = linscan_lsq(B, Xq, C, db_norms, R, knn)
# @time dists, idx = linscan_cq(B, Xq, C, db_norms, knn)
@time dists, idx = linscan_cq(B, Xq, C, knn)
if verbose; println("done"); end

rec = eval_recall(gt, idx, knn)

# @show B

# C = fvecs_read(m*h, C_path)
# C = fvecs_read(10, C_path)
# @show C
