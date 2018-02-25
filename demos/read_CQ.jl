using Rayuela

bpath = "/home/julieta/Desktop/CQ/build/temp/"

Cpath = bpath * "D"
Bpath = bpath * "B"


# Should be fvecs?
m, h, d = 8, 256, 128

function read_cq_fvecs(fname::AbstractString)

  dim, count = 0, 0
  vectors = Matrix{Cfloat}

  open(fname, "r") do fid
    count = read(fid, Cint, 1)[1]
    dim   = read(fid, Cint, 1)[1]
    vectors = read(fid, Cfloat, dim, count)
  end

  # vectors = vectors'
  return vectors
end

function read_cq_bvecs(fname::AbstractString)

  dim, count = 0, 0
  codes = Matrix{Cint}

  open(fname, "r") do fid
    # seekend(fid); @show position(fid); seekstart(fid)
    count = read(fid, Cint, 1)[1]
    dim   = read(fid, Cint, 1)[1]
    codes = read(fid, Cint, dim, count)
    # @show position(fid)
  end

  # @show count, dim
  # codes = codes'
  return codes
  # return convert(Matrix{Int16}, codes)
  # return convert(Matrix{Cuchar}, codes)
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
verbose = false
Xt = Rayuela.read_dataset(dataset_name, Int(1e5))
Xb = Rayuela.read_dataset(dataset_name * "_base", Int(1e6))
Xq = Rayuela.read_dataset(dataset_name * "_query", nquery, verbose)[:,1:nquery]
gt = Rayuela.read_dataset(dataset_name * "_groundtruth", nquery, verbose)
if dataset_name == "SIFT1M" || dataset_name == "GIST1M"
  gt = gt .+ 1
end
gt = convert( Vector{UInt32}, gt[1,1:nquery] )

qerr = qerror(Xt, B, C)
@printf("%e\n", qerr )

# Encode the base
B_base = ones(Int16, 8, Int(1e6))
ilsiter, icmiter, randord, npert, cpp, V = 1, 3, false, 0, true, true
B_base = Rayuela.encoding_icm(Xb, B_base, C, ilsiter, icmiter, randord, npert, cpp, V)

qerr = qerror(Xb, B_base, C)
@printf("%e\n", qerr )

# Compute and quantize the database norms
# norms_B, norms_C = get_norms_codebook(B, C)
# B_base_norms = quantize_norms( B_base, C, norms_C )
db_norms1     = vec( norms_C[ B_base_norms ] )

db_norms = zeros(Float32, Int(1e6))
knn = Int(1e3)
if V; print("Querying m=$m ... "); end
@time dists, idx = linscan_lsq(B_base, Xq, C, db_norms, eye(Float32, d), knn)
if V; println("done"); end

rec = eval_recall(gt, idx, knn)

# @show B

# C = fvecs_read(m*h, C_path)
# C = fvecs_read(10, C_path)
# @show C
