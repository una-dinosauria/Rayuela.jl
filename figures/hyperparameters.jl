using Rayuela

# Code to reproduce figure 3.1 in the paper.

function run_experiment()

  experiment_data = Dict{String,Vector{Float32}}()

  V = false
  X = read_dataset("SIFT1M", Int(1e5), V)
  X = X[:,1:Int(1e4)] # Keep the first 10 000 points

  max_iter = 64
  h = 256

  for m = [7, 15], icmiter = [1, 2, 4, 8]
    nperts = ifelse(m==7, [1, 2, 4, 7], [1, 2, 4, 8, 15])
    for npert in nperts

      randord = false
      nsplits = 2

      # We want the total number of icmiters to be 64
      max_ilsiters = 64 / icmiter
      ilsiters = [1, 2, 4, 8, 16, 32, 64]
      ilsiters_idx = find(max_ilsiters .== ilsiters )[1]
      ilsiters = ilsiters[1:ilsiters_idx]
      @show m, icmiter, nperts, ilsiters

      # Random initialization for B and C
      B = convert(Matrix{Int16}, rand(1:h, m, size(X,2)))
      C = Rayuela.update_codebooks_fast_bin(X, B, h, V)
      init_obj = [Rayuela.qerror(X, B, C)]

      # Now actually encode
      Bs, objs = Rayuela.encode_icm_cuda(X, B, C, ilsiters, icmiter, npert, randord, nsplits, V)

      objs = vcat(init_obj, objs)
      experiment_data["m$(m)_icmiter$(icmiter)_npert$(npert)"] = objs

      @show length(objs)
    end
  end

  return experiment_data
end

experiment_data = run_experiment()
make_figure(experiment_data)
