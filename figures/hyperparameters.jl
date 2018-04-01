using Rayuela

# Code to reproduce figure 3.1 in the paper.

function run_experiment()
  V = true
  X = read_dataset("SIFT1M", Int(1e5), V)
  X = X[:,1:Int(1e4)] # Keep the first 10 000 points

  max_iter = 64
  h = 256

  for m = [7, 15], icmiter = [1, 2, 4, 8]
    nperts = ifelse(m==7, [1, 2, 4, 7], [1, 2, 4, 8, 15])
    for npert in nperts
      B = convert(Matrix{Int16}, rand(1:h, m, size(X,2)))
      C = Rayuela.update_codebooks_fast_bin(X, B, h, V)

      # We want the total number of icmiters to be 64
      max_ilsiters = 64 / icmiter
      ilisters = [1, 2, 4, 8, 16, 32, 64]
      ilsiters_idx = find(max_ilsiters .== ilsiters )[1]
      ilisters = ilisters[1:ilsiters_idx]
      print( ilsiters )

      # Bs, objs = Rayuela.encode_icm_cuda(X, B, C, )
    end
  end

end

run_experiment()
# make_figure()
