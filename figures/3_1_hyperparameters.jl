using Rayuela
using PyPlot
# Code to reproduce figure 3.1

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
      nsplits = 1

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


function make_figure_pyplot(ed)
  # This is the code that actually makes the plot
  fig = figure(figsize=(18, 9))

  subplotidx = 1
  for m = [7, 15], icmiter = [1, 2, 4, 8]
    nperts = ifelse(m==7, [1, 2, 4, 7], [1, 2, 4, 8, 15])
    ax = subplot(2, 4, subplotidx)
    counter = 1
    for npert in nperts

      # We want the total number of icmiters to be 64
      ilsiters = [1, 2, 4, 8, 16, 32, 64]
      ilsiters = [x for x in ilsiters if x >= icmiter]
      x = vcat([0], ilsiters)
      y = ed["m$(m)_icmiter$(icmiter)_npert$(npert)"]
      @show x, y
      PyPlot.plot( x, y, label="\$k=$(npert)\$", lw=2 )

      ax[:set_xlim]([0, 64])
      ylim = ifelse(m == 7, [40500, 46000], [21500, 29000])
      ax[:set_ylim](ylim)
      ax[:set_xticks]([0, 8, 16, 32, 64])

      sz = 18
      title("\$m=$(m)\$, $(icmiter) ICM iter.", size=sz)
      ylabel("Quantization error", size=sz)
      xlabel("Total ICM iterations", size=sz)
      ticklabel_format(style="sci", axis="y", scilimits=[0,0])
      legend(loc="upper right", fontsize=sz-2)#, ncol=2)
    end
    subplotidx += 1
  end
  PyPlot.tight_layout()
  # show()
  PyPlot.savefig( "/home/julieta/Desktop/hyperparams_SIFT1M.pdf" )
end


# experiment_data = run_experiment()
make_figure_pyplot(experiment_data)
