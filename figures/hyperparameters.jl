using Rayuela
using Plots
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

function make_figure(ed)
  pyplot() # <-- use pyplot as a backend

  xs = [0, 1, 2, 4, 8, 16, 32, 64]
  ylim = (41000, 46000)
  p1 = plot(xs, ed["m7_icmiter1_npert1"], label="1", w=2, ylim=ylim)
  plot!(xs, ed["m7_icmiter1_npert2"], label="2", w=2, ylim=ylim)
  plot!(xs, ed["m7_icmiter1_npert4"], label="4", w=2, ylim=ylim)
  plot!(xs, ed["m7_icmiter1_npert7"], label="7", w=2, ylim=ylim)

  xs = [1, 2, 4, 8, 16, 32, 64]
  p2 = plot(xs, ed["m7_icmiter2_npert1"], label="1", w=2, ylim=ylim)
  plot!(xs, ed["m7_icmiter2_npert2"], label="2", w=2, ylim=ylim)
  plot!(xs, ed["m7_icmiter2_npert4"], label="4", w=2, ylim=ylim)
  plot!(xs, ed["m7_icmiter2_npert7"], label="7", w=2, ylim=ylim)

  xs = [0, 4, 8, 16, 32, 64]
  p3 = plot(xs, ed["m7_icmiter4_npert1"], label="1", w=2, ylim=ylim)
  plot!(xs, ed["m7_icmiter4_npert2"], label="2", w=2, ylim=ylim)
  plot!(xs, ed["m7_icmiter4_npert4"], label="4", w=2, ylim=ylim)
  plot!(xs, ed["m7_icmiter4_npert7"], label="7", w=2, ylim=ylim)

  xs = [0, 8, 16, 32, 64]
  p4 = plot(xs, ed["m7_icmiter8_npert1"], label="1", w=2, ylim=ylim)
  plot!(xs, ed["m7_icmiter8_npert2"], label="2", w=2, ylim=ylim)
  plot!(xs, ed["m7_icmiter8_npert4"], label="4", w=2, ylim=ylim)
  plot!(xs, ed["m7_icmiter8_npert7"], label="7", w=2, ylim=ylim)

  xs = [0, 1, 2, 4, 8, 16, 32, 64]
  ylim = (21500, 29000)
  p5 = plot(xs, ed["m15_icmiter1_npert1"], label="1", w=2, ylim=ylim)
  plot!(xs, ed["m15_icmiter1_npert2"], label="2", w=2, ylim=ylim)
  plot!(xs, ed["m15_icmiter1_npert4"], label="4", w=2, ylim=ylim)
  plot!(xs, ed["m15_icmiter1_npert8"], label="8", w=2, ylim=ylim)
  plot!(xs, ed["m15_icmiter1_npert15"], label="15", w=2, ylim=ylim)

  xs = [1, 2, 4, 8, 16, 32, 64]
  p6 = plot(xs, ed["m15_icmiter2_npert1"], label="1", w=2, ylim=ylim)
  plot!(xs, ed["m15_icmiter2_npert2"], label="2", w=2, ylim=ylim)
  plot!(xs, ed["m15_icmiter2_npert4"], label="4", w=2, ylim=ylim)
  plot!(xs, ed["m15_icmiter2_npert8"], label="8", w=2, ylim=ylim)
  plot!(xs, ed["m15_icmiter2_npert15"], label="15", w=2, ylim=ylim)

  xs = [0, 4, 8, 16, 32, 64]
  p7 = plot(xs, ed["m15_icmiter4_npert1"], label="1", w=2, ylim=ylim)
  plot!(xs, ed["m15_icmiter4_npert2"], label="2", w=2, ylim=ylim)
  plot!(xs, ed["m15_icmiter4_npert4"], label="4", w=2, ylim=ylim)
  plot!(xs, ed["m15_icmiter4_npert8"], label="8", w=2, ylim=ylim)
  plot!(xs, ed["m15_icmiter4_npert15"], label="15", w=2, ylim=ylim)

  xs = [0, 8, 16, 32, 64]
  p8 = plot(xs, ed["m15_icmiter8_npert1"], label="1", w=2, ylim=ylim)
  plot!(xs, ed["m15_icmiter8_npert2"], label="2", w=2, ylim=ylim)
  plot!(xs, ed["m15_icmiter8_npert4"], label="4", w=2, ylim=ylim)
  plot!(xs, ed["m15_icmiter8_npert8"], label="8", w=2, ylim=ylim)
  plot!(xs, ed["m15_icmiter8_npert15"], label="15", w=2, ylim=ylim)

  # plot(p1, p2, p3, p4, layout=(1,4), yaxis=("Quantization error"), ylim=(82000, 84000))
  plot(p1, p2, p3, p4, p5, p6, p7, p8, layout=(2,4), yaxis=("Quantization error"))

  # for m = [7, 15], icmiter = [1, 2, 4, 8]
  #   nperts = ifelse(m==7, [1, 2, 4, 7], [1, 2, 4, 8, 15])
  #
  #   for npert in nperts
  #
  #     # We want the total number of icmiters to be 64
  #     max_ilsiters = 64 / icmiter
  #     ilsiters = [1, 2, 4, 8, 16, 32, 64]
  #     ilsiters_idx = find(max_ilsiters .== ilsiters )[1]
  #     ilsiters = ilsiters[1:ilsiters_idx]
  #     @show m, icmiter, nperts, ilsiters
  #
  #     ed["m$(m)_icmiter$(icmiter)_npert$(npert)"] = objs
  #
  #     @show length(objs)
  #   end
  # end

end


# experiment_data = run_experiment()
make_figure(experiment_data)
