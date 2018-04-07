
using PyPlot, HDF5

ms      = [15, 7]
ilsit   = [32]

# Results path
RPATH = "/home/julieta/.julia/v0.6/Rayuela/results/"
dnames = ["sift1m", "convnet1m"]
sz = 12
linew = 2

# Set the overall size here
fig = figure("recall_plot", figsize=( (5+0.5)*length(dnames), 5))

pps    = Vector{Any}(length(ms))
ntrials = 10

function load_recall_curves(ntrials, dataset_name, RPATH, fname)
  recalls = zeros(ntrials, 1000)
  for aa = 1:ntrials
    recalls[aa, :] = h5read(joinpath(RPATH, dataset_name, fname), "$aa/recall")
  end
  meanrecall, stdrecall = mean(recalls,1)[toplot], std(recalls,1)[toplot]
end


subplotidx = 1;
for dataset_name = dnames
  ax = subplot(1,length(dnames),subplotidx)
  for midx = 1:length( ms )

    ax[:set_prop_cycle](nothing)  # Reset colour cycle
    m = ms[ midx ]

    # Load lsq
    for i=length(ilsit):-1:1
      meanrecall, stdrecall = load_recall_curves(ntrials, dataset_name, RPATH, "lsq_m$(m)_it100.h5")
      plot(toplot, meanrecall, label="LSQ-$(ilsit[i]) $((m+1)*8) bits", m == 7 ? "--" : "", lw=linew)
    end

    # Load rvq and ervq
    for baseline = ["ervq", "rvq"]
      meanrecall, stdrecall = load_recall_curves(ntrials, dataset_name, RPATH, "$(baseline)_m$(m)_it100.h5")
      plot(toplot, meanrecall[toplot], label="$(uppercase(baseline)) $((m+1)*8) bits", m == 7 ? "--" : "", lw=linew)
    end

    # Load pq and opq
    for baseline = ["opq", "pq"]
      meanrecall, stdrecall = load_recall_curves(ntrials, dataset_name, RPATH, "$(baseline)_m$(m+1)_it100.h5")
      plot(toplot, meanrecall[toplot], label="$(uppercase(baseline)) $((m+1)*8) bits", m == 7 ? "--" : "", lw=linew)
    end

    grid("on")
    title(dataset_name == "sift1m" ? "SIFT1M" : "Convnet1M", size=sz)
    ax[:set_xscale]("log")

    ax[:set_ylim]([dataset_name == "sift1m" ? 0.2 : 0, 1])
    ax[:set_xlim]([0,10.^3])

    ylabel("Recall@N", size=sz)
    xlabel("N", size=sz)
    legend(loc="lower right", fontsize=sz-2, ncol=2)

  end
  subplotidx += 1;
end

#sfile = "/home/julieta/Desktop/recall_$(dataset_name).pdf"
# sfile = "/home/julieta/Desktop/recall.pdf"
# sfile = "/ubc/cs/research/tracking-raid/julm/optimal-quantization/recall.pdf"
PyPlot.tight_layout()
# savefig( sfile );
#clf()
