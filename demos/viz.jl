using PyPlot
using HDF5
using Printf
import Statistics


function load_recall_curves(ntrials, toplot, dataset_name, RPATH, fname)
  recalls = zeros(ntrials, 1000)
  for trial = 1:ntrials
    recalls[trial, :] = h5read(joinpath(RPATH, dataset_name, fname), "$trial/recall")
  end
  meanrecall, stdrecall = Statistics.mean(recalls, dims=1)[toplot], Statistics.std(recalls, dims=1)[toplot]
end


function load_recall_curves(ntrials, toplot, dataset_name, RPATH, fname, nilsiters)
  recalls = zeros(ntrials, 1000)
  for trial = 1:ntrials
    # @show RPATH, dataset_name, fname
    recalls[trial, :] = h5read(joinpath(RPATH, dataset_name, fname), "$trial/recall_$nilsiters")
  end
  meanrecall, stdrecall = Statistics.mean(recalls, dims=1)[toplot], Statistics.std(recalls, dims=1)[toplot]
end


function get_title(dataset_name)
  if dataset_name == "sift1m"
    return "SIFT1M"
  elseif dataset_name == "convnet1m"
    return "Convnet1M"
  elseif dataset_name == "mnist"
    return "MNIST"
  elseif dataset_name == "labelme"
    return "LabelMe"
  end
end


function print_recalls(meanrecall, stdrecall)
  for j = [1, 2, 5, 10, 20, 50, 100]
    @printf("r@%d %.2f \\pm %.2f\n", j, 100 * meanrecall[j], 100 * stdrecall[j]);
  end
  println()
end


function make_plots(RPATH, dnames, sfile)
  ms      = [7]
  ilsit   = [32]

  sz = 13
  linew = 2
  toplot = 1:1000
  iterations = 25

  # Set the overall size here
  fig = figure("recall_plot", figsize=( (5+0.6)*length(dnames), 5))
  ntrials = 10

  subplotidx = 1;
  for dataset_name = dnames
    ax = subplot(1,length(dnames),subplotidx)
    for midx = 1:length( ms )

      ax[:set_prop_cycle](nothing)  # Reset colour cycle
      m = ms[ midx ]

      # Load srd
      for i=length(ilsit):-1:1
        meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "srd_m$(m)_it$(iterations).h5")
        println("$dataset_name srd m=$m"); print_recalls(meanrecall, stdrecall)
        plot(toplot, meanrecall[toplot], label="SRD-$(ilsit[i]) $((m+1)*8) bits", m == 7 ? "--" : "", lw=linew)
      end

      # Load lsq
      for i=length(ilsit):-1:1
        meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "lsq_m$(m)_it$(iterations).h5")
        println("$dataset_name lsq m=$m"); print_recalls(meanrecall, stdrecall)
        plot(toplot, meanrecall[toplot], label="LSQ-$(ilsit[i]) $((m+1)*8) bits", m == 7 ? "--" : "", lw=linew)
      end

      # if dataset_name == "convnet1m"
      #   # Skip the color for consistency
      #   plot([], [])
      # else
      #   bpath = joinpath("/home/julieta/Desktop/CQ/build/results/", lowercase(dataset_name) * "_$(m+1)", "trial_3")
      #   recall = h5read(joinpath(bpath, "recall.h5"), "recall")
      #   println("$dataset_name cq m=$(m+1)"); print_recalls(recall, zeros(size(recall)))
      #   plot(toplot, recall[toplot], label="CQ $((m+1)*8) bits", m == 7 ? "--" : "", lw=linew)
      # end

      # Load rvq and ervq
      for baseline = ["ervq", "rvq"]
        meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "$(baseline)_m$(m)_it$(iterations).h5")
        println("$dataset_name $baseline m=$m"); print_recalls(meanrecall, stdrecall)
        plot(toplot, meanrecall[toplot], label="$(uppercase(baseline)) $((m+1)*8) bits", m == 7 ? "--" : "", lw=linew)
      end

      # Load pq and opq
      for baseline = ["opq", "pq"]
        meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "$(baseline)_m$(m+1)_it$(iterations).h5")
        println("$dataset_name $baseline m=$m"); print_recalls(meanrecall, stdrecall)
        plot(toplot, meanrecall[toplot], label="$(uppercase(baseline)) $((m+1)*8) bits", m == 7 ? "--" : "", lw=linew)
      end

      grid(true)
      title(get_title(dataset_name), size=sz)
      ax[:set_xscale]("log")

      ymin = dataset_name == "sift1m" ? 0.2 : 0
      ymin = dataset_name == "mnist" ? 0.3 : ymin
      ymin = dataset_name == "labelme" ? 0.2 : ymin
      ax[:set_ylim]([ymin, 1])
      ax[:set_xlim]([1, dataset_name == "sift1m" || dataset_name == "convnet1m" ? 1000 : 100])

      ylabel("Recall@N", size=sz)
      xlabel("N", size=sz)
      legend(loc="lower right", fontsize=sz-2, ncol=2)

    end
    subplotidx += 1
  end

  PyPlot.tight_layout()
  savefig( sfile )
end


RPATH = "./results/"
dataset = "sift1m"
savename = "./results/sift_plot.pdf"
make_plots(RPATH, [dataset], savename)
