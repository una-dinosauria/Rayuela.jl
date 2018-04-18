
using PyPlot, HDF5


function load_recall_curves(ntrials, toplot, dataset_name, RPATH, fname)
  recalls = zeros(ntrials, 1000)
  for trial = 1:ntrials
    recalls[trial, :] = h5read(joinpath(RPATH, dataset_name, fname), "$trial/recall")
  end
  meanrecall, stdrecall = mean(recalls,1)[toplot], std(recalls,1)[toplot]
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


function make_plots(RPATH, dnames)
  ms      = [15, 7]
  ilsit   = [32]

  sz = 13
  linew = 2
  toplot = 1:1000

  # Set the overall size here
  fig = figure("recall_plot", figsize=( (5+0.6)*length(dnames), 5))
  ntrials = 10

  subplotidx = 1;
  for dataset_name = dnames
    ax = subplot(1,length(dnames),subplotidx)
    for midx = 1:length( ms )

      ax[:set_prop_cycle](nothing)  # Reset colour cycle
      m = ms[ midx ]

      # Load lsq
      for i=length(ilsit):-1:1
        meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "lsq_m$(m)_it100.h5")
        println("$dataset_name lsq m=$m"); print_recalls(meanrecall, stdrecall)
        plot(toplot, meanrecall[toplot], label="LSQ-$(ilsit[i]) $((m+1)*8) bits", m == 7 ? "--" : "", lw=linew)
      end

      if dataset_name == "convnet1m"
        # Skip the color for consistency
        plot([], [])
      else
        bpath = joinpath("/home/julieta/Desktop/CQ/build/results/", lowercase(dataset_name) * "_$(m+1)", "trial_3")
        recall = h5read(joinpath(bpath, "recall.h5"), "recall")
        println("$dataset_name cq m=$(m+1)"); print_recalls(recall, zeros(size(recall)))
        plot(toplot, recall[toplot], label="CQ $((m+1)*8) bits", m == 7 ? "--" : "", lw=linew)
      end

      # Load rvq and ervq
      for baseline = ["ervq", "rvq"]
        meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "$(baseline)_m$(m)_it100.h5")
        println("$dataset_name $baseline m=$m"); print_recalls(meanrecall, stdrecall)
        plot(toplot, meanrecall[toplot], label="$(uppercase(baseline)) $((m+1)*8) bits", m == 7 ? "--" : "", lw=linew)
      end

      # Load pq and opq
      for baseline = ["opq", "pq"]
        meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "$(baseline)_m$(m+1)_it100.h5")
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
  sfile = "/home/julieta/Desktop/recall_$(dnames[1])_$(dnames[2]).pdf"
  savefig( sfile )
end

function make_sparse_plot(RPATH)

  ilsit = 16
  sz = 13
  linew = 2
  toplot = 1:1000

  # Set the overall size here
  fig = figure("recall_plot", figsize=(5 + 0.6, 5))
  ntrials = 10

  dataset_name = "sift1m"
  ax = subplot(1, 1, 1)

  ax[:set_prop_cycle](nothing)  # Reset colour cycle
  m = 7

  # Load lsq
  # meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "SIFT1M_sc2.h5")
  recalls = zeros(ntrials, 1000)
  for trial = 1:ntrials
    recalls[trial, :] = h5read(joinpath(RPATH, dataset_name, "SIFT1M_sc2.h5"), "trial$trial/recall_$ilsit")
  end
  meanrecall, stdrecall = mean(recalls,1)[toplot], std(recalls,1)[toplot]
  println("$dataset_name slsq2-$(ilsit) m=$m"); print_recalls(meanrecall, stdrecall)
  plot(toplot, meanrecall[toplot], label="SLSQ2-$(ilsit) $((m+1)*8) bits", "-", lw=linew)

  # meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "SIFT1M_sc1.h5")
  for trial = 1:ntrials
    recalls[trial, :] = h5read(joinpath(RPATH, dataset_name, "SIFT1M_sc1.h5"), "trial$trial/recall_$ilsit")
  end
  meanrecall, stdrecall = mean(recalls,1)[toplot], std(recalls,1)[toplot]
  println("$dataset_name slsq1-$(ilsit) m=$m"); print_recalls(meanrecall, stdrecall)
  plot(toplot, meanrecall[toplot], label="SLSQ1-$(ilsit) $((m+1)*8) bits", "-", lw=linew)

  # plot([], [])
  # plot([], [])
  # plot([], [])
  ax[:set_prop_cycle](nothing)  # Reset colour cycle

  # Load pq and opq
  for baseline = ["opq", "pq"]
    meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "$(baseline)_m$(m+1)_it100.h5")
    println("$dataset_name $baseline m=$m"); print_recalls(meanrecall, stdrecall)
    plot(toplot, meanrecall[toplot], label="$(uppercase(baseline)) $((m+1)*8) bits", "--", lw=linew)
  end

  grid(true)
  title(get_title(dataset_name), size=sz)
  ax[:set_xscale]("log")

  ax[:set_ylim]([0.2, 1])
  ax[:set_xlim]([1, 1000])

  ylabel("Recall@N", size=sz)
  xlabel("N", size=sz)
  legend(loc="lower right", fontsize=sz-2, ncol=1)

  PyPlot.tight_layout()
  sfile = "/home/julieta/Desktop/recall_sparse.pdf"
  savefig( sfile )
end

function make_sift_table(RPATH)
  ntrials = 10
  toplot = collect(1:1000)
  dataset_name = "sift1m"
  ms = [7, 15]

  # Load pq and opq
  for baseline = ["pq", "opq"]
    print("$(uppercase(baseline)) ")
    for m = ms
      meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "$(baseline)_m$(m+1)_it100.h5")
      js = m == 7 ? [1, 10, 100] : [1, 2, 5]
      for j in js
        @printf("& \$%.2f \\pm %.2f\$ ", 100*meanrecall[j], 100 * stdrecall[j])
      end
    end
    println("\\\\")
  end

  # Load pq and opq
  for baseline = ["rvq", "ervq"]
    print("$(uppercase(baseline)) ")
    for m = ms
      meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "$(baseline)_m$(m)_it100.h5")
      js = m == 7 ? [1, 10, 100] : [1, 2, 5]
      for j in js
        @printf("& \$%.2f \\pm %.2f\$ ", 100*meanrecall[j], 100 * stdrecall[j])
      end
    end
    println("\\\\")
  end

  # Load cq
  print("CQ ")
  for m = ms
    bpath = joinpath("/home/julieta/Desktop/CQ/build/results/", lowercase(dataset_name) * "_$(m+1)", "trial_3")
    meanrecall = h5read(joinpath(bpath, "recall.h5"), "recall")
    stdrecall = zeros(size(meanrecall))
    js = m == 7 ? [1, 10, 100] : [1, 2, 5]
    for j in js
      # @printf("& \$%.2f \\pm %.2f\$ ", 100*meanrecall[j], 100 * stdrecall[j])
      @printf("& \$%.2f\$ ", 100*meanrecall[j])
    end
  end
  println("\\\\")

  # Load pq and opq
  print("LSQ-32 ")
  for m = ms
    meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "lsq_m$(m)_it100.h5")
    js = m == 7 ? [1, 10, 100] : [1, 2, 5]
    for j in js
      @printf("& \$\\mathbf{%.2f} \\pm %.2f\$ ", 100*meanrecall[j], 100 * stdrecall[j])
    end
  end
  println("\\\\")
end


function make_mnist_labelme_table(RPATH)
  ntrials = 10
  toplot = collect(1:1000)
  dnames = ["mnist", "labelme"]
  m = 7

  # Load pq and opq
  for baseline = ["pq", "opq"]
    print("$(uppercase(baseline)) ")
    for dataset_name = dnames
      meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "$(baseline)_m$(m+1)_it100.h5")
      for j in [1, 2, 5]
        @printf("& \$%.2f \\pm %.2f\$ ", 100*meanrecall[j], 100 * stdrecall[j])
      end
    end
    println("\\\\")
  end
  println("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")

  # Load pq and opq
  for baseline = ["rvq", "ervq"]
    print("$(uppercase(baseline)) ")
    for dataset_name = dnames
      meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "$(baseline)_m$(m)_it100.h5")
      for j in [1, 2, 5]
        @printf("& \$%.2f \\pm %.2f\$ ", 100*meanrecall[j], 100 * stdrecall[j])
      end
    end
    println("\\\\")
  end

  # Load cq
  print("CQ ")
  for dataset_name = dnames
    bpath = joinpath("/home/julieta/Desktop/CQ/build/results/", lowercase(dataset_name) * "_$(m+1)", "trial_3")
    meanrecall = h5read(joinpath(bpath, "recall.h5"), "recall")
    stdrecall = zeros(size(meanrecall))
    for j in [1, 2, 5]
      @printf("& \$%.2f\$ ", 100*meanrecall[j])
    end
  end
  println("\\\\")

  # Load pq and opq
  print("LSQ-8 ")
  for dataset_name = dnames
    meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "lsq_m$(m)_it100.h5")
    for j in [1, 2, 5]
      @printf("& \$\\mathbf{%.2f} \\pm %.2f\$ ", 100*meanrecall[j], 100 * stdrecall[j])
    end
  end
  println("\\\\")
end

RPATH = "/home/julieta/.julia/v0.6/Rayuela/results/"
# make_plots(RPATH, ["sift1m", "convnet1m"])
# make_plots(RPATH, ["mnist", "labelme"])

# make_sift_table(RPATH)
# make_mnist_labelme_table(RPATH)
make_sparse_plot(RPATH)
