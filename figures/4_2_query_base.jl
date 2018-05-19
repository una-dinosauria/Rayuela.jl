
using PyPlot, HDF5


function load_recall_curves(ntrials, toplot, dataset_name, RPATH, fname, ilsit)
  recalls = zeros(ntrials, 1000)
  for trial = 1:ntrials
    recalls[trial, :] = h5read(joinpath(RPATH, dataset_name, fname), "$trial/recall_$ilsit")
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
  @printf("r@%d %.2f \\pm %.2f\n", 1, 100 * meanrecall[1], 100 * stdrecall[1]);
  println()
end


function make_plots(RPATH, dnames)
  ms      = [7]
  ilsit   = [32]

  # Dictionary with time per iteration. In seconds
  # running_times_sit1m_64 =
  #   Dict("srd_gpu"  => (1.12 + 1.4) + (1.17 + 5.6), # (initialization, training)
  #        "lsq_gpu"  => (1.12 + 1.4) + (9.6  + 5.6), # (initialization, training)
  #        "cq"       => 42 * 60, # Training on the training set
  #        "ervq"     => (23.54 / 25) + 5.172 + (13.88), # (Init) + training + (encoding)
  #        "rvq"      => (23.54 / 25) + (13.88), # training + encoding
  #        "opq"      => 0.8 + 4.94,        # training + encoding
  #        "pq"       => 12.02 / 25 + 4.59) # training + encoding

  running_times_mnist =
    Dict("sr_64"       => 7.26 + 0.23, # encoding + codebook
         "sr_64_gpu"   => 1.15 + 0.23,
         "sr_128"      => 17.1 + 1.38,
         "sr_128_gpu"  => 2.50 + 1.38,
         "lsq_64"      => 7.26 + 18.1,
         "lsq_64_gpu"  => 1.15 + 18.1,
         "lsq_128"     => 17.1 + 63.82,
         "lsq_128_gpu" => 2.50 + 63.82,
         "chainq_64"   => 0,
         "chainq_128"  => 0,
         "cq_64"       => 14692, # ~4 hours for 30 iterations
         "cq_128"      => 55051, # ~15 hours for 30 iterations
         "ervq_64"     => 13.34,
         "ervq_128"    => 45.30,
         "rvq_64"      => 3.281,
         "rvq_128"     => 6.395,
         "opq_64"      => 3.07,
         "opq_128"     => 2.88,
         "pq_64"       => 9.81  / 10,
         "pq_128"      => 11.89 / 10)

   running_times_labelme =
     Dict("sr_64"       => 1.96 + 0.16, # encoding + codebook
          "sr_64_gpu"   => 0.36 + 0.16,
          "sr_128"      => 5.32 + 1.14,
          "sr_128_gpu"  => 1.15 + 1.14,
          "lsq_64"      => 1.96 + 4.27,
          "lsq_64_gpu"  => 0.36 + 4.27,
          "lsq_128"     => 5.32 + 10.26,
          "lsq_128_gpu" => 1.15 + 10.26,
          "chainq_sr_64"      => 0, # total
          "chainq_sr_64_gpu"  => 1.65, # total
          "chainq_sr_128"     => 0, # total
          "chainq_sr_128_gpu" => 3.24, # total
          "chainq_64"         => 0, # total
          "chainq_64_gpu"     => 1.65, # total
          "chainq_128"        => 0, # total
          "chainq_128_gpu"    => 3.24, # total
          "cq_64"       => 7100, # ~2 hours for 30 iterations
          "cq_128"      => 29808, # ~8 hours for 30 iterations
          "ervq_64"     => 2.99,
          "ervq_128"    => 10.68,
          "rvq_64"      => 0.756,
          "rvq_128"     => 1.438,
          "opq_64"      => 0.54,
          "opq_128"     => 0.60,
          "pq_64"       => 2.08 / 10,
          "pq_128"      => 2.17 / 10)

  sz = 13
  linew = 2
  toplot = 1:1000

  # Set the overall size here
  fig = figure("recall_plot", figsize=( (5+0.6)*length(dnames), 5))
  ntrials = 10

  ilsiters = [32, 64, 128]
  niters   = [25, 50, 100]

  times_dict = running_times_labelme

  subplotidx = 1;
  for dataset_name = dnames
    ax = subplot(1,length(dnames),subplotidx)
    for midx = 1:length( ms )

      ax[:set_prop_cycle](nothing)  # Reset colour cycle
      m = ms[ midx ]

      # Load srd
      for (ilsit, niter) in zip(iliters, niters)
        meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "src_m$(m)_it$niter.h5", ilsiter)
        println("$dataset_name srd m=$m"); print_recalls(meanrecall, stdrecall)

        # Compute the time
        opq_time = times_dict["opq_$(8(m+1))"] * niter
        chainq_time = times_dict["chainq_$(8(m+1))"] * niter
        total_time = opq_time + chainq_time + times_dict["sr_$(8(m+1))"] * niter

        plot(total_time, meanrecall[toplot], label="SRC-$(ilsit[i]) $((m+1)*8) bits", m == 7 ? "--" : "", lw=linew)
      end

      # # Load lsq
      # for i=length(ilsit):-1:1
      #   meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "lsq_m$(m)_it100.h5")
      #   println("$dataset_name lsq m=$m"); print_recalls(meanrecall, stdrecall)
      #   plot(toplot, meanrecall[toplot], label="LSQ-$(ilsit[i]) $((m+1)*8) bits", m == 7 ? "--" : "", lw=linew)
      # end
      #
      # if dataset_name == "convnet1m"
      #   # Skip the color for consistency
      #   plot([], [])
      # else
      #   bpath = joinpath("/home/julieta/Desktop/CQ/build/results/", lowercase(dataset_name) * "_$(m+1)", "trial_3")
      #   recall = h5read(joinpath(bpath, "recall.h5"), "recall")
      #   println("$dataset_name cq m=$(m+1)"); print_recalls(recall, zeros(size(recall)))
      #   plot(toplot, recall[toplot], label="CQ $((m+1)*8) bits", m == 7 ? "--" : "", lw=linew)
      # end
      #
      # # Load rvq and ervq
      # for baseline = ["ervq", "rvq"]
      #   meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "$(baseline)_m$(m)_it100.h5")
      #   println("$dataset_name $baseline m=$m"); print_recalls(meanrecall, stdrecall)
      #   plot(toplot, meanrecall[toplot], label="$(uppercase(baseline)) $((m+1)*8) bits", m == 7 ? "--" : "", lw=linew)
      # end
      #
      # # Load pq and opq
      # for baseline = ["opq", "pq"]
      #   meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "$(baseline)_m$(m+1)_it100.h5")
      #   println("$dataset_name $baseline m=$m"); print_recalls(meanrecall, stdrecall)
      #   plot(toplot, meanrecall[toplot], label="$(uppercase(baseline)) $((m+1)*8) bits", m == 7 ? "--" : "", lw=linew)
      # end

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

RPATH = "/home/julieta/.julia/v0.6/Rayuela/results/"
make_plots(RPATH, ["sift1m", "convnet1m"])
# make_plots(RPATH, ["mnist", "labelme"])
