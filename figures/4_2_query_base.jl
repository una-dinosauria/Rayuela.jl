
using PyPlot, HDF5


function load_recall_curves(ntrials, toplot, dataset_name, RPATH, fname, ilsit)
  recalls = zeros(ntrials, 1000)
  for trial = 1:ntrials
    fname = joinpath(RPATH, dataset_name, fname)
    # @show fname
    recalls[trial, :] = h5read(fname, "$trial/recall_$ilsit")
  end
  meanrecall, stdrecall = mean(recalls,1)[toplot], std(recalls,1)[toplot]
end


function load_recall_curves(ntrials, toplot, dataset_name, RPATH, fname)
  recalls = zeros(ntrials, 1000)
  for trial = 1:ntrials
    fname = joinpath(RPATH, dataset_name, fname)
    # @show fname
    recalls[trial, :] = h5read(fname, "$trial/recall")
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

  running_times_mnist =   # encoding + codebook
    Dict("sr_64"       => 7.26 + 0.23, # fast codebook
         "sr_128"      => 17.1 + 1.38,
         "lsq_64"      => 7.26 + 18.1, # no fast codebook
         "lsq_128"     => 17.1 + 63.82,
         "sr_64_gpu"   => 1.15 + 0.23, # gpu, fast codebook
         "sr_128_gpu"  => 2.50 + 1.38,
         "lsq_64_gpu"  => 1.15 + 18.1, # gpu, no fast codebook
         "lsq_128_gpu" => 2.50 + 63.82,
         "chainq_sr_64"      => 9.89,  # fast codebook
         "chainq_sr_128"     => 20.90, #
         "chainq_64"         => 12.76, # no fast codebook no gpu
         "chainq_128"        => 23.64, #
         "chainq_sr_64_gpu"  => 5.42,  # gpu, fast codebook
         "chainq_sr_128_gpu" => 7.86,  #
         "chainq_64_gpu"     => 8.22,  # gpu, no fast codebook
         "chainq_128_gpu"    => 12.94, #
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

   running_times_labelme = # encoding + codebook
     Dict("sr_64"       => 1.96 + 0.16, # fast codebook
          "sr_128"      => 5.32 + 1.14,
          "lsq_64"      => 1.96 + 4.27, # no fast codebook
          "lsq_128"     => 5.32 + 10.26,
          "sr_64_gpu"   => 0.36 + 0.16, # gpu, fast codebook
          "sr_128_gpu"  => 1.15 + 1.14,
          "lsq_64_gpu"  => 0.36 + 4.27, # gpu, no fast codebook
          "lsq_128_gpu" => 1.15 + 10.26,
          "chainq_sr_64"      => 2.92, # fast codebook
          "chainq_sr_128"     => 6.54, #
          "chainq_64"         => 3.94, # no fast codebook no gpu
          "chainq_128"        => 7.07, #
          "chainq_sr_64_gpu"  => 1.75, # gpu, fast codebook
          "chainq_sr_128_gpu" => 3.58, #
          "chainq_64_gpu"     => 2.78, # gpu, no fast codebook
          "chainq_128_gpu"    => 4.14, #
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

  subplotidx = 1;
  for dataset_name = dnames

    if dataset_name == "mnist"
      times_dict = running_times_mnist
    else
      times_dict = running_times_labelme
    end


    ax = subplot(1,length(dnames),subplotidx)
    for midx = 1:length( ms )

      ax[:set_prop_cycle](nothing)  # Reset colour cycle
      m = ms[ midx ]

      # Load srd
      for baseline = ["lsq", "src", "srd"]
        recalls, stdevs, total_times = [], [], []
        for niter in niters
          meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "$(baseline)_m$(m)_it$niter.h5")
          println("$dataset_name $(baseline) m=$m"); print_recalls(meanrecall, stdrecall)

          # Compute the time
          opq_time = times_dict["opq_$(8(m+1))"]
          chainq_time = times_dict["chainq_$(8(m+1))"]
          init_time = (opq_time + chainq_time) * niter

          push!(stdevs, stdrecall[1])
          push!(recalls, meanrecall[1])
          if baseline == "lsq"
            push!(total_times, init_time + (times_dict["lsq_$(8(m+1))"] * niter))
          else
            push!(total_times, init_time + (times_dict["sr_$(8(m+1))"] * niter))
          end
        end

        lstart = baseline == "lsq" ? "LSQ" : baseline == "src" ? "SR-C" : "SR-D"
        @show baseline, lstart
        errorbar(total_times, recalls, yerr=stdevs, label="$lstart $((m+1)*8) bits", fmt = m == 7 ? "-x" : "", lw=linew)
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


      # Load rvq and ervq
      for baseline = ["ervq", "rvq"]
        recalls, stdevs, total_times = [], [], []
        for niter in niters
          meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "$(baseline)_m$(m)_it$(niter).h5")
          # println("$dataset_name $baseline m=$m"); print_recalls(meanrecall, stdrecall)

          push!(stdevs, stdrecall[1])
          push!(recalls, meanrecall[1])

          init_time = baseline == "rvq" ? 0 : times_dict["rvq_$(8(m+1))"] * niter
          push!(total_times, init_time + (times_dict["$(baseline)_$(8(m+1))"] * niter))
        end
        errorbar(total_times, recalls, yerr=stdevs, label="$(uppercase(baseline)) $((m+1)*8) bits", fmt = m == 7 ? "-x" : "", lw=linew)
      end


      # Load pq and opq
      for baseline = ["opq", "pq"]
        recalls, stdevs, total_times = [], [], []
        for niter in niters
          meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "$(baseline)_m$(m+1)_it$(niter).h5")
          # println("$dataset_name $baseline m=$m"); print_recalls(meanrecall, stdrecall)

          push!(stdevs, stdrecall[1])
          push!(recalls, meanrecall[1])
          push!(total_times, times_dict["$(baseline)_$(8(m+1))"] * niter)
        end
        # plot(total_times, recalls, label="$(uppercase(baseline)) $((m+1)*8) bits", m == 7 ? "--x" : "", lw=linew)
        errorbar(total_times, recalls, yerr=stdevs, label="$(uppercase(baseline)) $((m+1)*8) bits", fmt = m == 7 ? "-x" : "", lw=linew)
      end

      grid(true)
      title(get_title(dataset_name), size=sz)
      ax[:set_xscale]("log")

      # ymin = dataset_name == "sift1m" ? 0.2 : 0
      # ymin = dataset_name == "mnist" ? 0.3 : ymin
      # ymin = dataset_name == "labelme" ? 0.2 : ymin
      # ax[:set_ylim]([ymin, 1])
      # ax[:set_xlim]([1, dataset_name == "mnist" ? 10_000 : 1_000])
      ax[:set_xlim]([1, 10_000])

      ylabel("Recall@1", size=sz)
      xlabel("Seconds", size=sz)
      # legend(loc="lower right", fontsize=sz-2, ncol=2)
      legend(loc="best", fontsize=sz-2, ncol=2)

    end
    subplotidx += 1
  end

  PyPlot.tight_layout()
  sfile = "/home/julieta/Desktop/recall_$(dnames[1])_$(dnames[2]).pdf"
  savefig( sfile )
end

RPATH = "/home/julieta/.julia/v0.6/Rayuela/results/"
# make_plots(RPATH, ["sift1m", "convnet1m"])
make_plots(RPATH, ["mnist", "labelme"])
