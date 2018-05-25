
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


function make_plots_query_base(RPATH, dnames)
  ms      = [7, 15]

  # Dictionaries with time per iteration. In seconds
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

  sz = 15
  linew = 2.0
  toplot = 1:1000

  # Set the overall size here
  fig = figure("recall_plot", figsize=((5+0.6)*length(dnames), 9))
  ntrials = 10
  niters  = [25, 50, 100]

  subplotidx = 1;

  for midx = 1:length( ms )
    m = ms[midx]


    for dataset_name = dnames

      if dataset_name == "mnist"
        times_dict = running_times_mnist
      else
        times_dict = running_times_labelme
      end

      ax = subplot(2, length(dnames), subplotidx)

      ax[:set_prop_cycle](nothing)  # Reset colour cycle

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
        errorbar(total_times, recalls, yerr=stdevs, label="$lstart", fmt = "-", lw=linew)
      end

      ax[:set_prop_cycle](nothing)
      # Load GPU
      for baseline = ["lsq", "src", "srd"]
        recalls, stdevs, total_times = [], [], []
        for niter in niters
          meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "$(baseline)_m$(m)_it$niter.h5")
          println("$dataset_name $(baseline) m=$m"); print_recalls(meanrecall, stdrecall)

          # Compute the time
          opq_time = times_dict["opq_$(8(m+1))"]
          chainq_time = times_dict["chainq_sr_$(8(m+1))_gpu"]
          init_time = (opq_time + chainq_time) * niter

          push!(stdevs, stdrecall[1])
          push!(recalls, meanrecall[1])
          if baseline == "lsq"
            push!(total_times, init_time + (times_dict["lsq_$(8(m+1))_gpu"] * niter))
          else
            push!(total_times, init_time + (times_dict["sr_$(8(m+1))_gpu"] * niter))
          end
        end

        lstart = baseline == "lsq" ? "LSQ" : baseline == "src" ? "SR-C" : "SR-D"
        @show baseline, lstart
        errorbar(total_times, recalls, yerr=stdevs, label="$lstart GPU", fmt="--", lw=linew)
      end

      # CQ
      bpath = joinpath("/home/julieta/Desktop/CQ/build/results/", lowercase(dataset_name) * "_$(m+1)", "trial_3")
      recall = h5read(joinpath(bpath, "recall.h5"), "recall")
      println("$dataset_name cq m=$(m+1)"); print_recalls(recall, zeros(size(recall)))
      plot(times_dict["cq_$(8(m+1))"], recall[1], label="CQ", "o", lw=linew)

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
        errorbar(total_times, recalls, yerr=stdevs, label="$(uppercase(baseline))", fmt = "-", lw=linew)
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
        errorbar(total_times, recalls, yerr=stdevs, label="$(uppercase(baseline))", fmt = "-", lw=linew)
      end

      grid(true)
      title(get_title(dataset_name) * " $((m+1)*8) bits", size=sz)
      ax[:set_xscale]("log")

      # ymin = dataset_name == "sift1m" ? 0.2 : 0
      # ymin = dataset_name == "mnist" ? 0.3 : ymin
      # ymin = dataset_name == "labelme" ? 0.2 : ymin
      ax[:set_ylim](m == 7 ? [0, 0.5] : [0.2, 0.7])
      # ax[:set_xlim]([1, dataset_name == "mnist" ? 10_000 : 1_000])
      ax[:set_xlim]([1, 100_000])

      ylabel("Recall@1", size=sz)
      xlabel("Seconds", size=sz)
      if subplotidx == 1
        legend(loc="best", fontsize=sz-2, ncol=3)
      end
      # legend(loc="lower right", fontsize=sz-2, ncol=2)

      subplotidx += 1
    end
  end


  function make_plots_query_base(RPATH, dnames)
    ms      = [7, 15]

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

    sz = 15
    linew = 2.0
    toplot = 1:1000

    # Set the overall size here
    fig = figure("recall_plot", figsize=((5+0.6)*length(dnames), 9))
    ntrials = 10
    niters  = [25, 50, 100]

    subplotidx = 1;

    for midx = 1:length( ms )
      m = ms[midx]


      for dataset_name = dnames

        if dataset_name == "mnist"
          times_dict = running_times_mnist
        else
          times_dict = running_times_labelme
        end

        ax = subplot(2, length(dnames), subplotidx)

        ax[:set_prop_cycle](nothing)  # Reset colour cycle

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
          errorbar(total_times, recalls, yerr=stdevs, label="$lstart", fmt = "-", lw=linew)
        end

        ax[:set_prop_cycle](nothing)
        # Load GPU
        for baseline = ["lsq", "src", "srd"]
          recalls, stdevs, total_times = [], [], []
          for niter in niters
            meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "$(baseline)_m$(m)_it$niter.h5")
            println("$dataset_name $(baseline) m=$m"); print_recalls(meanrecall, stdrecall)

            # Compute the time
            opq_time = times_dict["opq_$(8(m+1))"]
            chainq_time = times_dict["chainq_sr_$(8(m+1))_gpu"]
            init_time = (opq_time + chainq_time) * niter

            push!(stdevs, stdrecall[1])
            push!(recalls, meanrecall[1])
            if baseline == "lsq"
              push!(total_times, init_time + (times_dict["lsq_$(8(m+1))_gpu"] * niter))
            else
              push!(total_times, init_time + (times_dict["sr_$(8(m+1))_gpu"] * niter))
            end
          end

          lstart = baseline == "lsq" ? "LSQ" : baseline == "src" ? "SR-C" : "SR-D"
          @show baseline, lstart
          errorbar(total_times, recalls, yerr=stdevs, label="$lstart GPU", fmt="--", lw=linew)
        end

        # CQ
        bpath = joinpath("/home/julieta/Desktop/CQ/build/results/", lowercase(dataset_name) * "_$(m+1)", "trial_3")
        recall = h5read(joinpath(bpath, "recall.h5"), "recall")
        println("$dataset_name cq m=$(m+1)"); print_recalls(recall, zeros(size(recall)))
        plot(times_dict["cq_$(8(m+1))"], recall[1], label="CQ", "o", lw=linew)

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
          errorbar(total_times, recalls, yerr=stdevs, label="$(uppercase(baseline))", fmt = "-", lw=linew)
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
          errorbar(total_times, recalls, yerr=stdevs, label="$(uppercase(baseline))", fmt = "-", lw=linew)
        end

        grid(true)
        title(get_title(dataset_name) * " $((m+1)*8) bits", size=sz)
        ax[:set_xscale]("log")

        # ymin = dataset_name == "sift1m" ? 0.2 : 0
        # ymin = dataset_name == "mnist" ? 0.3 : ymin
        # ymin = dataset_name == "labelme" ? 0.2 : ymin
        ax[:set_ylim](m == 7 ? [0, 0.5] : [0.2, 0.7])
        # ax[:set_xlim]([1, dataset_name == "mnist" ? 10_000 : 1_000])
        ax[:set_xlim]([1, 100_000])

        ylabel("Recall@1", size=sz)
        xlabel("Seconds", size=sz)
        if subplotidx == 1
          legend(loc="best", fontsize=sz-2, ncol=3)
        end
        # legend(loc="lower right", fontsize=sz-2, ncol=2)

        subplotidx += 1
      end
    end


    # Put a legend below current axis
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #         fancybox=True, shadow=True, ncol=5)
    # legend(loc="upper center", fontsize=sz-2, bbox_to_anchor=(1.0, 1.0), ncol=1)

    PyPlot.tight_layout()
    sfile = "/home/julieta/Desktop/recall_vs_time_$(dnames[1])_$(dnames[2]).pdf"
    savefig( sfile )
end

function make_plots_train_query_base(RPATH, dnames)
  ms = [7, 15]

  # Dictionaries with time per iteration. In seconds
  running_times_sift_train = # encoding + codebook
    Dict("sr_64"       => 0, # fast codebook
         "sr_128"      => 0,
         "lsq_64"      => 0, # no fast codebook
         "lsq_128"     => 0,
         "sr_64_gpu"   => 0, # gpu, fast codebook
         "sr_128_gpu"  => 0,
         "lsq_64_gpu"  => 0, # gpu, no fast codebook
         "lsq_128_gpu" => 0,
         "chainq_sr_64"      => 0, # fast codebook
         "chainq_sr_128"     => 0, #
         "chainq_64"         => 0, # no fast codebook no gpu
         "chainq_128"        => 0, #
         "chainq_sr_64_gpu"  => 0, # gpu, fast codebook
         "chainq_sr_128_gpu" => 0, #
         "chainq_64_gpu"     => 0, # gpu, no fast codebook
         "chainq_128_gpu"    => 0, #
         "cq_64"       => ,
         "cq_128"      => 13*3600 + 24*60, # 13:24:28.23 -- train on base; 2:29:34.61 -- train on learn
         "ervq_64"     => 5.02,
         "ervq_128"    => 25.28,
         "rvq_64"      => 48.3 / 10, # training
         "rvq_128"     => 98.7 / 10,
         "opq_64"      => 9.64 / 10,
         "opq_128"     => 14.4 / 10,
         "pq_64"       => 7.30 / 10,
         "pq_128"      => 9.97 / 10)

   running_times_sift_encode = # encoding + codebook
     Dict("sr_64"       => 0, # fast codebook
          "sr_128"      => 0,
          "lsq_64"      => 0, # no fast codebook
          "lsq_128"     => 0,
          "sr_64_gpu"   => 0, # gpu, fast codebook
          "sr_128_gpu"  => 0,
          "lsq_64_gpu"  => 0, # gpu, no fast codebook
          "lsq_128_gpu" => 0,
          "chainq_sr_64"      => 0, # fast codebook
          "chainq_sr_128"     => 0, #
          "chainq_64"         => 0, # no fast codebook no gpu
          "chainq_128"        => 0, #
          "chainq_sr_64_gpu"  => 0, # gpu, fast codebook
          "chainq_sr_128_gpu" => 0, #
          "chainq_64_gpu"     => 0, # gpu, no fast codebook
          "chainq_128_gpu"    => 0, #
          "cq_64"       => 0,
          "cq_128"      => 0,
          "ervq_64"     => 39.19, # encoding
          "ervq_128"    => 150.7,
          "rvq_64"      => 39.19, # encoding
          "rvq_128"     => 150.7,
          "opq_64"      => 4.96,
          "opq_128"     => 8.86,
          "pq_64"       => 4.65,
          "pq_128"      => 8.39)

  sz = 15
  linew = 2.0
  toplot = 1:1000

  # Set the overall size here
  fig = figure("recall_plot", figsize=((5+0.6)*length(dnames), 9))
  ntrials = 10
  niters  = [25, 50, 100]

  subplotidx = 1;

  for midx = 1:length( ms )
    m = ms[midx]


    for dataset_name = dnames

      if dataset_name == "mnist"
        times_dict = running_times_mnist
      else
        times_dict = running_times_labelme
      end

      ax = subplot(2, length(dnames), subplotidx)

      ax[:set_prop_cycle](nothing)  # Reset colour cycle

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
        errorbar(total_times, recalls, yerr=stdevs, label="$lstart", fmt = "-", lw=linew)
      end

      ax[:set_prop_cycle](nothing)
      # Load GPU
      for baseline = ["lsq", "src", "srd"]
        recalls, stdevs, total_times = [], [], []
        for niter in niters
          meanrecall, stdrecall = load_recall_curves(ntrials, toplot, dataset_name, RPATH, "$(baseline)_m$(m)_it$niter.h5")
          println("$dataset_name $(baseline) m=$m"); print_recalls(meanrecall, stdrecall)

          # Compute the time
          opq_time = times_dict["opq_$(8(m+1))"]
          chainq_time = times_dict["chainq_sr_$(8(m+1))_gpu"]
          init_time = (opq_time + chainq_time) * niter

          push!(stdevs, stdrecall[1])
          push!(recalls, meanrecall[1])
          if baseline == "lsq"
            push!(total_times, init_time + (times_dict["lsq_$(8(m+1))_gpu"] * niter))
          else
            push!(total_times, init_time + (times_dict["sr_$(8(m+1))_gpu"] * niter))
          end
        end

        lstart = baseline == "lsq" ? "LSQ" : baseline == "src" ? "SR-C" : "SR-D"
        @show baseline, lstart
        errorbar(total_times, recalls, yerr=stdevs, label="$lstart GPU", fmt="--", lw=linew)
      end

      # CQ
      bpath = joinpath("/home/julieta/Desktop/CQ/build/results/", lowercase(dataset_name) * "_$(m+1)", "trial_3")
      recall = h5read(joinpath(bpath, "recall.h5"), "recall")
      println("$dataset_name cq m=$(m+1)"); print_recalls(recall, zeros(size(recall)))
      plot(times_dict["cq_$(8(m+1))"], recall[1], label="CQ", "o", lw=linew)

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
        errorbar(total_times, recalls, yerr=stdevs, label="$(uppercase(baseline))", fmt = "-", lw=linew)
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
        errorbar(total_times, recalls, yerr=stdevs, label="$(uppercase(baseline))", fmt = "-", lw=linew)
      end

      grid(true)
      title(get_title(dataset_name) * " $((m+1)*8) bits", size=sz)
      ax[:set_xscale]("log")

      # ymin = dataset_name == "sift1m" ? 0.2 : 0
      # ymin = dataset_name == "mnist" ? 0.3 : ymin
      # ymin = dataset_name == "labelme" ? 0.2 : ymin
      ax[:set_ylim](m == 7 ? [0, 0.5] : [0.2, 0.7])
      # ax[:set_xlim]([1, dataset_name == "mnist" ? 10_000 : 1_000])
      ax[:set_xlim]([1, 100_000])

      ylabel("Recall@1", size=sz)
      xlabel("Seconds", size=sz)
      if subplotidx == 1
        legend(loc="best", fontsize=sz-2, ncol=3)
      end
      # legend(loc="lower right", fontsize=sz-2, ncol=2)

      subplotidx += 1
    end
  end

RPATH = "/home/julieta/.julia/v0.6/Rayuela/results/"
# make_plots_query_base(RPATH, ["mnist", "labelme"])
make_plots_train_query_base(RPATH, ["sift1m", "convnet1m"])
