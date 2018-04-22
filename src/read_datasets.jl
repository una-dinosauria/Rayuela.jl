
export read_dataset

function read_dataset(
  dname::AbstractString,  # name of the dataset to load
  nvectors::Union{Integer, UnitRange},      # number of vectors to read
  V::Bool=false )         # whether to print progress

  if V println("Loading $(dname)... "); end

  # === Deep_babenko ===
  if dname == "Deep1M_babenko"

    fname = "./data/deep_babenko/deep1M_learn.fvecs"
    X     = fvecs_read(nvectors, fname)
    return X

  elseif dname == "Deep1M_babenko_groundtruth"

    fname = "./data/deep_babenko/deep1M_groundtruth.ivecs"
    X     = ivecs_read(nvectors, fname)
    return X

  elseif dname == "Deep1M_babenko_query"

    fname = "./data/deep_babenko/deep1M_queries.fvecs"
    X     = fvecs_read(nvectors, fname)
    return X

  elseif dname == "Deep1M_babenko_base"

    fname = "./data/deep_babenko/deep1M_base.fvecs"
    @show nvectors, fname
    X     = fvecs_read(nvectors, fname)
    return X

  # === GIST1M ===
  elseif dname == "GIST1M"

    fname = "./data/gist/gist_learn.fvecs";
    X     = fvecs_read(nvectors, fname)
    return X

  elseif dname == "GIST1M_query"

    fname = "./data/gist/gist_query.fvecs";
    X     = fvecs_read(nvectors, fname)
    return X

  elseif dname == "GIST1M_groundtruth"

    fname = "./data/gist/gist_groundtruth.ivecs";
    X     = ivecs_read(nvectors, fname)
    return X

  elseif dname == "GIST1M_base"

    fname = "./data/gist/gist_base.fvecs";
    X     = fvecs_read(nvectors, fname)
    return X

  # ====  SIFT1M ===
  elseif dname == "SIFT1M"

    fname = "./data/sift/sift_learn.fvecs";
    X     = fvecs_read(nvectors, fname)
    return X

  elseif dname == "SIFT1M_query"

    fname = "./data/sift/sift_query.fvecs"
    X     = fvecs_read(nvectors, fname)
    return X

  elseif dname == "SIFT1M_groundtruth"

    fname = "./data/sift/sift_groundtruth.ivecs"
    X     = ivecs_read(nvectors, fname)
    return X

  elseif dname == "SIFT1M_base"

    fname = "./data/sift/sift_base.fvecs";
    X     = fvecs_read(nvectors, fname)
    return X

  # === Convnet1M ===
  elseif dname == "Convnet1M_base"

    fname = "./data/feats/feats_m_128.mat"
    X     = h5read(fname, "feats_m_128_base");

  elseif dname == "Convnet1M"

    fname = "./data/feats/feats_m_128.mat"
    X     = h5read(fname, "feats_m_128_train");

  elseif dname == "Convnet1M_query"

    fname = "./data/feats/feats_m_128.mat"
    X     = h5read(fname, "feats_m_128_test");

  elseif dname == "Convnet1M_groundtruth"

    fname = "./data/feats/feats_m_128_gt.mat"
    X     = h5read(fname, "gt");

  # === Deep1M ===
  elseif dname == "Deep1M_base"

    fname = "./data/deep/deep.h5"
    X     = h5read(fname, "base");

  elseif dname == "Deep1M"

    fname = "./data/deep/deep.h5"
    X     = h5read(fname, "train");

  elseif dname == "Deep1M_query"

    fname = "./data/deep/deep.h5"
    X     = h5read(fname, "query");

  elseif dname == "Deep1M_groundtruth"

    fname = "./data/deep/deep.h5"
    X     = h5read(fname, "gt");
    return X

  # === Deep1B ===
  elseif dname == "Deep1B"

    fname = "./data/deep1b/learn_00"
    X     = fvecs_read(nvectors, fname)
    return X

  elseif dname == "Deep1B_base"

    # address on curiosity
    fname = "/scratch/julm/deep1b/base_all"
    #fname = "./data/deep1b/base_all"
    X     = fvecs_read(nvectors, fname)
    return X

  # === SIFT1B
  elseif dname == "SIFT1B_query"

    fname = "/hdd/sift1b/bigann_query.bvecs"
    X     = bvecs_read(nvectors,fname)
    X = convert(Matrix{Float32},X)
    return X

  elseif dname == "SIFT1B_base"

    # fname = "./data/sift1b/bigann_base.bvecs"
    fname = "/hdd/sift1b/bigann_base.bvecs"
    # fname = "/scratch/zakhmi/bigann_base.bvecs" # curiosity
    X     = bvecs_read(nvectors,fname)
    X = convert(Matrix{Float32},X)
    return X

  elseif dname == "SIFT1B" || dname == "SIFT10M"

    fname = "/hdd/sift1b/bigann_learn.bvecs"
    X     = bvecs_read(nvectors,fname)
    X = convert(Matrix{Float32},X)
    return X

  elseif dname == "SIFT1B_groundtruth"

    fname = "/hdd/sift1b/gnd/idx_1000M.ivecs"
    X     = ivecs_read(nvectors, fname)
    return X

  elseif dname == "SIFT1B_groundtruth_10M"

    fname = "/hdd/sift1b/gnd/idx_10M.ivecs"
    X     = ivecs_read(nvectors, fname)
    return X

  elseif dname == "SIFT1B_groundtruth_1M"

    fname = "/hdd/sift1b/gnd/idx_1M.ivecs"
    X     = ivecs_read(nvectors, fname)
    return X

  # === MNIST
  elseif dname == "MNIST_query"

    fname = "./data/mnist/mnist.h5";
    X     = h5read(fname, "test");
    X     = reshape( X, 28 * 28, 10000 );
    X     = convert(Matrix{Float32}, X);

  elseif dname == "MNIST" || dname == "MNIST_base"

    fname = "./data/mnist/mnist.h5";
    X     = h5read(fname, "train");
    X     = reshape( X, 28 * 28, 60000 );
    X     = convert(Matrix{Float32}, X);

  elseif dname == "MNIST_groundtruth"

    fname = "./data/mnist/mnist.h5";
    X     = h5read(fname, "gt");

  # === labelme22K
  elseif dname == "labelme_query"

    fname = "./data/labelme/label.h5";
    X     = h5read(fname, "query");
    X     = convert( Matrix{Float32}, X );

  elseif dname == "labelme" || dname == "labelme_base"

    fname = "./data/labelme/label.h5";
    X     = h5read(fname, "train");
    X     = convert( Matrix{Float32}, X );

  elseif dname == "labelme_groundtruth"

    fname = "./data/labelme/label.h5";
    X     = h5read(fname, "gt");

  else

    error("Dataset $(dname) unknown")

  end

  _, n = size( X );

  if typeof( nvectors ) <: Integer
    if nvectors <= n
      X = X[:, 1:nvectors];
    else
      error("Asked to read $nvectors vectors, but the datasets has only $n vectors.");
    end
  else
    X = X[:, nvectors];
  end

  return X
end
