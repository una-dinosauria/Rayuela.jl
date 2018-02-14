using HDF5

export read_dataset

function read_dataset(
  dname::AbstractString,  # name of the dataset to load
  nvectors::Union{Integer, UnitRange},      # number of vectors to read
  V::Bool=false )         # whether to print progress

  if V print("Loading $(dname)... "); end

  # ====  SIFT1M ===
  if dname == "SIFT1M"

    fname = "./data/sift/sift_learn.fvecs";
    X     = fvecs_read(nvectors, fname)

  elseif dname == "SIFT1M_query"

    fname = "./data/sift/sift_query.fvecs"
    X     = fvecs_read(nvectors, fname)

  elseif dname == "SIFT1M_groundtruth"

    fname = "./data/sift/sift_groundtruth.ivecs"
    X     = ivecs_read(nvectors, fname)

  elseif dname == "SIFT1M_base"

    fname = "./data/sift/sift_base.fvecs";
    X     = fvecs_read(nvectors, fname)

  else

    error("Dataset $(dname) unknown")

  end

  if V println("done."); end
  return X

end
