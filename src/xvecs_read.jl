
# Read a set of vectors stored in the xvec format (int + n * float)
# The function returns a set of output uint8 vector (one vector per column)
#
# Syntax:
#   v = bvecs_read (filename)        -> read all vectors
#   v = bvecs_read (filename, n)      -> read n vectors
#   v = bvecs_read (filename, [a b]) -> read the vectors from a to b (indices starts from 1)

# TODO(julieta) refactor these with metaprogrammming

export bvecs_read, fvecs_read, ivecs_read

function bvecs_read(
  bounds::UnitRange,
  filename::AbstractString="./data/sift1b/bigann_base.bvecs")

  @assert bounds.start >= 1

  # open the file and count the number of descriptors
  open(filename, "r") do fid

    # Read the vector size
    d = zeros(Int32, 1)
    read!(fid, d)
    vecsizeof = 1 * 4 + d[1]

    # Get the number of vectrors
    seekend(fid)
    vecnum = position(fid) / vecsizeof

    # compute the number of vectors that are really read and go in starting positions
    n = bounds.stop - bounds.start + 1
    seekstart(fid)
    skip(fid, (bounds.start - 1) * vecsizeof)

    # read n vectors
    v = zeros(UInt8, vecsizeof * n)
    read!(fid, v)
    v = reshape(v, vecsizeof, n)

    # Check if the first column (dimension of the vectors) is correct
    @assert sum( v[1,2:end] .== v[1, 1]) == n - 1
    @assert sum( v[2,2:end] .== v[2, 1]) == n - 1
    @assert sum( v[3,2:end] .== v[3, 1]) == n - 1
    @assert sum( v[4,2:end] .== v[4, 1]) == n - 1
    v = v[5:end, :]

    return v

  end
end

function bvecs_read(n::Integer, filename::AbstractString)
  return bvecs_read(1:n,filename)
end

function bvecs_read(n::Integer)
  return bvecs_read(1:n)
end


function fvecs_read(
  bounds::UnitRange,
  filename::AbstractString="./data/deep/deep10M.fvecs")

  @assert bounds.start >= 1

  # open the file and count the number of descriptors
  open(filename, "r") do fid

    # Read the vector size
    d = zeros(Int32, 1)
    read!(fid, d)
    vecsizeof = 1 * 4 + d[1] * 4

    # Get the number of vectrors
    seekend(fid)
    vecnum = position(fid) / vecsizeof

    # compute the number of vectors that are really read and go in starting positions
    n = bounds.stop - bounds.start + 1
    seekstart(fid)
    skip(fid, (bounds.start - 1) * vecsizeof)

    # read n vectors
    v = zeros(Float32, (d[1] + 1) * n)
    read!(fid, v)
    v = reshape(v, d[1] + 1, n)

    # Check if the first column (dimension of the vectors) is correct
    @assert sum( v[1,2:end] .== v[1, 1]) == n - 1
    v = v[2:end, :]

    return v

  end
end

function fvecs_read(n::Integer, filename::AbstractString)
  return fvecs_read(1:n,filename)
end

function fvecs_read(n::Integer)
  return fvecs_read(1:n)
end


function ivecs_read(
  bounds::UnitRange,
  filename::String="./data/deep_babenko/deep1M_groundtruth.ivecs")

  @assert bounds.start >= 1

  # open the file and count the number of descriptors
  open(filename, "r") do fid

    # Read the vector size
    d = zeros(Int32, 1)
    read!(fid, d)
    vecsizeof = 1 * 4 + d[1] * 4

    # Get the number of vectrors
    seekend(fid)
    vecnum = position(fid) / vecsizeof

    # compute the number of vectors that are really read and go in starting positions
    n = bounds.stop - bounds.start + 1
    seekstart(fid)
    skip(fid, (bounds.start - 1) * vecsizeof)

    # read n vectors
    v = zeros(Int32, (d[1] + 1) * n)
    read!(fid, v)
    v = reshape(v, d[1] + 1, n)

    # Check if the first column (dimension of the vectors) is correct
    @assert sum( v[1,2:end] .== v[1, 1]) == n - 1
    v = v[2:end, :]

    return v

  end
end

function ivecs_read(n::Integer, filename::String)
  return ivecs_read(1:n,filename)
end

function ivecs_read(n::Integer)
  return ivecs_read(1:n)
end
