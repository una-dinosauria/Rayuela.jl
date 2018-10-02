module CudaUtilsModule

import CUDAdrv
import CUDAdrv: CuModule, CuModuleFile, unsafe_unload!,
                CuFunction, cudacall, CuDevice, CuArray, CuDim

const ptxdict = Dict()
const mdlist = CuModule[]

function mdinit(devlist, ptxfile)
  global ptxdict
  global mdlist
  # isempty(mdlist) || error("mdlist is not empty")
  for dev in devlist
    # device(dev)
    CuDevice(dev)
    md = CuModuleFile(ptxfile)

    # Utils functions
    ptxdict["setup_kernel"] = CuFunction(md, "setup_kernel")
    ptxdict["perturb"]      = CuFunction(md, "perturb")
    # ptxdict["veccost"]      = CuFunction(md, "veccost")
    ptxdict["veccost2"]     = CuFunction(md, "veccost2")

    # Add vector to matrix
    ptxdict["vec_add"]         = CuFunction(md, "vec_add")

    # Forward pass of the viterbi algorithm
    ptxdict["viterbi_forward"] = CuFunction(md, "viterbi_forward")

    # ICM functions
    # ptxdict["condition_icm"]  = CuFunction(md, "condition_icm");
    # ptxdict["condition_icm2"] = CuFunction(md, "condition_icm2");
    ptxdict["condition_icm3"] = CuFunction(md, "condition_icm3");

    push!(mdlist, md)
  end
end

# mdclose() = (for md in mdlist; unsafe_unload!(md); end; empty!(mdlist); empty!(ptxdict))
mdclose() = (empty!(mdlist); empty!(ptxdict))

function finit()
  mdclose()
end

function init(devlist, ptxfile)
  mdinit(devlist, ptxfile)
end

# # Accumulates binaries to unaries for icm conditioning
# function condition_icm(
#   nblocks::Integer, nthreads::Integer,
#   d_ub::CuArray{Cfloat, 2},     # in/out. h-by-n. where we accumulate
#   d_bb::CuArray{Cfloat, 2},     # in. binaries
#   d_codek::CuArray{Cuchar, 1},  # in. n-long. indices into d_bb
#   n::Cint,                        # in. size( d_ub, 2 )
#   h::Cint)                        # in. size( d_ub, 1 )
#
#   fun = ptxdict["condition_icm"];
#
#   cudacall( fun, nblocks, nthreads,
#     (d_ub, d_bb, d_codek, n, h));
#
#   return nothing
# end

# function condition_icm2(
#   nblocks::Integer, nthreads::CuDim,
#   d_ub::CuArray{Cfloat, 2},     # in/out. h-by-n. where we accumulate
#   d_bb::CuArray{Cfloat, 2},     # in. binaries
#   d_codek::CuArray{Cuchar, 1},  # in. n-long. indices into d_bb
#   n::Cint,                        # in. size( d_ub, 2 )
#   h::Cint)                        # in. size( d_ub, 1 )
#
#   fun = ptxdict["condition_icm2")];
#
#   cudacall( fun, nblocks, nthreads,
#     (d_ub, d_bb, d_codek, n, h));
#
#   return nothing
# end

# CUDAdrv.Mem.Buffer
function condition_icm3(
  nblocks::Integer, nthreads::CuDim,
  # d_ub::CuArray{Cfloat, 2},     # in/out. h-by-n. where we accumulate
  # d_bbs::CuArray{Cfloat, 2},     # in. binaries
  # d_codek::CuArray{Cuchar, 2},  # in. n-long. indices into d_bb
  d_ub::CUDAdrv.Mem.Buffer,     # in/out. h-by-n. where we accumulate
  d_bbs::CUDAdrv.Mem.Buffer,     # in. binaries
  d_codek::CUDAdrv.Mem.Buffer,  # in. n-long. indices into d_bb
  conditioning::Cint,
  m::Cint,
  n::Cint)                        # in. size( d_ub, 2 )

  fun = ptxdict["condition_icm3"]
  cudacall(fun,
    (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cuchar}, Cint, Cint, Cint),
    d_ub, d_bbs, d_codek, conditioning, m, n; threads=(nblocks, nthreads))

  return nothing
end

function vec_add(
  nblocks::CuDim, nthreads::CuDim,
  d_matrix::CUDAdrv.Mem.Buffer,
  d_vec::CUDAdrv.Mem.Buffer,
  n::Cint,
  h::Cint)

  fun = ptxdict["vec_add"]
  cudacall(fun,
    Tuple{Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint},
    d_matrix, d_vec, n, h; blocks=nblocks, threads=nthreads)
  end

function viterbi_forward(
  nblocks::CuDim, nthreads::CuDim,
  d_matrix::CUDAdrv.Mem.Buffer,
  d_vec::CUDAdrv.Mem.Buffer,
  d_minv::CUDAdrv.Mem.Buffer,
  d_mini::CUDAdrv.Mem.Buffer,
  n::Cint, j::Cint)

  fun = ptxdict["viterbi_forward"]
  cudacall(fun,
    (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cint}, Cint, Cint),
    d_matrix, d_vec, d_minv, d_mini, n, j; blocks=nblocks, threads=nthreads)
end

# function veccost(
#   nblocks::CuDim, nthreads::CuDim,
#   d_rx::CuArray{Cfloat},
#   d_codebooks::CuArray{Cfloat},
#   d_codes::CuArray{Cuchar},
#   d_veccost::CuArray{Cfloat}, # out.
#   m::Cint,
#   n::Cint)
#
#   fun = ptxdict["veccost"];
#   cudacall( fun, nblocks, nthreads,
#     ( d_rx, d_codebooks, d_codes, d_veccost, m, n ));
# end

function veccost2(
  nblocks::CuDim, nthreads::CuDim,
  d_rx::CUDAdrv.Mem.Buffer,
  d_codebooks::CUDAdrv.Mem.Buffer,
  d_codes::CUDAdrv.Mem.Buffer,
  d_veccost::CUDAdrv.Mem.Buffer, # out.
  d::Cint,
  m::Cint,
  n::Cint)

  fun = ptxdict["veccost2"]
  cudacall(fun, nblocks, nthreads,
    (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cuchar}, Ptr{Cfloat}, Cint, Cint, Cint),
    d_rx, d_codebooks, d_codes, d_veccost, d, m, n,
    shmem=Int(d*sizeof(Cfloat)))
end

function perturb(
  nblocks::CuDim, nthreads::CuDim,
  state::CUDAdrv.Mem.Buffer,
  codes::CUDAdrv.Mem.Buffer,
  n::Cint,
  m::Cint,
  k::Cint)

  fun = ptxdict["perturb"]
  cudacall( fun, nblocks, nthreads,
    (Ptr{Void}, Ptr{Cuchar}, Cint, Cint, Cint),
    state, codes, n, m, k)
end

function setup_kernel(
  nblocks::CuDim, nthreads::CuDim,
  n::Cint,
  state::CUDAdrv.Mem.Buffer)

  fun = ptxdict["setup_kernel"]
  cudacall(fun, nblocks, nthreads,
    (Cint, Ptr{Void}),
    n, state)
end

end #cudaUtilsModule
