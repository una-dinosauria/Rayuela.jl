
export read_cq_bvecs, read_cq_fvecs, CQ_parameters, dump_CQ_parameters


# Read float binary file produced by CQ, like the codebooks D
function read_cq_fvecs(fname::AbstractString)

  dim, count = zero(Cint), zero(Cint)
  vectors = Matrix{Cfloat}

  open(fname, "r") do fid
    count = read(fid, Cint, 1)[1]
    dim   = read(fid, Cint, 1)[1]
    vectors = read(fid, Cfloat, dim, count)
  end

  return vectors
end


# Read int binary file produced by CQ, like the codes B
function read_cq_bvecs(fname::AbstractString)

  dim, count = zero(Cint), zero(Cint)
  codes = Matrix{Cint}

  open(fname, "r") do fid
    count = read(fid, Cint, 1)[1]
    dim   = read(fid, Cint, 1)[1]
    codes = read(fid, Cint, dim, count)
  end

  return codes
end


# Parameters that CQ receives
@with_kw mutable struct CQ_parameters
  PQ::Bool=false
  NCQ::Bool=false
  CQ::Bool=true
  Search::Bool=false

  # global parameters
  # points_count=1000000
  points_count::Int=100000
  dictionaries_count::Int=8
  words_count::Int=256
  space_dimension::Int=128
  # points_file=/home/julieta/Desktop/local-search-quantization/data/sift/sift_base.fvecs
  points_file::String="/home/julieta/Desktop/local-search-quantization/data/sift/sift_learn.fvecs"
  output_file_prefix::String="/home/julieta/Desktop/CQ/build/temp/"
  max_iter::Int=30

  # PQ parameters
  distortion_tol::Float32=0.0001
  read_partition::Int=0
  partition_file::String=""
  # if 101 then using closure cluster, else lloyd kmeans
  kmeans_method::Int=101

  # NCQ and CQ parameters
  num_sep::Int=20
  # initial from outside, if 1 then set the file name of dictinary and codes
  initial_from_outside=0
  dictionary_file::String=""
  binary_codes_file::String=""

  # CQ parameters
  mu::Float32=0.0004f0

  # Search parameters
  queries_count::Int=10000
  groundtruth_length::Int=100
  result_length::Int=1000
  queries_file::String="/home/julieta/Desktop/local-search-quantization/data/sift/sift_query.fvecs"
  groundtruth_file::String="/home/julieta/Desktop/local-search-quantization/data/sift/sift_groundtruth.ivecs"
  trained_dictionary_file::String="/home/julieta/Desktop/CQ/build/temp/D"
  trained_binary_codes_file::String="/home/julieta/Desktop/CQ/build/temp/B"
  output_retrieved_results_file::String="/home/julieta/Desktop/CQ/build/temp/recall"
end


# Print CQ parameters to a config file that the CQ binary can consume
function dump_CQ_parameters(p::CQ_parameters, fname::String)
  open(fname, "w") do fid
    for (name, typ) in zip(fieldnames(CQ_parameters), CQ_parameters.types)
      if typ == Bool || typ == Int
        println(fid, name, "=", Int(getfield(p, name)))
      else
        println(fid, name, "=", getfield(p, name))
      end
    end
  end
end

