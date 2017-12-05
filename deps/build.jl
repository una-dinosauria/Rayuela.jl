using BinDeps

@BinDeps.setup

deps = [
  cudautils   = library_dependency("cudautils")
  linscan_aqd = library_dependency("linscan_aqd")
  linscan_aqd_pairwise_byte = library_dependency("linscan_aqd_pairwise_byte")
  encode_icm_so = library_dependency("encode_icm_so")
]

prefix=joinpath(BinDeps.depsdir(linscan_aqd))
linscan_aqdbuilddir = joinpath(BinDeps.depsdir(linscan_aqd),"builds")

# === CPP linscan code ===
# TODO maybe refactor this?
provides(BuildProcess,
    (@build_steps begin
        CreateDirectory(linscan_aqdbuilddir)
        @build_steps begin
            ChangeDirectory(linscan_aqdbuilddir)
            FileRule(joinpath(prefix,"builds","linscan_aqd.so"),@build_steps begin
                `g++ -O3 -shared -fPIC ../src/linscan_aqd.cpp -o linscan_aqd.so -fopenmp`
            end)
        end
    end),linscan_aqd, os = :Linux, installed_libpath=joinpath(prefix,"builds"))

provides(BuildProcess,
    (@build_steps begin
        CreateDirectory(linscan_aqdbuilddir)
        @build_steps begin
            ChangeDirectory(linscan_aqdbuilddir)
            FileRule(joinpath(prefix,"builds","linscan_aqd_pairwise_byte.so"),@build_steps begin
                `g++ -O3 -shared -fPIC ../src/linscan_aqd_pairwise_byte.cpp -o linscan_aqd_pairwise_byte.so -fopenmp`
            end)
        end
    end),linscan_aqd_pairwise_byte, os = :Linux, installed_libpath=joinpath(prefix,"builds"))

provides(BuildProcess,
    (@build_steps begin
        CreateDirectory(linscan_aqdbuilddir)
        @build_steps begin
            ChangeDirectory(linscan_aqdbuilddir)
            FileRule(joinpath(prefix,"builds","encode_icm_so.so"),@build_steps begin
                `rm -f encode_icm_so.so`
                `g++ -O3 -shared -fPIC ../src/encode_icm.cpp -o encode_icm_so.so -fopenmp`
            end)
        end
    end),encode_icm_so, os = :Linux, installed_libpath=joinpath(prefix,"builds"))

# === CUDA code ===
provides(BuildProcess,
    (@build_steps begin
        CreateDirectory(linscan_aqdbuilddir)
        @build_steps begin
            ChangeDirectory(linscan_aqdbuilddir)
            FileRule(joinpath(prefix,"builds","cudautils.so"),@build_steps begin
                `nvcc -ptx ../src/cudautils.cu -o cudautils.ptx -arch=compute_35`
                `nvcc --shared -Xcompiler -fPIC -shared ../src/cudautils.cu -o cudautils.so -arch=compute_35`
            end)
        end
    end),cudautils, os = :Linux, installed_libpath=joinpath(prefix,"builds"))

# @BinDeps.install Dict([(:linscan_aqd, :linscan_aqd),
#                        (:linscan_aqd_pairwise_byte, :linscan_aqd_pairwise_byte)])

@BinDeps.install Dict([(:linscan_aqd, :linscan_aqd),
                       (:linscan_aqd_pairwise_byte, :linscan_aqd_pairwise_byte),
                       (:encode_icm_so, :encode_icm_so),
                       (:cudautils, :cudautils)])
