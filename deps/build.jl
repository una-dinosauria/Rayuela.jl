using BinDeps

@BinDeps.setup

# cudautils   = library_dependency("cudautils")
linscan_aqd = library_dependency("linscan_aqd", aliases=["linscan_aqd","linscan_aqd.so"])
linscan_aqd_pairwise_byte = library_dependency("linscan_aqd_pairwise_byte", aliases=["linscan_aqd_pairwise_byte","linscan_aqd_pairwise_byte.so"])
encode_icm_so = library_dependency("encode_icm_so", aliases=["encode_icm_so", "encode_icm_so.so"])

# deps = [cudautils, linscan_aqd, linscan_aqd_pairwise_byte, encode_icm_so]
deps = [linscan_aqd, linscan_aqd_pairwise_byte, encode_icm_so]

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
    end),linscan_aqd, os = :Unix, installed_libpath=joinpath(prefix,"builds"))

provides(BuildProcess,
    (@build_steps begin
        CreateDirectory(linscan_aqdbuilddir)
        @build_steps begin
            ChangeDirectory(linscan_aqdbuilddir)
            FileRule(joinpath(prefix,"builds","linscan_aqd_pairwise_byte.so"),@build_steps begin
                `g++ -O3 -shared -fPIC ../src/linscan_aqd_pairwise_byte.cpp -o linscan_aqd_pairwise_byte.so -fopenmp`
            end)
        end
    end),linscan_aqd_pairwise_byte, os = :Unix, installed_libpath=joinpath(prefix,"builds"))

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
    end),encode_icm_so, os = :Unix, installed_libpath=joinpath(prefix,"builds"))

@BinDeps.install Dict([(:linscan_aqd => :linscan_aqd),
                      (:linscan_aqd_pairwise_byte => :linscan_aqd_pairwise_byte),
                      (:encode_icm_so => :encode_icm_so)])

# === CUDA code ===
# provides(BuildProcess,
#     (@build_steps begin
#         CreateDirectory(linscan_aqdbuilddir)
#         @build_steps begin
#             ChangeDirectory(linscan_aqdbuilddir)
#             FileRule(joinpath(prefix,"builds","cudautils.so"),@build_steps begin
#                 `/usr/local/cuda/bin/nvcc -ptx ../src/cudautils.cu -o cudautils.ptx -arch=compute_35`
#                 `/usr/local/cuda/bin/nvcc --shared -Xcompiler -fPIC -shared ../src/cudautils.cu -o cudautils.so -arch=compute_35`
#             end)
#         end
#     end),cudautils, os = :Unix, installed_libpath=joinpath(prefix,"builds"))

# @BinDeps.install Dict([(:linscan_aqd, :linscan_aqd),
#                       (:linscan_aqd_pairwise_byte, :linscan_aqd_pairwise_byte),
#                       (:encode_icm_so, :encode_icm_so),
#                       (:cudautils, :cudautils)])
