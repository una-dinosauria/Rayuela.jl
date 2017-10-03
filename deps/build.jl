using BinDeps

@BinDeps.setup

linscan_aqd = library_dependency("linscan_aqd")
linscan_aqd_pairwise_byte = library_dependency("linscan_aqd_pairwise_byte")

prefix=joinpath(BinDeps.depsdir(linscan_aqd))
linscan_aqdbuilddir = joinpath(BinDeps.depsdir(linscan_aqd),"builds")

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

# @BinDeps.install Dict(:linscan_aqd => :linscan_aqd)
@BinDeps.install Dict([(:linscan_aqd, :linscan_aqd),
                        (:linscan_aqd_pairwise_byte, :linscan_aqd_pairwise_byte)])
