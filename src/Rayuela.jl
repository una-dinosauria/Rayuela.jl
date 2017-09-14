module Rayuela

using Clustering, Distances

export func

include("PQ.jl")

"""
    func(x::Int)

Returns double the number `x` plus `1`.
"""
func(x::Int) = 2x + 1

# package code goes here

end # module
