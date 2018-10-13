# Rayuela.jl documentation

[Rayuela](https://en.wikipedia.org/wiki/Hopscotch#Rayuela).jl is a package
that implements non-orthogonal multi-codebook quantization methods (MCQ).
These methods are useful for fast search of high-dimensional (d=100s-1000s) dense vectors.
Rayuela is written in [Julia](https://github.com/JuliaLang/julia).

This is not a production-ready library -- if you are looking for something
like that, maybe look at [faiss](https://github.com/facebookresearch/faiss).
Do note that the methods implemented by faiss and Rayuela.jl are almost entire orthogonal,
and that the libraries are distributed under different licenses.

Rayuela implements the main contributions that I made to this problem during my
PhD at [UBC](https://cs.ubc.ca), as well as multiple baselines for MCQ.
The package is my attempt to make my research reproducible
and accessible, and to make it easier for other people, specially newcomers, to
contribute to this field, where lack of reproducibility is a major barrier of entry IMO.

I originally intended to incorporate these contributions on top of [faiss](https://github.com/facebookresearch/faiss)
(see [faiss/#185](https://github.com/facebookresearch/faiss/issues/185)),
but I soon realized that doing so would considerably delay the completion of my PhD.
Julia is also more accessible (albeit a bit less performant) to quickly try and
test new research ideas.
In the future, savvier C++ programmers may port the most useful methods to faiss.

Documentation is a work in progress. PRs are more than welcome.
