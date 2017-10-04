# Rayuela.jl

[Rayuela](https://en.wikipedia.org/wiki/Hopscotch#Rayuela).jl is a package for fast search of high-dimensional (d=100-1000s)
dense vectors. The package is mostly focused on multi-codebook quantization
techniques, and is loosely inspired by [faiss](https://github.com/facebookresearch/faiss).

Rayuela implements the main contributions that I made to this problem during my
PhD at UBC. The package is my attempt to make my research reproducible and
accessible, and to make it easier for other people, specially newcomers, to
contribute to this field.

I originally intended to incorporate these contributions on top of [faiss](https://github.com/facebookresearch/faiss)
(see https://github.com/facebookresearch/faiss/issues/185), but I realized that
I would not have the time to do so and finish my PhD before 2018. Julia is also
more accessible (albeit a bit less performant) to quickly try and test new
research ideas. In the future, savvier C++ programmers can port the most useful methods to [faiss](https://github.com/facebookresearch/faiss).

## Installation

You can install the package via 

```julia
julia> Pkg.clone("https://github.com/una-dinosauria/Rayuela.jl.git")
```

if you do not have a github account, or

```julia
julia> Pkg.clone("git@github.com:una-dinosauria/Rayuela.jl.git")
```

if you do.

## Roadmap

### Implemented
- Product Quantization -- [TPAMI'11](https://hal.archives-ouvertes.fr/file/index/docid/514462/filename/paper_hal.pdf)
- Optimized Product Quantization / Cartesian K-means. [CVPR'13](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Norouzi_Cartesian_K-Means_2013_CVPR_paper.pdf), [CVPR'13](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Ge_Optimized_Product_Quantization_2013_CVPR_paper.pdf), [TPAMI'14](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/pami13opq.pdf)
- Tree Quantization* -- [CVPR'15](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Babenko_Tree_Quantization_for_2015_CVPR_paper.pdf)
- Local Search Quantization -- [ECCV'16](https://www.cs.ubc.ca/~julm/papers/eccv16.pdf)

### TODO
Things I'd like to get around implementing / wrapping from faiss some day
- Inverted Index -- [TPAMI'11](https://hal.archives-ouvertes.fr/file/index/docid/514462/filename/paper_hal.pdf), implemented in [faiss](https://github.com/facebookresearch/faiss)
- Inverted Multi-index -- [CPVR'12](https://pdfs.semanticscholar.org/5bfb/5a42483e9b7051fab5e972a3b4627a8d6a76.pdf), implemented in [faiss](https://github.com/facebookresearch/faiss)
- Locally optimized product quantization [CVPR'14](http://image.ntua.gr/iva/files/lopq.pdf), [code](https://github.com/yahoo/lopq), [project page](http://image.ntua.gr/iva/research/lopq/)
- Non-orthogonal multi-index --
 [CVPR'16](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Babenko_Efficient_Indexing_of_CVPR_2016_paper.pdf), [code](https://github.com/arbabenko/GNOIMI), [project page](http://sites.skoltech.ru/compvision/noimi/)
- Polysemous codes -- [ECCV'16](https://arxiv.org/pdf/1609.01882.pdf), implemented in [faiss](https://github.com/facebookresearch/faiss)
- Bolt -- [KDD'17](https://pdfs.semanticscholar.org/edae/41dc0b511cd0455388c9fd0720a086078cc6.pdf), [code](https://github.com/dblalock/bolt)
- Improvements to LSQ -- [under review]()
- All the GPU code in [faiss](https://github.com/facebookresearch/faiss/tree/master/gpu) :heart_eyes:

### TODO (no code, low priority)
I would like to implement these methods. Some of them report really good results but, to the best of my knowledge, the authors have never released code. Also, my time is not infinite so ¯\\\_(ツ)\_/¯
- Composite Quantization -- [ICML'14](https://pdfs.semanticscholar.org/eb18/329fe6466f36b0dbacd00e405c8f8618e1cf.pdf)
- Sparse Composite Quantization -- [CVPR'15](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhang_Sparse_Composite_Quantization_2015_CVPR_paper.pdf)
- Tree Quantization with Gurobi optimization -- [CVPR'15](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Babenko_Tree_Quantization_for_2015_CVPR_paper.pdf)
- Competitive Quantization -- [TKDE'16](https://www.researchgate.net/profile/Serkan_Kiranyaz/publication/306046688_Competitive_Quantization_for_Approximate_Nearest_Neighbor_Search/links/57bd58bb08ae6c703bc64909.pdf)
- Joint K-means quantization -- [ICPR'16](http://ieeexplore.ieee.org/document/7900200/#full-text-section) (pay-walled)
- Pyramid encoding quantization -- [EUSIPCO'17](http://www.eurasip.org/Proceedings/Eusipco/Eusipco2017/papers/1570339946.pdf)
- Arborescence coding -- [ICCV'17](http://sites.skoltech.ru/app/data/uploads/sites/25/2017/08/AnnArbor_ICCV17.pdf)

## Notes
\* The original implementation of [Tree Quantization](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Babenko_Tree_Quantization_for_2015_CVPR_paper.pdf)
requires Gurobi. We implement a special version of TQ that always create a chain
(not a general tree); thus encoding can be done with the Viterbi algorithm.
This method should have been a baseline in the TQ paper IMO.
