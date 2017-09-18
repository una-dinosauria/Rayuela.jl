# Rayuela.jl

Rayuela.jl is a package for fast search of high-dimensional (d=100-1000s)
dense vectors. The package is mostly focused on multi-codebook quantization
techniques, and is loosely inspired by [faiss](https://github.com/facebookresearch/faiss).

Rayuela implements the main contributions that I made to this problem during my
PhD at UBC. Rayuela is my attempt to make my research reproducible and
accessible, and to make it easier for other people, specially newcomers, to
contribute to this field.

I originally intended to incorporate these contributions on top of [faiss](https://github.com/facebookresearch/faiss)
(see https://github.com/facebookresearch/faiss/issues/185), but I realized that
I would not have the time to do so and finish my PhD before 2018. Julia is also
more accessible (albeit a bit less performant) to quickly try and test new
research ideas. In the future, successful methods can be ported to [faiss](https://github.com/facebookresearch/faiss).

## Implemented
- Product Quantization -- [TPAMI'11](https://hal.archives-ouvertes.fr/file/index/docid/514462/filename/paper_hal.pdf)
- Optimized Product Quantization / Cartesian K-means. [CVPR'13](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Norouzi_Cartesian_K-Means_2013_CVPR_paper.pdf), [CVPR'13](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Ge_Optimized_Product_Quantization_2013_CVPR_paper.pdf), [TPAMI'14](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/pami13opq.pdf)
- Tree Quantization* -- [CVPR'15](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Babenko_Tree_Quantization_for_2015_CVPR_paper.pdf)
- Local Search Quantization -- [ECCV'16](https://www.cs.ubc.ca/~julm/papers/eccv16.pdf)

## TODO
Things I'd like to get around implementing / wrapping from faiss some day
- Inverted Index -- [TPAMI'11](https://hal.archives-ouvertes.fr/file/index/docid/514462/filename/paper_hal.pdf), implemented in [faiss](https://github.com/facebookresearch/faiss)
- Inverted Multi-index -- [CPVR'12](https://pdfs.semanticscholar.org/5bfb/5a42483e9b7051fab5e972a3b4627a8d6a76.pdf), implemented in [faiss](https://github.com/facebookresearch/faiss)
- Non-orthogonal multi-index -- [CVPR'16](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Babenko_Efficient_Indexing_of_CVPR_2016_paper.pdf), [code](https://github.com/arbabenko/GNOIMI), [project page](http://sites.skoltech.ru/compvision/noimi/)
- Polysemous codes -- [ECCV'16](https://arxiv.org/pdf/1609.01882.pdf), implemented in [faiss](https://github.com/facebookresearch/faiss)
- All the GPU code in [faiss](https://github.com/facebookresearch/faiss) 😍

## TODO (hard, low priority)
I would love to implement these methods, but the authors have never released
code and my time is not infinite so ¯\\\_(ツ)\_/¯
- Composite Quantization -- [ICML'14](https://pdfs.semanticscholar.org/eb18/329fe6466f36b0dbacd00e405c8f8618e1cf.pdf)
- Sparse Composite Quantization -- [CVPR'15](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhang_Sparse_Composite_Quantization_2015_CVPR_paper.pdf)
- Tree Quantization with Gurobi optimization -- [CVPR'15](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Babenko_Tree_Quantization_for_2015_CVPR_paper.pdf)

## Notes
\* The original implementation of [Tree Quantization](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Babenko_Tree_Quantization_for_2015_CVPR_paper.pdf)
requires Gurobi. We implement a special version of TQ that always create a chain
(not a general tree); thus encoding can be done with the Viterbi algorithm.
This method should have been a baseline in the TQ paper IMO.
