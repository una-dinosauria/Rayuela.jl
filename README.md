# Rayuela.jl

[![][gitlab-img]][gitlab-url] [![][docs-latest-img]][docs-latest-url]

[gitlab-img]: https://gitlab.com/JuliaGPU/Rayuela.jl/badges/master/pipeline.svg
[gitlab-url]: https://gitlab.com/JuliaGPU/Rayuela.jl/pipelines

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://una-dinosauria.github.io/Rayuela.jl/latest/

This is the code for the paper

Julieta Martinez, Shobhit Zakhmi, Holger H. Hoos, James J. Little.
LSQ++: Lower running time and higher recall in multi-codebook quantization. In ECCV 2018. [[CVF](http://openaccess.thecvf.com/content_ECCV_2018/html/Julieta_Martinez_LSQ_lower_runtime_ECCV_2018_paper.html)].

[Rayuela](https://en.wikipedia.org/wiki/Hopscotch#Rayuela).jl is a package
that implements non-orthogonal multi-codebook quantization methods (MCQ).
These methods are useful for fast search of high-dimensional (d=100s-1000s) dense vectors.
Rayuela is written in [Julia](https://github.com/JuliaLang/julia).

This is not a production-ready library -- if you are looking for something
like that, maybe look at [faiss](https://github.com/facebookresearch/faiss).
Do note that the methods implemented by faiss and Rayuela.jl are almost entire orthogonal,
~~and that the libraries are distributed under different licenses~~
as of May 27 2019, FAISS is MIT licensed.

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

## Authors

The code in this package was written by
[Julieta Martinez](https://github.com/una-dinosauria/), with help from
[Joris Clement](https://github.com/flyingdutchman23) and
[Shobhit Zakhmi](https://github.com/Shobhit31).

## Requirements

This package is written in [Julia](https://github.com/JuliaLang/julia) 1.6, with some extension in C++ and CUDA.
You also need a CUDA-ready GPU. We have tested this code on an Nvidia Titan Xp GPU.

## Data and experiment setup

You may explore the library with `SIFT1M`, a classical retrieval dataset:

```bash
mkdir data
cd data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xvzf sift.tar.gz
rm sift.tar.gz
cd ..
```

Also make a directory for the results

```
mkdir -p results/sift1m
```
## Installing

Before all else, make sure that you have the `g++` compiler available from the command line, and the `nvcc` compiler availible at path `/usr/local/cuda/bin/nvcc`.

Clone this repo
```bash
git clone git@github.com:una-dinosauria/Rayuela.jl.git
```

Then, open julia and type `]` to enter Package mode:

```julia
julia>
(@v1.6) pkg>
```

Now you can activate this repo:

```julia
(@v1.6) pkg> activate .
(Rayuela) pkg>
```

This should load the relevant dependencies into your environment

Now you can run the demo:

```julia
julia> include("./demos/demos_train_query_base.jl")
```

For query/base/protocol (example by default runs on SIFT1M), or

```julia
julia> include("./demos/demos_query_base.jl")
```

For query/base protocol (example by default runs on LabelMe22K)

This will showcase PQ, OPQ, RVQ, ERVQ, ChainQ and LSQ/LSQ++ (SR-C and SR-D).

The rest of the datasets used in our ECCV'18 publication can be found at [gdrive](https://drive.google.com/drive/folders/1MnJLHpg5LP6pPQxQuL0VjnM03vHPvgP1?usp=sharing).

## Roadmap

### Implemented
- Product Quantization -- [TPAMI'11](https://hal.archives-ouvertes.fr/file/index/docid/514462/filename/paper_hal.pdf)
- Optimized Product Quantization / Cartesian K-means. [CVPR'13](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Norouzi_Cartesian_K-Means_2013_CVPR_paper.pdf), [CVPR'13](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Ge_Optimized_Product_Quantization_2013_CVPR_paper.pdf), [TPAMI'14](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/pami13opq.pdf)
- Tree Quantization<sup>[1](#ft1)</sup> -- [CVPR'15](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Babenko_Tree_Quantization_for_2015_CVPR_paper.pdf)
- Residual Vector Quantization -- [Sensors'10](http://www.mdpi.com/1424-8220/10/12/11259/htm)
- Stacked Quantizers (aka Enhanced Residual Vector Quantization) -- [arxiv](https://arxiv.org/abs/1411.2173)/[CBMI'14 (paywalled)](http://ieeexplore.ieee.org/abstract/document/6849842/)
- Local Search Quantization -- [ECCV'16](https://www.cs.ubc.ca/~julm/papers/eccv16.pdf)
- Local Search Quantization++ -- [ECCV'18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Julieta_Martinez_LSQ_lower_runtime_ECCV_2018_paper.pdf)
- Competitive Quantization -- [TKDE'16](https://ieeexplore.ieee.org/abstract/document/7539664/)
- Recall evaluation code

### TODO
Things I'd like to get around implementing / porting / wrapping some day (PRs are welcome!)
- Inverted Index -- [TPAMI'11](https://hal.archives-ouvertes.fr/file/index/docid/514462/filename/paper_hal.pdf), implemented in [faiss](https://github.com/facebookresearch/faiss)
- Inverted Multi-index -- [CPVR'12](https://pdfs.semanticscholar.org/5bfb/5a42483e9b7051fab5e972a3b4627a8d6a76.pdf), implemented in [faiss](https://github.com/facebookresearch/faiss)
- Locally optimized product quantization [CVPR'14](http://image.ntua.gr/iva/files/lopq.pdf), [code](https://github.com/yahoo/lopq), [project page](http://image.ntua.gr/iva/research/lopq/)
- Non-orthogonal multi-index --
 [CVPR'16](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Babenko_Efficient_Indexing_of_CVPR_2016_paper.pdf), [code](https://github.com/arbabenko/GNOIMI), [project page](http://sites.skoltech.ru/compvision/noimi/)
- Bolt -- [KDD'17](https://pdfs.semanticscholar.org/edae/41dc0b511cd0455388c9fd0720a086078cc6.pdf), [code](https://github.com/dblalock/bolt)
- Composite Quantization -- [ICML'14](https://pdfs.semanticscholar.org/eb18/329fe6466f36b0dbacd00e405c8f8618e1cf.pdf), [original code](https://github.com/hellozting/CompositeQuantization) (released late 2017, written in c++ with Microsoft's extensions)

### TODO (no code, low priority)
I would like to implement these methods. Some of them report really good results but, to the best of my knowledge, the authors have never released code. Also, my time is not infinite so ¯\\\_(ツ)\_/¯

- Sparse Composite Quantization -- [CVPR'15](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhang_Sparse_Composite_Quantization_2015_CVPR_paper.pdf)
- Tree Quantization with Gurobi optimization -- [CVPR'15](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Babenko_Tree_Quantization_for_2015_CVPR_paper.pdf)
- Joint K-means quantization -- [ICPR'16](http://ieeexplore.ieee.org/document/7900200/#full-text-section) (pay-walled)
- Pyramid encoding quantization -- [EUSIPCO'17](http://www.eurasip.org/Proceedings/Eusipco/Eusipco2017/papers/1570339946.pdf)
- Arborescence coding -- [ICCV'17](http://sites.skoltech.ru/app/data/uploads/sites/25/2017/08/AnnArbor_ICCV17.pdf)

## Citing
If you find this code useful, consider citing our work:
```
Julieta Martinez, Shobhit Zakhmi, Holger H. Hoos, James J. Little. "LSQ++:
Lower running time and higher recall in multi-codebook quantization",
ECCV 2018.
```
or
```
Julieta Martinez. "Algorithms for Large-Scale Multi-Codebook Quantization".
PhD thesis, 2018. (Coming soon)
```

As well as the corresponding paper of the method that you are using (see above).

## Notes
<a id="ft1">1</a>: The original implementation of [Tree Quantization](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Babenko_Tree_Quantization_for_2015_CVPR_paper.pdf)
requires a mixed-integer programming solver such as [Gurobi](http://www.gurobi.com/) for updating the codebooks.
We implement a special version of TQ that always create a chain
(not a general tree); thus encoding can be done with the Viterbi algorithm,
and codebook update is simpler and faster.
This method should have been a baseline in the TQ paper IMO.

## License
MIT
