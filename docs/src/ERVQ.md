# Enhanced residual vector quantization (ERVQ)

Enhanced residual vector quantization (ERVQ), also known as Stacked quantizers (SQ) is an non-orthogonal MCQ method.

The method is typically intialized with RVQ, and fine-tunes the codebooks to obtain a better approximation that can be
quantized efficiently with [`quantize_rvq`](@ref).

```@docs
quantize_ervq
train_ervq
```

## Reference

The main ideas come from

* Ai, L., Junqing, Y., Guan, T., & He, Y. (2014, June). Efficient approximate nearest neighbor search by optimized residual vector quantization. In _Content-Based Multimedia Indexing (CBMI), 2014 12th International Workshop on_ (pp. 1-4). IEEE. [[PDF](https://ieeexplore.ieee.org/abstract/document/6849842)]

Independently, Martinez released a similar method with a some improvements and an open-source implementation:

* Martinez, J., Hoos, H. H., & Little, J. J. (2014). Stacked quantizers for compositional vector compression. _arXiv preprint arXiv:1411.2173_. [[ArXiv](https://arxiv.org/abs/1411.2173)]
