# Local search quantization (LSQ)

Local search quantization (LSQ) is an non-orthogonal MCQ method.

LSQ uses fully dimensional codebooks. Codebook update is done via least squares, and encoding is done with
iterated local search (ILS), using randomized iterated conditional modes (ICM) as a local search subroutine.

```@docs
encoding_icm
train_lsq
```

## Reference

Martinez, J., Clement, J., Hoos, H. H., & Little, J. J. (2016). Revisiting additive quantization. In _European Conference on Computer Vision_ (pp. 137-153). Springer, Cham. [[PDF](https://www.cs.ubc.ca/~julm/papers/eccv16.pdf)]
