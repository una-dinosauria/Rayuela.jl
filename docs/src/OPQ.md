# Optimized product quantization (OPQ)

Optimized product quantization is an orthogonal MCQ method that also learns a rotation `R` of the data.

```@docs
quantize_opq
train_opq
```

## Reference

The main ideas were published at the same time by two independent groups:

* Ge, T., He, K., Ke, Q., & Sun, J. (2013). Optimized product quantization for approximate nearest neighbor search. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ (pp. 2946-2953). [[PDF](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Ge_Optimized_Product_Quantization_2013_CVPR_paper.pdf)]
* Norouzi, M., & Fleet, D. J. (2013). Cartesian k-means. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ (pp. 3017-3024). [[PDF](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Norouzi_Cartesian_K-Means_2013_CVPR_paper.pdf)]

An extended version was later publised in a computer vision journal:
* Ge, T., He, K., Ke, Q., & Sun, J. (2014). Optimized product quantization. _IEEE transactions on pattern analysis and machine intelligence_, 36(4), 744-755. [[PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/pami13opq.pdf)]
