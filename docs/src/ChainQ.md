# Chain quantization (ChainQ)

Chain quantization (ChainQ) is an non-orthogonal MCQ method.

ChainQ uses codebooks that only have a dependency with the previous and next codebook, therefore creating a chain.
This allows the use of efficient polynomial (max-product) algorithms to find optimal encoding.

```@docs
quantize_chainq
train_chainq
```

## Reference

Babenko, A., & Lempitsky, V. (2015). Tree quantization for large-scale similarity search and classification. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ (pp. 4240-4248). [[PDF](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Babenko_Tree_Quantization_for_2015_CVPR_paper.pdf)]
