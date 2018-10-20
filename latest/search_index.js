var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Rayuela.jl-documentation-1",
    "page": "Home",
    "title": "Rayuela.jl documentation",
    "category": "section",
    "text": "Rayuela.jl is a package that implements non-orthogonal multi-codebook quantization methods (MCQ). These methods are useful for fast search of high-dimensional (d=100s-1000s) dense vectors. Rayuela is written in Julia.This is not a production-ready library – if you are looking for something like that, maybe look at faiss. Do note that the methods implemented by faiss and Rayuela.jl are almost entire orthogonal, and that the libraries are distributed under different licenses.Rayuela implements the main contributions that I made to this problem during my PhD at UBC, as well as multiple baselines for MCQ. The package is my attempt to make my research reproducible and accessible, and to make it easier for other people, specially newcomers, to contribute to this field, where lack of reproducibility is a major barrier of entry IMO.I originally intended to incorporate these contributions on top of faiss (see faiss/#185), but I soon realized that doing so would considerably delay the completion of my PhD. Julia is also more accessible (albeit a bit less performant) to quickly try and test new research ideas. In the future, savvier C++ programmers may port the most useful methods to faiss.Documentation is a work in progress. PRs are more than welcome."
},

{
    "location": "PQ.html#",
    "page": "Product quantization (PQ)",
    "title": "Product quantization (PQ)",
    "category": "page",
    "text": ""
},

{
    "location": "PQ.html#Rayuela.quantize_pq",
    "page": "Product quantization (PQ)",
    "title": "Rayuela.quantize_pq",
    "category": "function",
    "text": "quantize_pq(X, C, V=false) -> B\n\nGiven data and PQ codeboks, quantize.\n\nArguments\n\nX::Matrix{T}: d-by-n data to quantize\nC::Vector{Matrix{T}}: m-long vector with d/m-by-h matrix entries. Each matrix is a PQ codebook.\nV::Bool: Whether to print progress\n\nReturns\n\nB::Matrix{Int16}: m-by-n matrix with the codes that approximate X\n\n\n\n\n\n"
},

{
    "location": "PQ.html#Rayuela.train_pq",
    "page": "Product quantization (PQ)",
    "title": "Rayuela.train_pq",
    "category": "function",
    "text": "train_pq(X, m, h, niter=25, V=false) -> C, B, error\n\nTrains a product quantizer.\n\nArguments\n\nX::Matrix{T}: d-by-n data to quantize\nm::Integer: Number of codebooks\nh::Integer: Number of entries in each codebook (typically 256)\nniter::Integer: Number of k-means iterations to use\nV::Bool: Whether to print progress\n\nReturns\n\nC::Vector{Matrix{T}}: m-long vector with d-by-h matrix entries. Each matrix is a codebook of size approximately d/m-by-h.\nB::Matrix{Int16}: m-by-n matrix with the codes\nerror::T: The quantization error after training\n\n\n\n\n\n"
},

{
    "location": "PQ.html#Product-quantization-(PQ)-1",
    "page": "Product quantization (PQ)",
    "title": "Product quantization (PQ)",
    "category": "section",
    "text": "Product quantization is an orthogonal MCQ method that uses k-means as its main subroutine.quantize_pq\ntrain_pq"
},

{
    "location": "PQ.html#Reference-1",
    "page": "Product quantization (PQ)",
    "title": "Reference",
    "category": "section",
    "text": "Jégou, H., Douze, M., & Schmid, C. (2011). Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence, 33(1), 117-128. [PDF]"
},

{
    "location": "OPQ.html#",
    "page": "Optimized product quantization (OPQ)",
    "title": "Optimized product quantization (OPQ)",
    "category": "page",
    "text": ""
},

{
    "location": "OPQ.html#Rayuela.quantize_opq",
    "page": "Optimized product quantization (OPQ)",
    "title": "Rayuela.quantize_opq",
    "category": "function",
    "text": "quantize_opq(X, R, C, V=false) -> B\n\nGiven data and PQ/OPQ codeboks, quantize.\n\nArguments\n\nX::Matrix{T}: d-by-n data to quantize\nR::Matrix{T}: d-by-d rotation to apply to the data before quantizing\nC::Vector{Matrix{T}}: m-long vector with d/m-by-h matrix entries. Each matrix is a (O)PQ codebook.\nV::Bool: Whether to print progress\n\nReturns\n\nB::Matrix{Int16}: m-by-n matrix with the codes that approximate X\n\n\n\n\n\n"
},

{
    "location": "OPQ.html#Rayuela.train_opq",
    "page": "Optimized product quantization (OPQ)",
    "title": "Rayuela.train_opq",
    "category": "function",
    "text": "train_opq(X, m, h, niter, init, V=false) -> C, B, R, error\n\nTrains an optimized product quantizer.\n\nArguments\n\nX::Matrix{T}: d-by-n data to quantize\nm::Integer: Number of codebooks\nh::Integer: Number of entries in each codebook (typically 256)\nniter::Integer: Number of iterations to use\ninit::String: Method used to intiialize R, either \"natural\" (identity) or \"random\".\nV::Bool: Whether to print progress\n\nReturns\n\nB::Matrix{Int16}: m-by-n matrix with the codes\nC::Vector{Matrix{T}}: m-long vector with d-by-h matrix entries. Each matrix is a codebook of size approximately d/m-by-h.\nR::Matrix{T}: d-by-d learned rotation for the data\nobj::Vector{T}: The quantization error each iteration\n\n\n\n\n\n"
},

{
    "location": "OPQ.html#Optimized-product-quantization-(OPQ)-1",
    "page": "Optimized product quantization (OPQ)",
    "title": "Optimized product quantization (OPQ)",
    "category": "section",
    "text": "Optimized product quantization is an orthogonal MCQ method that also learns a rotation R of the data.quantize_opq\ntrain_opq"
},

{
    "location": "OPQ.html#Reference-1",
    "page": "Optimized product quantization (OPQ)",
    "title": "Reference",
    "category": "section",
    "text": "The main ideas were published at the same time by two independent groups:Ge, T., He, K., Ke, Q., & Sun, J. (2013). Optimized product quantization for approximate nearest neighbor search. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2946-2953). [PDF]\nNorouzi, M., & Fleet, D. J. (2013). Cartesian k-means. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 3017-3024). [PDF]An extended version was later publised in a computer vision journal:Ge, T., He, K., Ke, Q., & Sun, J. (2014). Optimized product quantization. IEEE transactions on pattern analysis and machine intelligence, 36(4), 744-755. [PDF]"
},

{
    "location": "RVQ.html#",
    "page": "Residual vector quantization (RVQ)",
    "title": "Residual vector quantization (RVQ)",
    "category": "page",
    "text": ""
},

{
    "location": "RVQ.html#Rayuela.quantize_rvq",
    "page": "Residual vector quantization (RVQ)",
    "title": "Rayuela.quantize_rvq",
    "category": "function",
    "text": "quantize_rvq(X, C, V=false) -> B, singletons\n\nGiven data and full-dimensional codebooks, quantize.\n\nArguments\n\nX::Matrix{T}: d-by-n data to quantize\nC::Vector{Matrix{T}}: m-long vector with d-by-h matrix entries. Each matrix is a codebook.\nV::Bool: Whether to print progress\n\nReturns\n\nB::Matrix{Int16}: m-by-n codes that approximate X\nsingletons::Vector{Matrix{T}}: m matrices with unused codebook entries\n\n\n\n\n\n"
},

{
    "location": "RVQ.html#Rayuela.train_rvq",
    "page": "Residual vector quantization (RVQ)",
    "title": "Rayuela.train_rvq",
    "category": "function",
    "text": "train_rvq(X, m, h, niter=25, V=false) -> C, B, error\n\nTrain a residual quantizer.\n\nArguments\n\nX::Matrix{T}: d-by-n data to quantize\nm::Integer: Number of codebooks\nh::Integer: Number of entries in each codebook (typically 256)\nniter::Integer: Number of iterations to use\nV::Bool: Whether to print progress\n\nReturns\n\nC::Vector{Matrix{T}}: m-long vector with d-by-h matrix entries. Each matrix is a codebook of size approximately d/m-by-h.\nB::Matrix{Int16}: m-by-n matrix with the codes\nerror::T: The quantization error after training\n\n\n\n\n\n"
},

{
    "location": "RVQ.html#Residual-vector-quantization-(RVQ)-1",
    "page": "Residual vector quantization (RVQ)",
    "title": "Residual vector quantization (RVQ)",
    "category": "section",
    "text": "Residual vector quantization (RVQ) is an non-orthogonal MCQ method.RVQ iteratively runs k-means to learn full-dimensional codebooksquantize_rvq\ntrain_rvq"
},

{
    "location": "RVQ.html#Reference-1",
    "page": "Residual vector quantization (RVQ)",
    "title": "Reference",
    "category": "section",
    "text": "Chen, Y., Guan, T., & Wang, C. (2010). Approximate nearest neighbor search by residual vector quantization. Sensors, 10(12), 11259-11273. [PDF]"
},

{
    "location": "ERVQ.html#",
    "page": "Enhanced residual vector quantization (ERVQ)",
    "title": "Enhanced residual vector quantization (ERVQ)",
    "category": "page",
    "text": ""
},

{
    "location": "ERVQ.html#Rayuela.quantize_ervq",
    "page": "Enhanced residual vector quantization (ERVQ)",
    "title": "Rayuela.quantize_ervq",
    "category": "function",
    "text": "quantize_ervq(X, C, V=false) -> B, singletons\n\nGiven data and full-dimensional codebooks, quantize. This methods is identical to quantize_rvq\n\nArguments\n\nX::Matrix{T}: d-by-n data to quantize\nC::Vector{Matrix{T}}: m-long vector with d-by-h matrix entries. Each matrix is a codebook.\nV::Bool: Whether to print progress\n\nReturns\n\nB::Matrix{Int16}: m-by-n codes that approximate X\nsingletons::Vector{Matrix{T}}: m matrices with unused codebook entries\n\n\n\n\n\n"
},

{
    "location": "ERVQ.html#Rayuela.train_ervq",
    "page": "Enhanced residual vector quantization (ERVQ)",
    "title": "Rayuela.train_ervq",
    "category": "function",
    "text": "train_ervq(X, B, C, m, h, niter, V=false) -> C, B, error\n\nTrain an enhanced residual quantizer / stacked quantizer. This method is typically initialized by Residual vector quantization (RVQ)\n\nArguments\n\nX::Matrix{T}: d-by-n data to quantize\nB::Matrix{T2}: m-by-n matrix with pre-trained codes\nC::Vector{Matrix{T}}: m-long vector with d-by-h matrices. Each matrix is a pretrained codebook of size approximately d-by-h.\nm::Integer: Number of codebooks\nh::Integer: Number of entries in each codebook (typically 256)\nniter::Integer: Number of iterations to use\nV::Bool: Whether to print progress\n\nT <: AbstractFloat and T2 <: Integer\n\nReturns\n\nC::Vector{Matrix{T}}: m-long vector with d-by-h matrix entries. Each matrix is a codebook of size approximately d-by-h.\nB::Matrix{Int16}: m-by-n matrix with the codes\nerror::T: The quantization error after training\n\n\n\n\n\n"
},

{
    "location": "ERVQ.html#Enhanced-residual-vector-quantization-(ERVQ)-1",
    "page": "Enhanced residual vector quantization (ERVQ)",
    "title": "Enhanced residual vector quantization (ERVQ)",
    "category": "section",
    "text": "Enhanced residual vector quantization (ERVQ), also known as Stacked quantizers (SQ) is an non-orthogonal MCQ method.The method is typically intialized with RVQ, and fine-tunes the codebooks to obtain a better approximation that can be quantized efficiently with quantize_rvq.quantize_ervq\ntrain_ervq"
},

{
    "location": "ERVQ.html#Reference-1",
    "page": "Enhanced residual vector quantization (ERVQ)",
    "title": "Reference",
    "category": "section",
    "text": "The main ideas come fromAi, L., Junqing, Y., Guan, T., & He, Y. (2014, June). Efficient approximate nearest neighbor search by optimized residual vector quantization. In Content-Based Multimedia Indexing (CBMI), 2014 12th International Workshop on (pp. 1-4). IEEE. [PDF]Independently, Martinez released a similar method with a some improvements and an open-source implementation:Martinez, J., Hoos, H. H., & Little, J. J. (2014). Stacked quantizers for compositional vector compression. arXiv preprint arXiv:1411.2173. [ArXiv]"
},

{
    "location": "ChainQ.html#",
    "page": "Chain quantization (ChainQ)",
    "title": "Chain quantization (ChainQ)",
    "category": "page",
    "text": ""
},

{
    "location": "ChainQ.html#Rayuela.quantize_chainq",
    "page": "Chain quantization (ChainQ)",
    "title": "Rayuela.quantize_chainq",
    "category": "function",
    "text": "quantize_chainq(X, C, use_cuda=false, use_cpp=false) -> B, ellapsed\n\nGiven data and chain codebooks, find codes using the Viterbi algorithm chain quantizer.\n\nArguments\n\nX::Matrix{T}: d-by-n data to quantize\nC::Vector{Matrix{T}}: m-long vector with d-by-h matrices. Each matrix is a pretrained codebook of size approximately d-by-h.\nuse_cuda::Bool: whether to use a CUDA implementation\nuse_cpp::Bool: whether to use a c++ implementation\n\nIf both use_cuda and use_cpp are true, the CUDA implementation is used.\n\nReturns\n\nB::Matrix{Int16}: m-by-n matrix with the codes\nellapsed::Float64: The time spent encoding\n\n\n\n\n\n"
},

{
    "location": "ChainQ.html#Rayuela.train_chainq",
    "page": "Chain quantization (ChainQ)",
    "title": "Rayuela.train_chainq",
    "category": "function",
    "text": "train_chainq(X, m, h, R, B, C, niter, V=false) -> C, B, R, error\n\nTrain a chain quantizer. This method is typically initialized by Optimized product quantization (OPQ)\n\nArguments\n\nX::Matrix{T}: d-by-n data to quantize\nm::Integer: Number of codebooks\nh::Integer: Number of entries in each codebook (typically 256)\nR::Matrix{T}: d-by-d rotation matrix for initialization\nB::Matrix{Int16}: m-by-n matrix with pre-trained codes for initialization\nC::Vector{Matrix{T}}: m-long vector with d-by-h matrices. Each matrix is a pretrained codebook of size approximately d-by-h.\nniter::Integer: Number of iterations to use\nV::Bool: Whether to print progress\n\nReturns\n\nC::Vector{Matrix{T}}: m-long vector with d-by-h matrix entries. Each matrix is a codebook of size approximately d-by-h.\nB::Matrix{Int16}: m-by-n matrix with the codes\nR::Matrix{T}: d-by-d optimized rotation matrix\nerror::T: The quantization error after training\n\n\n\n\n\n"
},

{
    "location": "ChainQ.html#Chain-quantization-(ChainQ)-1",
    "page": "Chain quantization (ChainQ)",
    "title": "Chain quantization (ChainQ)",
    "category": "section",
    "text": "Chain quantization (ChainQ) is an non-orthogonal MCQ method.ChainQ uses codebooks that only have a dependency with the previous and next codebook, therefore creating a chain. This allows the use of efficient polynomial (max-product) algorithms to find optimal encoding.quantize_chainq\ntrain_chainq"
},

{
    "location": "ChainQ.html#Reference-1",
    "page": "Chain quantization (ChainQ)",
    "title": "Reference",
    "category": "section",
    "text": "Babenko, A., & Lempitsky, V. (2015). Tree quantization for large-scale similarity search and classification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 4240-4248). [PDF]"
},

{
    "location": "LSQ.html#",
    "page": "Local search quantization (LSQ)",
    "title": "Local search quantization (LSQ)",
    "category": "page",
    "text": ""
},

{
    "location": "LSQ.html#Rayuela.encoding_icm",
    "page": "Local search quantization (LSQ)",
    "title": "Rayuela.encoding_icm",
    "category": "function",
    "text": "encoding_icm(X, oldB, C, ilsiter, icmiter, randord, npert, cpp=true, V=false) -> B\n\nGiven data and chain codebooks, find codes using iterated local search with ICM.\n\nArguments\n\nX::Matrix{T}: d-by-n data to quantize\nOldB::Matrix{Int16}: m-by-n initial set of codes\nilsiter::Integer: Number of iterated local search (ILS) iterations\nicmiter::Integer: Number of iterated conditional modes (ICM) iterations\nrandord::Bool: Whether to use random order\nnpert::Integer: Number of codes to perturb\ncpp::Bool=true: Whether to use the c++ implementation\nV::Bool=false: Whehter to print progress\n\nReturns\n\nB::Matrix{Int16}: m-by-n matrix with the new codes\n\n\n\n\n\n"
},

{
    "location": "LSQ.html#Rayuela.train_lsq",
    "page": "Local search quantization (LSQ)",
    "title": "Rayuela.train_lsq",
    "category": "function",
    "text": "train_lsq(X, m, h, R, B, C, niter, ilsiter, icmiter, randord, npert, cpp=true, V=false) -> C, B, obj\n\nTrain a local-search quantizer. This method is typically initialized by Chain quantization (ChainQ)\n\nArguments\n\nX::Matrix{T}: d-by-n data to quantize\nm::Integer: Number of codebooks\nh::Integer: Number of entries in each codebook (typically 256)\nR::Matrix{T}: d-by-d rotation matrix for initialization\nB::Matrix{Int16}: m-by-n matrix with pre-trained codes for initialization\nC::Vector{Matrix{T}}: m-long vector with d-by-h matrices. Each matrix is a pretrained codebook of size approximately d-by-h\nniter::Integer: Number of iterations to use\nilster::Integer: Number of iterated local search (ILS) iterations\nicmiter::Integer: Number of iterated conditional modes (ICM) iterations\nrandord::Bool: Whether to visit the nodes in a random order in ICM\nnpert::Integer: Number of codes to perturb\ncpp::Bool: Whether to use a c++ implementation for encoding\nV::Bool: Whether to print progress\n\nReturns\n\nC::Vector{Matrix{T}}: m-long vector with d-by-h matrix entries. Each matrix is a codebook of size approximately d-by-h\nB::Matrix{Int16}: m-by-n matrix with the codes\nobj::Vector{T}: niter-long vector with the quantization error after each iteration\n\n\n\n\n\n"
},

{
    "location": "LSQ.html#Local-search-quantization-(LSQ)-1",
    "page": "Local search quantization (LSQ)",
    "title": "Local search quantization (LSQ)",
    "category": "section",
    "text": "Local search quantization (LSQ) is an non-orthogonal MCQ method.LSQ uses fully dimensional codebooks. Codebook update is done via least squares, and encoding is done with iterated local search (ILS), using randomized iterated conditional modes (ICM) as a local search subroutine.encoding_icm\ntrain_lsq"
},

{
    "location": "LSQ.html#Reference-1",
    "page": "Local search quantization (LSQ)",
    "title": "Reference",
    "category": "section",
    "text": "Martinez, J., Clement, J., Hoos, H. H., & Little, J. J. (2016). Revisiting additive quantization. In European Conference on Computer Vision (pp. 137-153). Springer, Cham. [PDF]"
},

]}
