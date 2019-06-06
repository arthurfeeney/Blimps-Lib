

## NR-LSH

##### [Original paper](https://papers.nips.cc/paper/7559-norm-ranging-lsh-for-maximum-inner-product-search.pdf) on Norm-Ranging Locality Sensitive Hashing of which I am not an author. The authors claim that it improves SimpleLSH.


This implementation is header-only at the moment. Matrix operations use [Eigen](https://www.eigen.tuxfamily.org/index.php?title=Main_Page). It uses boost/multiprecision to allow for very long hash codes.
Tests use [Catch2](https://githubcom/catchorg/Catch2). This is header-only as well.

In the bind directory, there are some minimal python bindings that were made with [pybind11](https://www.github.com/pybind/pybind11). The movielenstest uses these bindings to make a pureSVD recommender system for the [MovieLens 10M dataset](https://grouplens.org/datasets/movielens/10m) in the spirit of [ALSH paper](https://www.arxiv.org/pdf/1405.5869.pdf), which I am also not an author of.


## LSH for Maximum Inner Product Search

[Bachrach et al.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf) show that MIPS is reducible to Near Neighbor Search, but
the dimension must be increased. So, if the data is transformed, it is possible
to use LSH for MIPS! The transformation used by SimpleLSH is

![\Large P(X)= (x, \sqrt{1 - ||x||_2^2})](https://latex.codecogs.com/svg.latex?x%3D%5Cfrac%7B-b%5Cpm%5Csqrt%7Bb%5E2-4ac%7D%7D%7B2a%7D)

This makes vectors with large inner products also (probably) have similar hashes.

![inner product vs hash similarity](images/inner_product_vs_simplelsh_similarity_32bits.png)
