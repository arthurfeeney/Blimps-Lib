

## Blimps Lib

##### [Original paper](https://papers.nips.cc/paper/7559-norm-ranging-lsh-for-maximum-inner-product-search.pdf) on Norm-Ranging Locality Sensitive Hashing of which I am not an author. The authors claim that it improves SimpleLSH.


This implementation is header-only at the moment. Matrix operations use [Eigen](https://www.eigen.tuxfamily.org/index.php?title=Main_Page). It uses boost/multiprecision to allow for very long hash codes.
Tests use [Catch2](https://githubcom/catchorg/Catch2). This is header-only as well.

In the bind directory, there are some minimal python bindings that were made with [pybind11](https://www.github.com/pybind/pybind11). The movielenstest uses these bindings to make a pureSVD recommender system for the [MovieLens 10M dataset](https://grouplens.org/datasets/movielens/10m) in the spirit of [ALSH paper](https://www.arxiv.org/pdf/1405.5869.pdf), which I am also not an author of.

It also provides a simple C++ implementation of and Python bindings for vanilla-LSH.  

The tables all expose these functions (for inner products and distance)
 * probe(query, adj) finds the item resulting in the largest inner product in the 'adj' highest ranked buckets
 * k_probe(k, query, adj) finds the k items resulting in the largest inner products in the 'adj' highest ranked buckets
 * probe_approx(query, c, adj) finds the first item with an inner product greater than c
 * k_probe_approx(k, query, c, adj) finds the first k items with inner products greater than c

## Usage

All needed libraries are included in external/. So, it should be decently portable. unit tests and synthetic data tests can be run using the Makefile. Running the movielenstest will not work since the data is not included in this repository. The examples in pyexamples and synthetic show how to use the library. The Python bindings can be compiled used "make binding".

## LSH for Maximum Inner Product Search

[Bachrach et al.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf) show that MIPS is reducible to Near Neighbor Search, but
the dimension must be increased. So, if the data is transformed, it is possible
to use LSH for MIPS! The transformation used by SimpleLSH, that maps the d-dimensional unit ball onto the (d+1)-dimensional unit sphere, is defined as

![\Large P(x)= \big(x, \sqrt{1 - ||x||_2^2}\big)](http://latex.codecogs.com/gif.latex?%5CLarge%20P%28x%29%3D%20%5Cbig%28x%2C%20%5Csqrt%7B1%20-%20%7C%7Cx%7C%7C_2%5E2%7D%5Cbig%29)

This makes vectors with large inner products also (probably) have similar hashes (on the unit sphere, the nearest vector is also the one that maximizes inner product). The following plots highlight this. Using SimpleLSH with 32 bits, we plot inner product vs hash similarity. We generated many random 50-dim vectors. For a random query q, the x-axis is the inner product with q. The y-axis is the hashes' similarity with q's hash. Similarity is defined as the number of matching bits. The black line is the best fit line using np.polyfit. As you can see, as the inner product increases, the similarity tends to increase as well.  

The source code to make these plots is in [synthetic/simple_lsh_plot.py](/synthetic/simple_lsh_plot.py).

![inner product vs hash similarity](/image/inner_product_vs_simplelsh_similarity_32bits.png)
