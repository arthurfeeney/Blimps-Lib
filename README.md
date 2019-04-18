

## A C++17 Implementation of Norm-Ranging Locality Sensitive Hashing

##### [Original paper](https://papers.nips.cc/paper/7559-norm-ranging-lsh-for-maximum-inner-product-search.pdf) of which I am not an author.   
 

This implementation is header-only at the moment. Matrix operations use [Eigen](https://www.eigen.tuxfamily.org/index.php?title=Main_Page). It uses boost/multiprecision to allow for very long hash codes.
Tests use [Catch2](https://githubcom/catchorg/Catch2). This is header-only as well. So, it shouldn't be too terrible to run my tests.  

I also included some minimal python bindings, which were made with [pybind11](https://www.github.com/pybind/pybind11), in the "bind/" directory. The movielenstest uses these bindings to make a little recommender system with the [MovieLens 10M dataset](https://grouplens.org/datasets/movielens/10m) (which is not included in this repositoty!) that was also used by the original [ALSH paper](https://www.arxiv.org/pdf/1405.5869.pdf), which I am also not an author of.





