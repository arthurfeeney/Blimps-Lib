

# A (incomplete) C++17 Implementation of Norm-Ranging Locality Sensitive Hashing

##### [Original paper](https://papers.nips.cc/paper/7559-norm-ranging-lsh-for-maximum-inner-product-search.pdf) of which I am not an author.   
 

This implementation is header-only at the moment, so it is simple to setup. The only dependency is [Eigen](https://www.eigen.tuxfamily.org/index.php?title=Main_Page). I also included some minimal python bindings, which were made with [pybind11](https://www.github.com/pybind/pybind11), in the "bind/" directory. The movielenstest includes some little "toy" tests with the [MovieLens 10M dataset](https://grouplens.org/datasets/movielens/10m) (which is not included in this repositoty!) that was also used by the original [ALSH paper](https://www.arxiv.org/pdf/1405.5869.pdf), which I am also not an author of.

Tests use [Catch2](https://githubcom/catchorg/Catch2). Like Eigen, this is header-only. So it shouldn't be too terrible to run my tests. 

### Some Todo:

* Track number of buckets and partitions queried in addition to comparisons.
* Make better use of OpenMP. 
* Bigger movielens test.
* Refactor index builder
* Change base data structure from vector. 
* Figure out why a vector w/ > unit length is being passed to SimpleLSH.



