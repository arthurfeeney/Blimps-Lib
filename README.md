

# A (incomplete) C++17 Implementation of Norm-Ranging Locality Sensitive Hashing

##### [Original paper](https://papers.nips.cc/paper/7559-norm-ranging-lsh-for-maximum-inner-product-search.pdf) of which I am not an author.   
 

This implementation is header-only at the moment, so it is simple to setup. The only dependency is [Eigen](https://www.eigen.tuxfamily.org/index.php?title=Main_Page). I also included some minimal python bindings, which were made with [pybind11](https://www.github.com/pybind/pybind11), in the "bind/" directory. The movielenstest includes some little "toy" tests with the [MovieLens 10M dataset](https://grouplens.org/datasets/movielens/10m) (which is not included in this repositoty!) that was also used by the original [ALSH paper](https://www.arxiv.org/pdf/1405.5869.pdf), which I am also not an author of.

### Some Todo:

* ~~statistics on bucket distributions and partitions.~~
    * ~~make python bindings~~
    * Add stats specfically for non-empty buckets
* refactor index builder (C++17 -> C++11 changes made some things ugly)
    * Some operations redundant too?
* ~~Implement multi-probe described in paper (and add python bindings for it)~~
* much more thorough movielens experiments that mimick the ALSH paper. 
* ~~supprt k-approximate MIPS.~~
* Add statistics to check how many comparisons are made during search.
    * add an optional argument to set whether or not to track?`
    * MUST ACCOUNT FOR THEADING!
* Fix poor-ish bucket distributions (this may just be how it is?)
* Better use of OpenMP. 
    * can and should definitely be used during MIPS...
    * refactor for this...
* refactor stats so return type depends on Cont::value_type.

* make table::look_in go through entire bucket, not stop at first MIP.
