

# A (very early/incomplete) C++11 Implementation of NR-LSH

##### [Original paper](https://www.arxiv.org/pdf/1410.5410.pdf) of which I am not an author.   
 

This implementation is header-only at the moment, so it is simple to use. The only dependency is [Eigen](https://www.eigen.tuxfamily.org/index.php?title=Main_Page). I also included some minimal python bindings, which were made with [pybind11](https://www.github.com/pybind/pybind11), in the "bind/" directory. The movielenstest includes some little "toy" tests with the movielens dataset that was also used by the original [ALSH paper](https://www.arxiv.org/pdf/1405.5869.pdf), which I am also not an author of.

### Todo:

* statistics on bucket distributions and partitions.
    * make python bindings
* refactor index builder (C++17 -> C++11 changes made some things ugly)
    * Some operations redundant too?
* Implement multi-probe described in paper (and add python bindings for it)
* much more thorough movielens experiments that mimick the ALSH paper. 
