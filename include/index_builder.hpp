
#pragma once

#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <Eigen/Core>
#include <cmath>

#include "simple_lsh.hpp"
#include "index_builder.hpp"

/*
 * Contains operations for "index-building" 
 * for nr-lsh. 
 *
 * Needs to be refactored, badly.
 */

namespace nr {

template<typename VectCont, typename IntCont>
auto max_norm(const VectCont& data, 
              const IntCont& partition) 
{
    /*
     * Finds the largest norm in a partition of data.
     */

    using Scalar = typename VectCont::value_type::value_type;

    Scalar max = -1;
    for(size_t i = 0; i < partition.size(); ++i) {
        Scalar norm = data.at(partition.at(i)).norm();
        if(norm > max) {
            max = norm;
        }   
    }

    return max;
}


template<typename VectCont>
std::vector<std::vector<int64_t>>
partitioner(const VectCont& dataset, int64_t m) 
{
    /*
     * Params:
     * :dataset: some matrix datset.
     * :m: the number of sub-datasets to partition.
     */

    using VectComponent = typename VectCont::value_type::value_type;
    
    std::vector<VectComponent> norms(dataset.size()); 

    for(size_t i = 0; i < dataset.size(); ++i) {
        norms.at(i) = dataset.at(i).norm();
    }

    std::vector<int64_t> ranking(dataset.size());
    std::iota(ranking.begin(), ranking.end(), 0);
    
    std::sort(ranking.begin(), 
              ranking.end(),
              [norms](int64_t x, int64_t y) {
                return norms.at(x) < norms.at(y);
              });

    std::vector<std::vector<int64_t>> partitions(m, std::vector<int64_t>(0));

    int64_t current_partition = 0;
    for(size_t i = 0; i < dataset.size(); ++i) {
        if(i != 0 && (i % (dataset.size() / m) == 0)) 
        {
            ++current_partition;
        } 

        if(current_partition < m) {
            partitions.at(current_partition).push_back(ranking.at(i));
        }
        else {
            // put any overflow into the last partition. 
            partitions.at(m-1).push_back(ranking.at(i));
        }
    }
    return partitions;
}


template<typename VectCont>
std::pair<std::vector<std::vector<typename VectCont::value_type>>,
          std::vector<typename VectCont::value_type::value_type>>
normalizer(const VectCont& dataset, 
           const std::vector<std::vector<int64_t>>& partitions) 
{
    using Vect = typename VectCont::value_type;

    std::vector<std::vector<Vect>> normalized_dataset(partitions.size(), 
                                                   std::vector<Vect>(0));
    
    std::vector<typename Vect::value_type> U(partitions.size());
    
    // initialize partitions of normalized dataset.
    for(size_t p = 0; p < partitions.size(); ++p) {
        normalized_dataset.at(p) = std::vector<Vect>(partitions.at(p).size());
    }

    for(size_t p = 0; p < partitions.size(); ++p) {

        auto Up = max_norm(dataset, partitions.at(p)); 
        
        U.at(p) = Up; 
        
        for(size_t i = 0; i < partitions.at(p).size(); ++i) {
            auto normalized = dataset.at(partitions.at(p).at(i)) / Up;
            normalized_dataset.at(p).at(i) = normalized; 
        }
    }
    return std::make_pair(normalized_dataset, U);
}

template<typename PartCont>
std::vector<std::vector<int64_t>>
simple_LSH_partitions(const PartCont& partitioned_dataset, SimpleLSH hash)
{
    //SimpleLSH hash(bits, partitioned_dataset[0][0].rows());
    
    size_t m = partitioned_dataset.size();

    std::vector<std::vector<int64_t>> indices(m, std::vector<int64_t>(0));

    for(size_t i = 0; i < m; ++i) {
        indices.at(i) = std::vector<int64_t>(partitioned_dataset.at(i).size());
    }
    
    for(size_t j = 0; j < partitioned_dataset.size(); ++j) {
        for(size_t p_idx = 0; p_idx < partitioned_dataset.at(j).size(); ++p_idx) 
        {
            indices.at(j).at(p_idx) = hash(partitioned_dataset.at(j).at(p_idx));
        }
    }
    return indices;
}

}




