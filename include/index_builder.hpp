

#pragma once

#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <Eigen/Core>
#include <cmath>

#include "simple_lsh.hpp"

/*
 * Contains operations for "index-building" 
 * for nr-lsh. 
 *
 * Needs to be refactored.
 */



template<typename VectCont, typename IntCont>
auto max_norm(const VectCont& data, 
              const IntCont& partition) 
    -> typename VectCont::value_type::value_type
{
    /*
     * Finds the largest norm in a partition of data.
     */

    using Scalar = typename VectCont::value_type::value_type;

    Scalar max = -1;
    for(int64_t i = 0; i < partition.size(); ++i) {
        Scalar norm = data[partition[i]].norm();
        if(norm > max) {
            max = norm;
        }   
    }
    return max;
}


template<typename VectCont>
auto partitioner(const VectCont& dataset, int64_t m) 
    -> std::vector<std::vector<int64_t>>
{
    /*
     * Params:
     * :dataset: some matrix datset.
     * :m: the number of sub-datasets to partition.
     */
    
    Eigen::Index dimension = dataset[0].rows();

    std::vector<typename VectCont::value_type::value_type> norms(dataset.size()); 

    for(int64_t i = 0; i < dataset.size(); ++i) {
        norms[i] = dataset[i].norm();
    }

    std::vector<int64_t> ranking(dataset.size());
    std::iota(ranking.begin(), ranking.end(), 0);
    
    std::sort(ranking.begin(), 
              ranking.end(),
              [&](int64_t x, int64_t y) {
                return norms[x] < norms[y];
              });

    std::vector<std::vector<int64_t>> partitions(m, std::vector<int64_t>(0));

    int64_t current_partition = 0;
    for(int64_t i = 0; i < dataset.size(); ++i) {
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
    using T = typename VectCont::value_type;

    std::vector<std::vector<T>> normalized_dataset(partitions.size(), 
                                                   std::vector<T>(0));
    
    std::vector<typename T::value_type> U(partitions.size());
    
    for(int64_t p = 0; p < partitions.size(); ++p) {
        normalized_dataset[p] = std::vector<T>(partitions[p].size());
    }

    for(int64_t p = 0; p < partitions.size(); ++p) {

        auto Up = max_norm(dataset, partitions[p]); 
        
        U[p] = Up; // tables store the normalizer, so it can be undone after querying
        
        for(int64_t i = 0; i < partitions[p].size(); ++i) {
            auto normalized = dataset[partitions[p][i]] / Up;
            normalized_dataset[p][i] = normalized;//.push_back(normalized); 
        }
    }
    return std::make_pair(normalized_dataset, U);
}

template<typename PartCont>
auto simple_LSH_partitions(const PartCont& partitioned_dataset,
                            SimpleLSH hash)
    -> std::vector<std::vector<int64_t>>
{
    //SimpleLSH hash(bits, partitioned_dataset[0][0].rows());
    
    int m = partitioned_dataset.size();

    std::vector<std::vector<int64_t>> indices(m, std::vector<int64_t>(0));

    for(int64_t i = 0; i < m; ++i) {
        indices[i] = std::vector<int64_t>(partitioned_dataset[i].size());
    }
    
    for(int64_t j = 0; j < partitioned_dataset.size(); ++j) {
        for(int64_t p_idx = 0; p_idx < partitioned_dataset[j].size(); ++p_idx) 
        {
            indices[j][p_idx] = hash(partitioned_dataset[j][p_idx]);
        }
    }
    return indices;
}






