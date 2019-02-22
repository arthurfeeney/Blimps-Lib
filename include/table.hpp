
#pragma once

#include <vector>
#include <list>
#include <utility>
#include <cmath>
#include <numeric>
#include <queue>
#include <omp.h>

#include <iostream>

#include "simple_lsh.hpp"
#include "stats.hpp"

template<typename Vect>
class Table {
private:
    using KV = std::pair<Vect, int64_t>;
    int64_t num_buckets;
    std::vector<std::list<KV>> table;
    SimpleLSH hash;
    typename Vect::value_type normalizer; // this partitions Up normalizer

public:
    Table(SimpleLSH hash, int64_t num_buckets):
        num_buckets(num_buckets),
        table(num_buckets),
        hash(hash),
        normalizer(0)
    {}

    //template<template<typename Sub> typename Cont>
    void fill(const std::vector<Vect>& normalized_partition, 
              const std::vector<int64_t>& indices,
              const std::vector<int64_t>& ids,
              typename Vect::value_type Up)
    {
        for(size_t i = 0; i < indices.size(); ++i) {
            KV to_insert = std::make_pair(normalized_partition.at(i), 
                                          ids.at(i));
            // use % to make sure all indices in range. 
            table.at(indices.at(i) % table.size()).push_back(to_insert);
        }    
        normalizer = Up;
    }


    std::pair<bool, KV> MIPS(const Vect& q) {
        int64_t idx = hash(q) % table.size();

        if(table.at(idx).size() == 0) {
            return std::make_pair(false, KV());
        }

        KV max = *table.at(idx).begin();
        double big_dot = q.dot(max.first);

        for(auto iter = ++table.at(idx).begin(); 
                 iter != table.at(idx).end(); 
                 ++iter) {
            KV current = *iter;
            double dot = q.dot(current.first);
            if(dot > big_dot) {
                big_dot = dot;
                max = current;
            }
        }

        return std::make_pair(true, max);
    }


    std::pair<bool, KV> probe(const Vect& q, int64_t n_to_probe) {

        int64_t idx = hash(q) % table.size();
        std::vector<int64_t> rank = probe_ranking(idx);

        KV max = std::make_pair(Vect(1), -1); // initialize to impossible val
        double big_dot = -9999999;

        for(int64_t r = 0; r < n_to_probe; ++r) {

            for(const auto& current : table.at(rank.at(r))) {

                double dot = q.dot(current.first);

                if(dot > big_dot) {
                    big_dot = dot;
                    max = current;
                    std::cout << "found bigger" << '\n';
                }
            }            
        }

        if(max.second < 0) { // no large inner products were found.
            return std::make_pair(false, KV());
        }

        return std::make_pair(true, max);
    }


    int64_t hamming_distance(int64_t x, int64_t y) const {
        int64_t dist = 0;
        for(int64_t z = x ^ y; z > 0; z >>= 1) {
            dist += x & 1;
        }
        return dist;
    }


    int64_t sim(int64_t idx, int64_t other) const {
        double l = static_cast<double>(hamming_distance(idx, other));
        double L = static_cast<double>(hash.bit_count());

        return normalizer * std::cos(3.14159265358979 * (1 - (l / L)));
    }


    std::vector<int64_t> probe_ranking(int64_t idx) const {
        std::vector<int64_t> rank(table.size(), 0);
        std::iota(rank.begin(), rank.end(), 0);

        // want things with a small hamming distance to be in the front.
        // so sort in ascedning order 
        std::sort(rank.begin(), rank.end(), 
                  [&](int64_t x, int64_t y) {
                    return sim(idx, x) < sim(idx, y);       
                  });
        
        return rank;
    }

    void print_stats() {
        std::vector<size_t> bucket_sizes(table.size(), 0);
        
        #pragma omp parallel for
        for(size_t i = 0; i < bucket_sizes.size(); ++i) {
            bucket_sizes.at(i) = table.at(i).size();
        }

        size_t max = *std::max_element(bucket_sizes.begin(),
                                       bucket_sizes.end());
        size_t min = *std::min_element(bucket_sizes.begin(),
                                       bucket_sizes.end());

        auto var = NR_stats::variance(bucket_sizes); 
        auto stdev = std::sqrt(var);

        std::cout << "\tmean:  " << NR_stats::mean(bucket_sizes) << '\n';
        std::cout << "\tmax:   " << max << '\n';
        std::cout << "\tmin:   " << min << '\n';
        std::cout << "\tvar:   " << var << '\n';
        std::cout << "\tstdev: " << stdev << '\n';
    }
};
