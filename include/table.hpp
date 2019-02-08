
#pragma once

#include <vector>
#include <list>
#include <utility>
#include <cmath>

#include "simple_lsh.hpp"

template<typename Vect>
class Table {
private:
    using KV = std::pair<Vect, int64_t>;
    std::vector<std::list<KV>> table;
    SimpleLSH hash;
    typename Vect::value_type normalizer; // this partitions Up normalizer

public:
    Table(SimpleLSH hash):
        table(std::pow(2, hash.bit_count())),
        hash(hash),
        normalizer(0)
    {}

    //template<template<typename Sub> typename Cont>
    void fill(const std::vector<Vect>& normalized_partition, 
              const std::vector<int64_t>& indices,
              const std::vector<int64_t>& ids,
              typename Vect::value_type Up)
    {
        for(int64_t i = 0; i < indices.size(); ++i) {
            KV to_insert = std::make_pair(normalized_partition[i], ids[i]);
            // use % to make sure all indices in range. 
            table[indices[i] % table.size()].push_back(to_insert);
        }    
        normalizer = Up;
    }


    std::pair<bool, KV> MIPS(const Vect& q) {
        int64_t idx = hash(q) % table.size();

        if(table[idx].size() == 0) {
            return std::make_pair(false, KV());
        }

        KV max = *table[idx].begin();
        double big_dot = q.dot(max.first);

        for(auto iter = ++table[idx].begin(); 
                 iter != table[idx].end(); 
                 ++iter) 
        {
            KV current = *iter;
            double dot = q.dot(current.first);
            if(dot > big_dot) {
                big_dot = dot;
                max = current;
            }
        }
        return std::make_pair(true, max);
    }
};
