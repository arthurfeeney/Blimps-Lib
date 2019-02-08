
#pragma once

#include <vector>
#include <utility>

#include "tables.hpp"
#include "simple_lsh.hpp"


/*
 * instead of using multi-probe, this uses multiple tables 
 * Multi-probe for NR-LSH appears to be much harder to implement. 
 * As each sub-table is normalized differently. 
 * Will implement multi-probe in the future. 
 */


template<typename Vect>
class NR_MultiTable {
private:
    using KV = std::pair<Vect, int64_t>;

    std::vector<Tables<Vect>> nr_tables;

public:
    NR_MultiTable(int64_t num_tables, 
           int64_t num_partitions, 
           int64_t bits, 
           int64_t dim):
        nr_tables(num_tables, Tables<Vect>(0,0,0))//num_partitions,
                                           //bits, dim))
    {
        for(auto& t : nr_tables) {
            t = Tables<Vect>(num_partitions, bits, dim);
        }
    }

    template<typename Cont>
    void fill(const Cont& data)
    {
        for(auto& tables : nr_tables) {
            tables.fill(data);
        }
    }

    std::pair<bool, KV> MIPS(const Vect& q) {
        std::vector<KV> x(0);
        for(auto t : nr_tables) {
            std::pair<bool, KV> xj = t.MIPS(q);
            if(xj.first) {
                x.push_back(xj.second);
            }
        }
        if(x.size() == 0) {
            return std::make_pair(false, KV());
        }
        
        double big_dot = q.dot(x[0].first);
        KV max = x[0];

        for(int i = 1; i < x.size(); ++i) {
            double dot = q.dot(x[i].first);
            if(dot > big_dot) {
                big_dot = dot;
                max = x[i];
            }
        }
        return std::make_pair(true, max);
    } 

};
