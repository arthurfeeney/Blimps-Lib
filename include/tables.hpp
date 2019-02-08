
#pragma once

#include <vector>
#include <unordered_map>
#include <utility>

#include "index_builder.hpp"
#include "simple_lsh.hpp"
#include "table.hpp"

template<typename Vect>
class Tables {
private:
    using KV = std::pair<Vect, int64_t>;

    SimpleLSH hash;
    std::vector<Table<Vect>> tables;

public:

    Tables(int64_t num_tables, int64_t bits, int64_t dim):
        hash(bits, dim),
        tables(num_tables, Table<Vect>(hash))
    {}
    
    template<typename Cont>
    void fill(const Cont& data)        
    {
        /*
         * partition data and put partitions into different tables.
         */
        auto parts = partitioner(data, tables.size());
        
        auto normal_data_and_U = normalizer(data, parts);
        auto normal_data = normal_data_and_U.first;
        auto U = normal_data_and_U.second;

        auto indices = simple_LSH_partitions(normal_data, hash); 

        std::vector<std::vector<Vect>> parted_data(parts.size());

        for(int64_t p = 0; p < parts.size(); ++p) {
            for(int64_t i = 0; i < parts[p].size(); ++i) {
                parted_data.at(p).push_back(data[parts[p][i]]);
            }
        }

        for(int64_t p = 0; p < tables.size(); ++p) {
            tables[p].fill(parted_data[p],
                           indices[p],
                           parts[p],
                           U[p]);
        }
    }

    std::pair<bool, KV> MIPS(const Vect& q) const {
        /*
         * Search partitions in each table for MIP with q.
         */
        std::vector<KV> x(0);
        for(auto t : tables) {
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
