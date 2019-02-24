
#pragma once

#include <iostream>
#include <vector>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <omp.h>

#include "index_builder.hpp"
#include "simple_lsh.hpp"
#include "table.hpp"

namespace nr {

template<typename Vect>
class Tables {
private:
    using Component = typename Vect::value_type;
    using KV = std::pair<Vect, int64_t>;

    int64_t num_partitions;
    int64_t num_buckets;
    SimpleLSH hash;
    std::vector<Table<Vect>> tables;
    std::vector<typename Vect::value_type> normalizers;

public:
    Tables():num_partitions(0), hash(0,0) {} // empty constructor. 

    Tables(int64_t num_partitions, 
           int64_t bits, 
           int64_t dim, 
           int64_t num_buckets):
        num_partitions(num_partitions),
        num_buckets(num_buckets),
        hash(bits, dim),
        tables(num_partitions, Table<Vect>(hash, num_buckets))
    {}
    
    template<typename Cont>
    void fill(const Cont& data, bool is_normalized)        
    {
        /*
         * partition data and put partitions into different tables.
         */
        auto parts = partitioner(data, tables.size()); 
        auto normal_data_and_U = normalizer(data, parts);
        auto normal_data = normal_data_and_U.first;
        normalizers = normal_data_and_U.second;
        auto indices = simple_LSH_partitions(normal_data, hash); 

        std::vector<std::vector<Vect>> parted_data(parts.size());

        for(size_t p = 0; p < parts.size(); ++p) {
            for(size_t i = 0; i < parts.at(p).size(); ++i) {
                parted_data.at(p).push_back(data.at(parts.at(p).at(i)));
            }
        }

        #pragma omp parallel for
        for(size_t p = 0; p < tables.size(); ++p) {
            tables.at(p).fill(parted_data.at(p),
                              indices.at(p),
                              parts.at(p),
                              normalizers.at(p), 
                              is_normalized);
        }
    }

    std::pair<bool, KV> MIPS(const Vect& q) const {
        /*
         * Specific for MultiTable.
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
        
        return std::make_pair(true, *std::max_element(x.begin(), x.end(),
                              [&](KV y, KV z) {
                                return q.dot(y.first) < q.dot(z.first);
                              }));
    }

    std::pair<bool, KV> probe(const Vect& q, int64_t n_to_probe) {
        std::vector<KV> x(0);

        //for(auto& table : tables) {
        for(auto it = tables.begin(); it != tables.end(); ++it) {
            std::pair<bool, KV> xj = (*it).probe(q, n_to_probe);
            if(xj.first) {
                x.push_back(xj.second);
            }
        }

        KV ret = *std::max_element(x.begin(), x.end(), 
                                   [&](KV y, KV z) {
                                     return q.dot(y.first) < q.dot(z.first);
                                   });
                                    
        // if x.size() == 0, nothing was found so it should return false
        return std::make_pair(x.size() != 0, ret);
    }

    std::vector<std::vector<int64_t>> sub_tables_rankings(int64_t idx) {
        std::vector<std::vector<int64_t>> rankings(tables.size());

        for(auto it = tables.begin(); it < tables.end(); ++it) {
            rankings[it - tables.begin()] = (*it).probe_ranking(idx);
        }
        
        return rankings;
    }

    std::pair<bool, KV> probe_approx(const Vect& q, Component c) {
        auto rankings = sub_tables_rankings(hash(q) % num_buckets);

        // iterate column major until dot(q, x) > c is found.
        for(size_t col = 0; col < num_buckets; ++col) {
            for(size_t t = 0; t < rankings.size(); ++t) {
                auto found = tables[t].look_in(rankings[t][col], q, c);
                if(found.first) {
                    return found;
                }
            }
        }

        return std::make_pair(false, KV());
    } 

    std::pair<bool, std::vector<KV>> 
    k_probe_approx(int64_t k, const Vect& q, Component c) {
        auto rankings = sub_tables_rankings(hash(q) % num_buckets);

        std::vector<KV> vects(0);

        for(size_t col = 0; col < num_buckets; ++col) {
            for(size_t t = 0; t < rankings.size(); ++t) {
                auto found = tables[t].look_in(rankings[t][col], q, c);
                if(found.first) {
                    vects.push_back(found.second);
                }
                if(vects.size() == k) {
                    return std::make_pair(true, vects);
                }
            }
        }
        
        return std::make_pair(vects.size() != 0, vects);
    }

    void print_stats() {
        int64_t table_id = 1;
        for(auto& table : tables) {
            std::cout << "table " << table_id << '\n';
            table.print_stats(); 
            ++table_id;
        }
    }
};

}
