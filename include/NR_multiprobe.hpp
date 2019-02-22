
#pragma once

#include <vector>
#include <utility>

#include "tables.hpp"
#include "simple_lsh.hpp"
#include "stats.hpp"

/*
 * Multiprobe implementation of NR-LSH
 */


template<typename Vect>
class NR_MultiProbe {
private:
    using KV = std::pair<Vect, int64_t>;

    Tables<Vect> probe_table;

public:
    NR_MultiProbe(int64_t num_partitions, 
                  int64_t bits, 
                  int64_t dim,
                  int64_t num_buckets):
        probe_table(num_partitions, bits, dim, num_buckets)
    {}

    template<typename Cont>
    void fill(const Cont& data) {
        probe_table.fill(data);
    }

    std::pair<bool, KV> probe(const Vect& q, int64_t n_to_probe) {
        return probe_table.probe(q, n_to_probe); 
    }

    void print_stats() {
        probe_table.print_stats();
    }
};
