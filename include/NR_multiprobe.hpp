
#pragma once

#include <vector>
#include <utility>

#include "tables.hpp"
#include "simple_lsh.hpp"
#include "stats.hpp"

/*
 * Multiprobe implementation of NR-LSH
 */

namespace nr {

class MultiProbe {

};

template<typename Vect>
class NR_MultiProbe {
private:
    using Component = typename Vect::value_type;
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
    void fill(const Cont& data, bool is_normalized) {
        probe_table.fill(data, is_normalized);
    }

    std::pair<bool, KV> probe(const Vect& q, int64_t n_to_probe) {
        return probe_table.probe(q, n_to_probe); 
    }

    std::pair<bool, KV> probe_approx(const Vect& q, Component c) {
        return probe_table.probe_approx(q, c);
    }

    std::pair<bool, std::vector<KV>> 
    k_probe_approx(int64_t k, const Vect& q, double c) {
        return probe_table.k_probe_approx(k, q, c); 
    }

    void print_stats() {
        probe_table.print_stats();
    }
};

}
