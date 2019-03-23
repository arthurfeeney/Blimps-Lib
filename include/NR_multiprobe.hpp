
#pragma once

#include <gsl/gsl>
#include <limits>
#include <utility>
#include <vector>

#include "simple_lsh.hpp"
#include "stat_tracker.hpp"
#include "stats.hpp"
#include "tables.hpp"

/*
 * Multiprobe implementation of NR-LSH
 */

namespace nr {

template <typename Vect> class NR_MultiProbe {
private:
  using Component = typename Vect::value_type;
  using KV = std::pair<Vect, int64_t>;

  Tables<Vect> probe_table;

public:
  NR_MultiProbe(int64_t num_partitions, int64_t bits, int64_t dim,
                size_t num_buckets)
      : probe_table(num_partitions, bits, dim, num_buckets) {}

  template <typename Cont> void fill(const Cont &data, bool is_normalized) {
    // if is_normalized is true, then the data input to the table is
    // normalized, otherwise, it is inserted as the original unnormalised
    // value.
    if (data.size() < probe_table.size()) {
      throw std::runtime_error("NR_MultiProbe::fill."
                               "data.size() < num partitions");
    }
    probe_table.fill(data, is_normalized);
  }

  std::pair<bool, KV> probe(const Vect &q, int64_t n_to_probe) {
    // searches through the first n_to_probe most likely buckets
    // stat tracker isn't needed for this because it searches all the buckets.
    return probe_table.probe(q, n_to_probe);
  }

  std::tuple<bool, KV, StatTracker> probe_approx(const Vect &q, Component c) {
    // searches until it finds some x with dot(x, q) > c
    return probe_table.probe_approx(q, c);
  }

  std::tuple<bool, std::vector<KV>, StatTracker>
  k_probe_approx(int64_t k, const Vect &q, double c) {
    // searches until it finds k vectors, x where all x have dot(x, q) > c
    return probe_table.k_probe_approx(k, q, c);
  }

  KV find_max_inner(const Vect &q) {
    /*
     * Finds the true maximum inner product in the dataset, not an
     * approximate one.
     * This exists in case the user does not have access to the
     * original dataset for some reason.
     * It is also multi-threaded, so it may be better.
     */
    return probe_table.MIPS(q).second;
  }

  void print_stats() { probe_table.print_stats(); }
};

} // namespace nr
