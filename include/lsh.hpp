
#pragma once

#include <iostream>
#include <limits>
#include <list>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "kv_comparator.hpp"
#include "multiprobe.hpp"
#include "simple_lsh.hpp"
#include "stat_tracker.hpp"
#include "stats/stats.hpp"
#include "stats/topk.hpp"
#include "tables.hpp"

/*
 * Multiprobe implementation of Locality Sensitive Hashing
 */

namespace nr {

template <typename Vect> class LSH_MultiProbe { //: MultiProbe<Vect> {
private:
  using Component = typename Vect::value_type;
  using KV = std::pair<Vect, int64_t>;

  std::vector<std::list<KV>> table;
  int64_t dim;
  SimpleLSH<Component> hash_function;

  static double sim(size_t x, size_t y, size_t num_buckets) {
    // finds the number of bits in x and y that are the same.
    return static_cast<double>(
        stats::same_bits(x, y, std::floor(std::log2(num_buckets)) + 1));
  }

  static std::vector<size_t> probe_ranking(size_t idx, size_t num_buckets) {
    std::vector<size_t> rank(num_buckets, 0);
    std::iota(rank.begin(), rank.end(), 0);
    // sort in descending order. Most similar in the front.
    std::sort(rank.begin(), rank.end(), [&](size_t x, size_t y) {
      return sim(idx, x, num_buckets) > sim(idx, y, num_buckets);
    });
    return rank;
  }

public:
  LSH_MultiProbe(int64_t bits, int64_t dimension, size_t num_buckets)
      : table(num_buckets, std::list<KV>()), dim(dimension),
        hash_function(bits, dimension) {}

  LSH_MultiProbe(const LSH_MultiProbe &other) {
    tables(other.tables);
    dim = other.dim;
    hash_function(other.hash_function);
  }

  template <typename Cont>
  void fill(const Cont &data, bool is_normalized = false) {
    for (const KV &datum : data) {
      const size_t idx = hash_function.hash_max(datum.first, table.size());
      table.at(idx).push_front(datum);
    }
  }

  std::pair<std::optional<KV>, StatTracker> probe(const Vect &q, int64_t adj) {
    /*
     * Probe adj buckets. Return vector closest to q.
     */

    StatTracker tracker;

    const size_t idx = hash_function.hash_max(q, table.size());
    const std::vector<size_t> ranking(probe_ranking(idx, table.size()));

    KV neighbor = KV();
    Component min_dist = std::numeric_limits<Component>::max();

    for (size_t i = 0; i < adj; ++i) {
      tracker.incr_buckets_probed();
      const size_t probe_idx = ranking.at(i);
      const std::list<KV> &bucket = table.at(probe_idx);
      for (const KV &x : bucket) {
        tracker.incr_comparisons();
        Component dist = (q - x.first).norm();
        if (dist < min_dist) {
          neighbor = x;
          min_dist = dist;
        }
      }
    }
    return std::make_pair(std::make_optional(neighbor), tracker);
  }

  std::pair<std::optional<std::vector<KV>>, StatTracker>
  k_probe(int64_t k, const Vect &q, size_t adj) {}

  std::pair<std::optional<KV>, StatTracker>
  probe_approx(const Vect &q, Component c, int64_t adj) {}

  std::pair<std::optional<std::vector<KV>>, StatTracker>
  k_probe_approx(int64_t k, const Vect &q, Component c, size_t adj) {}

  KV find_max_inner(const Vect &q) {}

  void print_stats() {}

  std::vector<std::list<KV>> data() { return table; }

  size_t num_tables() const { return 1; }
}; // namespace nr

} // namespace nr
