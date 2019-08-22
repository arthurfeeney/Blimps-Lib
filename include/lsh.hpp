
#pragma once

#include <iostream>
#include <limits>
#include <list>
#include <numeric>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "kv_comparator.hpp"
#include "multiprobe.hpp"
#include "sign_lsh.hpp"
#include "stat_tracker.hpp"
#include "stats/stats.hpp"
#include "stats/topk.hpp"
#include "tables.hpp"

/*
 * Implementation of Mulitprobe Locality Sensitive Hashing
 * Uses 1 table, but probes multiple buckets that are likely to contain
 * values near the query.
 * Fast but has poor recall.
 */

namespace nr {

template <typename Vect, typename Hash = SignLSH<typename Vect::value_type>>
class LSH_MultiProbe : public MultiProbe<Vect> {
private:
  using Component = typename Vect::value_type;
  using KV = std::pair<Vect, int64_t>;

  std::vector<std::list<KV>> table;
  int64_t dim;
  // defaults to SignLSH, but user can pass in a PStableLSH.
  Hash hash_function;

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

  std::vector<size_t> rank(const Vect &q, size_t max_hash, int64_t adj) const {
    // return indices of the top 'adj' ranked buclets.
    // not static because it uses the hash function.
    const size_t idx = hash_function.hash_max(q, max_hash);
    std::vector<size_t> ranks = probe_ranking(idx, max_hash);
    return std::vector<size_t>(ranks.begin(), ranks.begin() + adj);
  }

  void manage_topk(std::vector<std::pair<KV, Component>> &topk, int64_t k,
                   const Vect &query, KV pos) {
    /*
     * Clunky function to track the topk vectors closest to q.
     */
    if (k < 1) {
      throw std::runtime_error(
          "LSH_MultiProbe::manage_topk, k must be positive");
    }
    Component dist = (query - pos.first).norm();
    topk.push_back(std::make_pair(pos, dist));
    // sorting by distance should be fast.
    // don't have to recompute distances.
    // sort nearest to most distant. Removing last element should be faster
    std::sort(topk.begin(), topk.end(),
              [](std::pair<KV, Component> x, std::pair<KV, Component> y) {
                return x.second < y.second;
              });
    if (topk.size() >= static_cast<size_t>(k + 1)) {
      // remove most distant element.
      topk.pop_back();
    }
  }

public:
  LSH_MultiProbe(int64_t bits, int64_t dimension, size_t num_buckets)
      : table(num_buckets, std::list<KV>()), dim(dimension),
        hash_function(bits, dimension) {}

  LSH_MultiProbe(int64_t bits, int64_t dimension)
      : table(static_cast<size_t>(std::pow(2, bits)), std::list<KV>()),
        dim(dimension), hash_function(bits, dimension) {}

  LSH_MultiProbe(const LSH_MultiProbe &other) {
    tables(other.tables);
    dim = other.dim;
    hash_function(other.hash_function);
  }

  template <typename Cont>
  void fill(const Cont &data, bool is_normalized = false) {
    int64_t id = 0;
    for (const auto &datum : data) {
      const size_t idx = hash_function.hash_max(datum, table.size());
      table.at(idx).push_front(std::make_pair(datum, id));
      ++id;
    }
  }

  template <typename Op>
  void iter_table(const Vect &q, int64_t adj, StatTracker &tracker, Op op) {
    /*
     * The probe functions iterate through all adj buckets.
     * Using this function avoids a lot of code duplication.
     * The specific operation used by the probe function is passed in.
     */
    for (const size_t &probe_idx : rank(q, table.size(), adj)) {
      tracker.incr_buckets_probed();
      for (const KV &x : table.at(probe_idx)) {
        tracker.incr_comparisons();
        op(q, x);
      }
    }
  }

  std::pair<std::optional<KV>, StatTracker> probe(const Vect &q, int64_t adj) {
    /*
     * Probe adj buckets. Return vector closest to q.
     */
    StatTracker tracker;
    KV neighbor = KV();
    Component min_dist = std::numeric_limits<Component>::max();
    // search through the highest ranked buckets for neighbor.
    iter_table(q, adj, tracker,
               [&neighbor, &min_dist](const Vect &q, const KV &x) {
                 Component dist = (q - x.first).norm();
                 if (dist < min_dist) {
                   neighbor = x;
                   min_dist = dist;
                 }
               });
    if (min_dist == std::numeric_limits<Component>::max()) {
      // only probed empty buckets, so min-dist didn't change.
      return {std::nullopt, tracker};
    }
    return {std::make_optional(neighbor), tracker};
  }

  std::pair<std::optional<std::vector<KV>>, StatTracker>
  k_probe(int64_t k, const Vect &q, size_t adj) {
    /*
     * probe adj buckets. Return the k probed vectors that are closest to q
     * output is most distant to nearest.
     */
    StatTracker tracker;
    // topk is set sorted by distance to query.
    // distanct objects are at the front. close at at end.
    std::vector<std::pair<KV, Component>> topk(0);

    // sketchy? Passing this allows manage_topk to be called.
    iter_table(q, adj, tracker, [this, &topk, &k](const Vect &q, const KV &x) {
      this->manage_topk(topk, k, q, x);
    });

    return proc_k_probe_output(topk, tracker);
  }

  std::pair<std::optional<std::vector<KV>>, StatTracker>
  proc_k_probe_output(std::vector<std::pair<KV, Component>> &topk,
                      const StatTracker &tracker) const {
    if (topk.size() == 0) {
      return {std::nullopt, tracker};
    }
    // sort output so it is distant to nearest
    std::sort(
        topk.begin(), topk.end(),
        [](const std::pair<KV, Component> &x,
           const std::pair<KV, Component> &y) { return x.second > y.second; });

    // copy topk into vector of proper return type
    std::vector<KV> topk_out(topk.size());
    std::generate(topk_out.begin(), topk_out.end(), [&topk, n = -1]() mutable {
      ++n;
      return topk.at(n).first;
    });
    return {std::make_optional(topk_out), tracker};
  }

  std::pair<std::optional<KV>, StatTracker>
  probe_approx(const Vect &q, Component c, int64_t adj) {
    /*
     * Probe adj buckets. Return the first vector within distance c of
     * the query q.
     */
    StatTracker tracker;

    for (const size_t &probe_idx : rank(q, table.size(), adj)) {
      tracker.incr_buckets_probed();
      for (const KV &x : table.at(probe_idx)) {
        tracker.incr_comparisons();
        if ((q - x.first).norm() <= c) {
          return {std::make_optional(x), tracker};
        }
      }
    }
    return {std::nullopt, tracker};
  }

  std::pair<std::optional<std::vector<KV>>, StatTracker>
  k_probe_approx(int64_t k, const Vect &q, Component c, size_t adj) {
    /*
     * return the first k neighbors in adj found while searching through
     * the highest ranked buckets.
     * Output is most distant found to nearest
     */
    if (k < 1) {
      throw std::runtime_error(
          "LSH_MultiProbe::k_probe_approx, k must be positive");
    }

    StatTracker tracker;

    std::vector<KV> topk(0);
    for (const size_t &probe_idx : rank(q, table.size(), adj)) {
      tracker.incr_buckets_probed();
      for (const KV &x : table.at(probe_idx)) {
        tracker.incr_comparisons();
        if ((q - x.first).norm() <= c) {
          topk.push_back(x);
          if (topk.size() == static_cast<size_t>(k)) {
            return proc_k_probe_approx_output(q, topk, tracker);
          }
        }
      }
    }
    return proc_k_probe_approx_output(q, topk, tracker);
  }

  std::pair<std::optional<std::vector<KV>>, StatTracker>
  proc_k_probe_approx_output(const Vect &q, std::vector<KV> &topk,
                             StatTracker &tracker) const {
    /*
     * Function to format the topk items found.
     */
    if (topk.size() == 0) {
      return {std::nullopt, tracker};
    }
    // should be sorted distant to nearest.
    std::sort(topk.begin(), topk.end(), [&q](KV x, KV y) {
      return (x.first - q).norm() > (y.first - q).norm();
    });
    return {std::make_optional(topk), tracker};
  }

  void print_stats() {}

  std::vector<std::list<KV>> data() { return table; }

  size_t num_tables() const { return 1; }
};

} // namespace nr
