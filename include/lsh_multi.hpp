
#pragma once

#include <boost/multiprecision/cpp_int.hpp>
#include <iostream>
#include <limits>
#include <list>
#include <numeric>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fast_sim.hpp"
#include "kv_comparator.hpp"
#include "multiprobe.hpp"
#include "sign_lsh.hpp"
#include "stat_tracker.hpp"
#include "stats/stats.hpp"
#include "stats/topk.hpp"
#include "tables.hpp"

/*
 * Implementation of Mulitprobe Locality Sensitive Hashing
 * that uses an arbitrary number of tables and uses multiprobe.
 */

namespace mp = boost::multiprecision;
namespace nr {

template <typename Vect, typename Hash = SignLSH<typename Vect::value_type>>
class LSH_MultiProbe_MultiTable : public MultiProbe<Vect> {
private:
  using Component = typename Vect::value_type;
  using KV = std::pair<Vect, int64_t>;
  using Table = std::unordered_map<size_t, std::list<KV>>;
  using MultiTable = std::vector<Table>;

  MultiTable tables;
  int64_t dim;
  int64_t num_buckets;
  std::vector<Hash> hash_functions;

  std::vector<size_t> rank(const Vect &q, size_t table, size_t max_hash,
                           int64_t adj) {
    /*
     * Returns the adj indices that are most similar to q's hash.
     * These have the highest chance of containing a neighbor of q.
     */
    const static size_t bit_lim = std::floor(std::log2(num_buckets)) + 1;
    const size_t idx = hash_functions.at(table).hash_max(q, num_buckets);
    return fast_sim_2bit(idx, bit_lim);
  }

public:
  LSH_MultiProbe_MultiTable(int64_t num_tables, int64_t bits, int64_t dimension,
                            int64_t num_buckets)
      : tables(num_tables, std::unordered_map<size_t, std::list<KV>>()),
        dim(dimension), num_buckets(num_buckets), hash_functions() {
    /*
     * Each table has its own hash function.
     */
    for (int64_t i = 0; i < num_tables; ++i) {
      hash_functions.emplace_back(bits, dim);
    }
  }

  LSH_MultiProbe_MultiTable(const LSH_MultiProbe_MultiTable &other) {
    tables(other.tables);
    dim = other.dim;
    hash_function(other.hash_function);
  }

  template <typename Cont>
  void fill(const Cont &data, bool is_normalized = false) {
    /*
     * fills the hash tables with input data.
     * The "is_normalized" argument is not necessary for this table.
     * It defaults to false so a value does not need to be passed in.
     */
    for (size_t table = 0; table < tables.size(); ++table) {
      const auto &hash = hash_functions.at(table);
      int64_t id = 0;
      for (const auto &datum : data) {
        const size_t hash_value = hash.hash_max(datum, num_buckets);
        tables.at(table)[hash_value].push_back({datum, id});
        ++id;
      }
    }
  }

  std::pair<std::optional<KV>, StatTracker> probe(const Vect &q, int64_t adj) {
    /*
     * returns vector closest to q found in the adj highest ranked buckets.
     * If every bucket checked is empty, then no neighbor will be found.
     * This is the only case it can return nullopt.
     */
    StatTracker tracker;
    KV neighbor = KV();
    Component min_dist = std::numeric_limits<Component>::max();
    for (size_t table = 0; table < tables.size(); ++table) {
      tracker.incr_tables_probed();
      for (size_t idx : rank(q, table, num_buckets, adj)) {
        tracker.incr_buckets_probed();
        for (const KV &x : tables.at(table)[idx]) {
          tracker.incr_comparisons();
          Component dist = (q - x.first).norm();
          if (dist < min_dist) {
            neighbor = x;
            min_dist = dist;
          }
        }
      }
    }
    if (min_dist == std::numeric_limits<Component>::max())
      return {std::nullopt, tracker};
    return {neighbor, tracker};
  }

  std::pair<std::optional<std::vector<KV>>, StatTracker>
  k_probe(int64_t k, const Vect &q, size_t adj) {
    /*
     * Returns the k vectors in adj highest ranked buckets that are closest to
     * the input vector q.
     */
    StatTracker tracker;
    std::vector<std::pair<KV, Component>> topk(0);
    topk.reserve(k + 1);
    Component largest_dist = std::numeric_limits<Component>::max();

    for (size_t table = 0; table < tables.size(); ++table) {
      tracker.incr_tables_probed();
      for (size_t idx : rank(q, table, num_buckets, adj)) {
        tracker.incr_buckets_probed();
        for (const KV &x : tables.at(table)[idx]) {
          tracker.incr_comparisons();
          Component dist = (q - x.first).norm();
          largest_dist = k_probe_step(k, x, dist, topk, largest_dist);
        }
      }
    }
    return k_probe_output(topk, tracker);
  }

  std::pair<std::optional<KV>, StatTracker>
  probe_approx(const Vect &q, Component c, int64_t adj) {
    /*
     * Returns the first vector within distance c that is found in the
     * adj highest ranked buckets.
     */
    StatTracker tracker;
    for (size_t table = 0; table < tables.size(); ++table) {
      tracker.incr_tables_probed();
      for (size_t idx : rank(q, table, num_buckets, adj)) {
        tracker.incr_buckets_probed();
        for (const KV &x : tables.at(table)[idx]) {
          tracker.incr_comparisons();
          Component dist = (q - x.first).norm();
          if (dist <= c) {
            return {x, tracker};
          }
        }
      }
    }
    return {std::nullopt, tracker};
  }

  std::pair<std::optional<std::vector<KV>>, StatTracker>
  k_probe_approx(int64_t k, const Vect &q, Component c, size_t adj) {
    /*
     * Function finds the first k items that are within distance c from
     * the query. Output is ordered by distant to nearest.
     */
    StatTracker tracker;
    std::vector<std::pair<KV, Component>> topk(0);
    topk.reserve(k + 1); // allocate now so it doesn't need to resize
    for (size_t table = 0; table < tables.size(); ++table) {
      tracker.incr_tables_probed();
      for (size_t idx : rank(q, table, num_buckets, adj)) {
        tracker.incr_buckets_probed();
        for (const KV &x : tables.at(table)[idx]) {
          tracker.incr_comparisons();
          Component dist = (q - x.first).norm();
          if (dist <= c)
            topk.push_back({x, dist});
          if (topk.size() == static_cast<size_t>(k))
            return k_probe_approx_output(k, topk, tracker);
        }
      }
    }
    return k_probe_approx_output(k, topk, tracker);
  }

  bool contains(const Vect &q) {
    /*
     * Check if q is contained in the tables.
     * Only need to check one because all tables contain all vectors
     */
    const Table &t = tables.at(0);
    const Hash &h = hash_functions.at(0);
    const size_t idx = h.hash_max(q, num_buckets);
    auto search = t.find(idx);
    if (search == t.end())
      return false;
    const std::list<KV> &l = (*search).second;
    auto vect_equality = [q](const KV &x) { return x.first.isApprox(q); };
    auto q_iter = std::find_if(l.begin(), l.end(), vect_equality);
    return q_iter != l.end();
  }

  Component k_probe_step(const int64_t k, const KV &x, const Component dist,
                         std::vector<std::pair<KV, Component>> &topk,
                         const Component largest_dist) {
    /*
     * Checks if x should be added to the topk. If it should,
     * it is added and the largest_dist is updated.
     */
    if (topk.size() < static_cast<size_t>(k)) {
      topk.push_back({x, dist});
      std::sort(topk.begin(), topk.end(), [](const auto &x, const auto &y) {
        return x.second > y.second;
      });
    } else if (dist < largest_dist) {
      insert_in_topk({x, dist}, topk);
    }
    return topk.at(0).second;
  }

  void insert_in_topk(const std::pair<KV, Component> &to_add,
                      std::vector<std::pair<KV, Component>> &topk) {
    /*
     * insert to_add into the topk so that the topk is distant to nearest.
     */
    stats::insert_unique_inplace(
        to_add, topk,
        [](const auto &x, const auto &y) { return x.second < y.second; },
        [](const auto &x, const auto &y) {
          return x.first.second == y.first.second;
        });
  }

  std::pair<std::optional<std::vector<KV>>, StatTracker>
  k_probe_output(const std::vector<std::pair<KV, Component>> &topk,
                 StatTracker tracker) {
    /*
     * simple function to process the output for k_probe.
     * if the topk is empty, it returns nullopt. otherwise, it copies
     * topk into a vector<KV> and returns that.
     */
    if (topk.size() == 0)
      return {std::nullopt, tracker};
    else {
      std::vector<KV> topk_out(topk.size());
      for (size_t i = 0; i < topk.size(); ++i)
        topk_out.at(i) = topk.at(i).first;
      return {topk_out, tracker};
    }
  }

  std::pair<std::optional<std::vector<KV>>, StatTracker>
  k_probe_approx_output(int64_t k, std::vector<std::pair<KV, Component>> &topk,
                        StatTracker &tracker) const {
    /*
     * Function to format the topk for k_probe_approx.
     * the topk should be order distant to nearest.
     */
    if (topk.size() == 0)
      return {std::nullopt, tracker};
    // should be sorted distant to nearest.
    std::sort(topk.begin(), topk.end(),
              [](const auto &x, const auto &y) { return x.second > y.second; });
    std::vector<KV> topk_output(topk.size());
    for (size_t i = 0; i < topk.size(); ++i) {
      topk_output.at(i) = topk.at(i).first;
    }
    return {topk_output, tracker};
  }

  void print_stats() {}

  MultiTable data() { return tables; }

  size_t num_tables() const { return tables.size(); }
}; // namespace nr

} // namespace nr
