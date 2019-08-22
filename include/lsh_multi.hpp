
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
class LSH_MultiProbe_MultiTable { // }: public MultiProbe<Vect> {
private:
  using Component = typename Vect::value_type;
  using KV = std::pair<Vect, int64_t>;
  using MultiTable = std::vector<std::unordered_map<size_t, std::list<KV>>>;

  MultiTable tables;
  int64_t dim;
  int64_t num_buckets;

  // tracks all of the keys in each table.
  // Storing keys after tables are filled allows for faster rankings
  std::vector<std::unordered_set<size_t>> table_keys;

  // defaults to SignLSH, but user can pass in a PStableLSH.
  std::vector<Hash> hash_functions;

  static double sim(size_t x, size_t y, size_t num_buckets) {
    // finds the number of bits in x and y that are the same.
    return static_cast<double>(
        stats::same_bits(x, y, std::floor(std::log2(num_buckets)) + 1));
  }

  std::vector<size_t> probe_ranking(size_t table, size_t idx,
                                    size_t num_buckets) {
    // std::vector<size_t> rank(num_buckets, 0);
    // std::iota(rank.begin(), rank.end(), 0);
    // sort in descending order. Most similar in the front.
    // manipulates table keys in-place. Hopefully still fast
    std::vector<size_t> rank(table_keys.at(table).begin(),
                             table_keys.at(table).end());
    std::sort(rank.begin(), rank.end(), [&](size_t x, size_t y) {
      return sim(idx, x, num_buckets) > sim(idx, y, num_buckets);
    });
    return rank;
  }

  std::vector<size_t> rank(const Vect &q, size_t table, size_t max_hash,
                           int64_t adj) {
    // return indices of the top 'adj' ranked buclets.
    // not static because it uses the hash function.
    const size_t idx = hash_functions.at(table).hash_max(q, num_buckets);
    std::vector<size_t> ranks = probe_ranking(table, idx, max_hash);
    return std::vector<size_t>(ranks.begin(), ranks.begin() + adj);
  }

  void manage_topk(std::vector<std::pair<KV, Component>> &topk, int64_t k,
                   const Vect &query, KV pos) {
    /*
     * Clunky function to track the topk vectors closest to q.
     */
    if (k < 1) {
      throw std::runtime_error(
          "LSH_MultiProbe_MultiTable::manage_topk, k must be positive");
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
  LSH_MultiProbe_MultiTable(int64_t num_tables, int64_t bits, int64_t dimension,
                            int64_t num_buckets)
      : tables(num_tables, std::unordered_map<size_t, std::list<KV>>()),
        dim(dimension), num_buckets(num_buckets), table_keys(num_tables),
        hash_functions() {
    for (int64_t i = 0; i < num_tables; ++i) {
      hash_functions.emplace_back(bits, dim);
    }
  }

  LSH_MultiProbe_MultiTable(const LSH_MultiProbe_MultiTable &other) {
    tables(other.tables);
    dim = other.dim;
    table_keys = other.table_keys;
    hash_function(other.hash_function);
  }

  template <typename Cont>
  void fill(const Cont &data, bool is_normalized = false) {
    for (size_t table = 0; table < tables.size(); ++table) {
      const auto &hash = hash_functions.at(table);
      int64_t id = 0;
      for (const auto &datum : data) {
        const size_t hash_value = hash.hash_max(datum, num_buckets);
        tables.at(table)[hash_value].push_back({datum, id});
        ++id;
      }
      for (const auto &kv_pair : tables.at(table)) {
        table_keys.at(table).insert(kv_pair.first);
      }
    }
  }

  std::pair<std::optional<KV>, StatTracker> probe(const Vect &q, int64_t adj) {
    StatTracker tracker;
    KV neighbor = KV();
    Component min_dist = std::numeric_limits<Component>::max();
    for (size_t table = 0; table < tables.size(); ++table) {
      for (size_t idx : rank(q, table, num_buckets, adj)) {
        for (const KV &x : tables.at(table).at(idx)) {
          tracker.incr_comparisons();
          Component dist = (q - x.first).norm();
          if (dist < min_dist) {
            neighbor = x;
            min_dist = dist;
          }
        }
      }
    }
    return {neighbor, tracker};
  }

  std::pair<std::optional<std::vector<KV>>, StatTracker>
  k_probe(int64_t k, const Vect &q, size_t adj) {}

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
    StatTracker tracker;
    for (size_t table = 0; table < tables.size(); ++table) {
      tracker.incr_tables_probed();
      for (size_t idx : rank(q, table, num_buckets, adj)) {
        tracker.incr_buckets_probed();
        for (const KV &x : tables.at(table).at(idx)) {
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
  k_probe_approx(int64_t k, const Vect &q, Component c, size_t adj) {}

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

  void print_stats() const {}

  MultiTable data() { return tables; }

  size_t num_tables() const { return tables.size(); }
};

} // namespace nr
