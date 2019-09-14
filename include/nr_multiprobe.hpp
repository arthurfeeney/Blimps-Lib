
#pragma once

#include <algorithm>
#include <iostream>
#include <limits>
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
 * Multiprobe implementation of NR-LSH
 */

namespace nr {

template <typename Vect> class NR_MultiProbe : public MultiProbe<Vect> {
private:
  using Component = typename Vect::value_type;
  using KV = std::pair<Vect, int64_t>;

  std::vector<Tables<Vect>> probe_tables;
  int64_t dim;

public:
  NR_MultiProbe(int64_t num_tables, int64_t num_partitions, int64_t bits,
                int64_t dim, size_t num_buckets)
      : probe_tables(num_tables), dim(dim) {
    for (auto &probe_table : probe_tables) {
      probe_table = Tables<Vect>(num_partitions, bits, dim, num_buckets);
    }
  }

  NR_MultiProbe(const NR_MultiProbe &other) {
    probe_tables(other.probe_tables);
  }

  template <typename Cont> void fill(const Cont &data, bool is_normalized) {
    /*
     * if is_normalized is true, then the data input to the table is
     * normalized, otherwise, it is inserted as the original unnormalised
     * value.
     */
    for (auto &probe_table : probe_tables) {
      if (data.size() < probe_table.size()) {
        throw std::runtime_error("NR_MultiProbe::fill."
                                 "data.size() < num partitions");
      }
      probe_table.fill(data, is_normalized);
    }
  }

  std::pair<std::optional<KV>, StatTracker> probe(const Vect &q, int64_t adj) {
    /*
     * returns the vector in adj highest ranked buckets that has
     * the largest inner product with q.
     */

    StatTracker tracker;

    for (auto &probe_table : probe_tables) {
      auto p = probe_table.probe(q, adj);
      std::optional<KV> out = p.first;
      tracker += p.second;
      if (out)
        return std::make_pair(out, tracker);
    }
    return std::make_pair(std::nullopt, tracker);
  }

  std::pair<std::optional<std::vector<KV>>, StatTracker>
  k_probe(int64_t k, const Vect &q, size_t adj) {
    /*
     * returns the k vectors in adj highest ranked buckets that have the largest
     * inner products with q.
     */
    if (k < 1)
      throw std::runtime_error("NR_MultiProbe::k_probe, k < 1");

    StatTracker tracker;
    // store KV pair and the inner product value with q; avoid recomputing
    std::vector<std::pair<KV, Component>> topk(0);
    // reserve memory now so it never needs to be resized in loops.
    topk.reserve(k + 1);
    k_probe_tables(k, q, adj, topk);
    // copy output to just a vector of KV.
    std::vector<KV> topk_out(topk.size());
    std::generate(topk_out.begin(), topk_out.end(),
                  [&topk, n = -1]() mutable { return topk.at(++n).first; });
    return {topk_out, tracker};
  }

  std::pair<std::optional<KV>, StatTracker>
  probe_approx(const Vect &q, Component c, int64_t adj) {
    /*
     * returns the first vector in adj highest ranked buckets that has
     * an inner product with q that is greater than c.
     */
    StatTracker tracker;
    for (auto &probe_table : probe_tables) {
      tracker.incr_tables_probed();
      auto p = probe_table.probe_approx(q, c, adj);
      tracker += p.second;
      if (p.first)
        return std::make_pair(p.first.value(), tracker);
    }
    return std::make_pair(std::nullopt, tracker);
  }

  std::pair<std::optional<std::vector<KV>>, StatTracker>
  k_probe_approx(int64_t k, const Vect &q, Component c, size_t adj) {
    /*
     * returns the k vectors from adj buckets that have the largest inner
     * products with q.
     */
    StatTracker tracker;
    std::vector<KV> vects(0);
    for (auto &probe_table : probe_tables) {
      if (k - vects.size() > 0) {
        auto found = probe_table.k_probe_approx(k - vects.size(), q, c, adj);
        tracker += found.second;
        if (found.first) {
          std::vector<KV> v = found.first.value();
          insert_kv(vects, v);
        }
      }
    }
    if (vects.size() == 0) {
      std::make_pair(std::nullopt, tracker);
    }
    return std::make_pair(vects, tracker);
  }

  bool contains(const Vect &q) {
    /*
     * Check if q is contained in the tables.
     * Only need to check one because all tables contain all vectors
     * More complicated than LSH contains: must use each partitions' normalizers
     * when searching. Still only need to search one table.
     */
    return probe_tables.at(0).contains(q);
  }

  void k_probe_tables(int64_t k, const Vect &q, size_t adj,
                      std::vector<std::pair<KV, Component>> &topk) {
    /*
     * Iterates over each probe_table.
     * In each probe_table, it starts iterating over the highest ranked
     * buckets of each partition and works it way down to the lowest ranked
     * buckets.
     */
    Component smallest_inner = std::numeric_limits<Component>::min();
    for (size_t probe = 0; probe < probe_tables.size(); ++probe) {
      const auto rankings(probe_tables.at(probe).rank_around_query(q, adj));
      for (size_t col = 0; col < adj; ++col) {
        for (size_t t = 0; t < probe_tables.at(probe).size(); ++t) {
          const std::list<KV> &bucket =
              probe_tables.at(probe).at(t).at(rankings.at(t).at(col));
          smallest_inner = probe_bucket(k, q, bucket, smallest_inner, topk);
        }
      }
    }
  }

  Component probe_bucket(int64_t k, const Vect &q, const std::list<KV> &bucket,
                         Component smallest_inner,
                         std::vector<std::pair<KV, Component>> &topk) {
    /*
     * Iterates cross the bucket looking for inner products that are larger
     * than the current smallest inner product in the topk.
     * If something is smallet than the smallest thing in the topk, it clearly
     * is not in the topk and can be skipped!
     *
     * if less than k items have been found so far, it just adds them into
     * the topk.
     */
    for (const KV &item : bucket) {
      const Component inner = q.dot(item.first);
      if (topk.size() < static_cast<size_t>(k)) {
        build_topk({item, inner}, topk);
      } else if (inner > smallest_inner) {
        insert_in_topk({item, inner}, topk);

        // since a new item larger than the smallest element of topk
        // was added, there is a new smallest inner.
        smallest_inner = topk.at(0).second;
      }
    }
    return smallest_inner;
  }

  void build_topk(const std::pair<KV, Component> &to_add,
                  std::vector<std::pair<KV, Component>> &topk) {
    /*
     * Add to_add to the top and sort it.
     * keeping it sorted is necessary since the smallest element is in the
     * front of the topk.
     */
    topk.push_back(to_add);
    std::sort(
        topk.begin(), topk.end(),
        [](const std::pair<KV, Component> &x,
           const std::pair<KV, Component> &y) { return x.second < y.second; });
  }

  void insert_in_topk(const std::pair<KV, Component> &to_add,
                      std::vector<std::pair<KV, Component>> &topk) {
    /*
     * inserts values into the topk inplace. The previous smallest value is
     * automatically overwritten, so the size stays at k.
     *
     * takes two comparator functions: greater and equality.
     */
    stats::insert_unique_inplace(
        to_add, topk,
        [](const std::pair<KV, Component> &x,
           const std::pair<KV, Component> &y) { return x.second > y.second; },
        // two items are equal if they have the same id.
        [](const std::pair<KV, Component> &x,
           const std::pair<KV, Component> &y) {
          return x.first.second == y.first.second;
        });
  }

  static typename std::vector<KV>::const_iterator
  find_value(const std::vector<KV> &vects, int64_t value) {
    for (auto iter = vects.begin(); iter != vects.end(); ++iter) {
      if ((*iter).second == value) {
        return iter;
      }
    }
    return vects.end();
  }

  static void insert_kv(std::vector<KV> &vects,
                        const std::vector<KV> &to_insert) {
    // inserts only key-value pairs with unique values.
    // modifies in-place.
    for (auto &kv : to_insert) {
      if (find_value(vects, kv.second) == vects.end()) {
        // kv is not in vects.
        vects.push_back(kv);
      }
    }
  }

  void print_stats() {
    for (auto &probe_table : probe_tables) {
      probe_table.print_stats();
    }
  }

  size_t num_tables() const { return probe_tables.size(); }
}; // namespace nr

} // namespace nr
