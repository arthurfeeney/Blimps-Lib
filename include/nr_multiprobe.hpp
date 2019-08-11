
#pragma once

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
    // if is_normalized is true, then the data input to the table is
    // normalized, otherwise, it is inserted as the original unnormalised
    // value.
    for (auto &probe_table : probe_tables) {
      if (data.size() < probe_table.size()) {
        throw std::runtime_error("NR_MultiProbe::fill."
                                 "data.size() < num partitions");
      }
      probe_table.fill(data, is_normalized);
    }
  }

  std::pair<std::optional<KV>, StatTracker> probe(const Vect &q, int64_t adj) {
    // searches through the first adj most likely buckets

    StatTracker tracker;

    for (auto &probe_table : probe_tables) {
      auto p = probe_table.probe(q, adj);
      std::optional<KV> out = p.first;
      tracker += p.second;
      if (out) {
        return std::make_pair(out, tracker);
      }
    }
    return std::make_pair(std::nullopt, tracker);
  }

  std::pair<std::optional<std::vector<KV>>, StatTracker>
  k_probe(int64_t k, const Vect &q, size_t adj) {
    StatTracker tracker;

    if (k < 1) {
      throw std::runtime_error("NR_MultiProbe::k_probe, k must be positive");
    }

    std::vector<KV> topk(0);

    // use each sub-tables rankings to find the top-k largest
    // items of all the buckets.
    for (size_t probe = 0; probe < probe_tables.size(); ++probe) {
      auto rankings = probe_tables.at(probe).rank_around_query(q);
      for (size_t col = 0; col < adj; ++col) {
        for (size_t t = 0; t < probe_tables.at(probe).size(); ++t) {
          const std::list<KV> &bucket =
              probe_tables.at(probe).at(t).at(rankings.at(t).at(col));
          for (const KV &item : bucket) {
            if (topk.size() < static_cast<size_t>(k)) {
              topk.push_back(item);
              std::sort(topk.begin(), topk.end(), [&q](KV x, KV y) {
                return q.dot(x.first) < q.dot(y.first);
              });
            } else {
              stats::insert_inplace(item, topk, [&q](KV x, KV y) {
                return q.dot(x.first) > q.dot(y.first);
              });
            }
          }
        }
      }
    }
    return std::make_pair(topk, tracker);
  }

  std::pair<std::optional<KV>, StatTracker>
  probe_approx(const Vect &q, Component c, int64_t adj) {
    // searches until it finds some x with dot(x, q) > c
    StatTracker tracker;

    for (auto &probe_table : probe_tables) {
      tracker.incr_tables_probed();
      auto p = probe_table.probe_approx(q, c, adj);
      tracker += p.second;
      if (p.first) {
        return std::make_pair(p.first.value(), tracker);
      }
    }
    return std::make_pair(std::nullopt, tracker);
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

  std::pair<std::optional<std::vector<KV>>, StatTracker>
  k_probe_approx(int64_t k, const Vect &q, Component c, size_t adj) {
    // searches until it finds k vectors, x where all x have dot(x, q) > c
    // return probe_table.k_probe_approx(k, q, c);

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

  KV find_max_inner(const Vect &q) {
    /*
     * Finds the true maximum inner product in the dataset, not an
     * approximate one.
     * This exists in case the user does not have access to the
     * original dataset for some reason.
     * just looks through a single probe_tyable since it will contain everything
     */
    return probe_tables.at(0).MIPS(q).second;
  }

  void print_stats() {
    for (auto &probe_table : probe_tables) {
      probe_table.print_stats();
    }
  }

  size_t num_tables() const { return probe_tables.size(); }
};

} // namespace nr
