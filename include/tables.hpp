
#pragma once

#include <algorithm>
#include <boost/multiprecision/cpp_int.hpp>
#include <iostream>
#include <iterator>
#include <omp.h>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "index_builder.hpp"
#include "simple_lsh.hpp"
#include "stat_tracker.hpp"
#include "table.hpp"

namespace nr {

template <typename Vect> class Tables {
private:
  using Component = typename Vect::value_type;
  using KV = std::pair<Vect, int64_t>;

  int64_t num_partitions; // corresponds to the number of tables.
  size_t num_buckets;     // is the size of each table.
  SimpleLSH<Component> hash;
  std::vector<Table<Vect>> tables;
  std::vector<Component> normalizers;

public:
  Tables() : hash(0, 0) {}

  Tables(int64_t num_partitions, int64_t bits, int64_t dim, size_t num_buckets)
      : num_partitions(num_partitions), num_buckets(num_buckets),
        hash(bits, dim),
        tables(num_partitions, Table<Vect>(hash, num_buckets)) {
    std::cout << "hash dim: " << hash.dimension();
  }

  Tables(Tables &&other) {
    num_partitions = other.num_partitions;
    num_buckets = other.num_buckets;
    hash = other.hash;
    tables = other.tables;
    normalizers = other.normalizers;
  }

  Tables &operator=(Tables &&other) {
    num_partitions = other.num_partitions;
    num_buckets = other.num_buckets;
    hash = other.hash;
    tables = other.tables;
    normalizers = other.normalizers;
    return *this;
  }

  template <typename Cont> void fill(const Cont &data, bool is_normalized) {
    /*
     * partition data and put partitions into different tables.
     */
    /* auto parts = partitioner(data, tables.size());
    auto normal_data_and_U = normalizer(data, parts);
    auto normal_data = normal_data_and_U.first;
    this->normalizers = normal_data_and_U.second;
    auto indices = simple_LSH_partitions<decltype(normal_data), Component>(
        normal_data, hash, num_buckets);
    */
    // std::vector<std::vector<Vect>> parted_data(parts.size());

    // each thread inserts into its own partitions, so this should be
    // safe even though there is a push_back.
    /*
    for (size_t p = 0; p < parts.size(); ++p) {
      for (size_t i = 0; i < parts.at(p).size(); ++i) {
        parted_data.at(p).push_back(normal_data.at(p).at(i));
      }
    }
    */

    auto tup = IndexBuilder::build(data, num_partitions, num_buckets, hash);

    auto &parts = std::get<0>(tup);
    auto &normal_data = std::get<1>(tup);
    auto &normalizers = std::get<2>(tup);
    auto &indices = std::get<3>(tup);

    std::vector<std::vector<Vect>> parted_data(parts.size());
    for (size_t p = 0; p < parts.size(); ++p) {
      for (size_t i = 0; i < parts.at(p).size(); ++i) {
        parted_data.at(p).push_back(normal_data.at(p).at(i));
      }
    }

    for (size_t p = 0; p < tables.size(); ++p) {
      tables.at(p).fill(parted_data.at(p), indices.at(p), parts.at(p),
                        normalizers.at(p), is_normalized);
    }
  }

  std::pair<std::optional<KV>, StatTracker> probe(const Vect &q,
                                                  int64_t adj) const {
    /*
     * Find the vector in the adj highest ranked bucket of each partition
     * that has the largets inner product with q.
     */
    std::vector<KV> x(0);
    StatTracker table_tracker;
    for (auto &table : tables) {
      auto found = table.probe(q, adj);
      std::optional<KV> candidate_vect = found.first;
      if (candidate_vect)
        x.push_back(candidate_vect.value());
      table_tracker += found.second;
      table_tracker.incr_partitions_probed();
    }
    if (x.size() == 0)
      return {std::nullopt, table_tracker};

    KV ret = *std::max_element(x.begin(), x.end(), [&](KV y, KV z) {
      return q.dot(y.first) < q.dot(z.first);
    });
    return {ret, table_tracker};
  }

  auto sub_tables_rankings(int64_t idx, int64_t k) const {
    /*
     * finds rankings for each partition.
     */
    std::vector<std::vector<int64_t>> rankings(tables.size());
    for (size_t i = 0; i < tables.size(); ++i)
      rankings.at(i) = tables.at(i).probe_ranking(idx, k);
    return rankings;
  }

  std::vector<std::vector<int64_t>> rank_around_query(const Vect &q,
                                                      int64_t adj) const {
    int64_t idx = static_cast<int64_t>(hash.hash_max(q, num_buckets));
    return sub_tables_rankings(idx, adj);
  }

  std::pair<std::optional<KV>, StatTracker>
  probe_approx(const Vect &q, Component c, int64_t adj) const {
    /*
     * finds the first vector in any of the adj hiehgest ranked partitions
     * with q.dot(x) > c
     */
    StatTracker table_tracker;
    auto rankings = rank_around_query(q, adj);
    for (int64_t col = 0; col < adj; ++col) {
      for (size_t t = 0; t < tables.size(); ++t) {
        auto found = tables.at(t).look_in(rankings[t][col], q, c);
        table_tracker += found.second; // add partition's stats to total
        if (found.first) {
          // probed t partitions before value was found.
          table_tracker.k_partitions_probed(t);
          return std::make_pair(found.first.value(), table_tracker);
        }
      }
    }
    // If nothing found, it looked through the top ranked buckets.
    table_tracker.k_partitions_probed(rankings.size());
    return {std::nullopt, table_tracker};
  }

  std::pair<std::optional<std::vector<KV>>, StatTracker>
  k_probe_approx(int64_t k, const Vect &q, Component c, size_t adj) const {
    if (k < 0) {
      throw std::runtime_error(
          "tables::k_probe_approx. k must be non-negative");
    }
    StatTracker table_tracker;

    size_t idx = hash.hash_max(q, num_buckets);
    auto rankings = sub_tables_rankings(idx, adj);

    std::vector<KV> vects(0);
    for (size_t col = 0; col < adj; ++col) {
      for (size_t t = 0; t < rankings.size(); ++t) {

        auto found = tables[t].look_in_until(rankings.at(t).at(col), q, c,
                                             k - vects.size());

        table_tracker += found.second; // add partitions' stats together

        if (found.first) {
          // append vects from bucket to the ones already found.
          std::vector<KV> from_bucket = found.first.value();
          vects.insert(vects.end(), from_bucket.begin(), from_bucket.end());
        }
        if (vects.size() == static_cast<size_t>(k)) {
          // look_in_until stops when it finds k - vects.size things, so
          // checking for equality is safe.
          // if k things found, return success immediately.
          return {vects, table_tracker};
        }
      }
    }
    // if everything was searched, and at least one thing was found, return
    // success, if everything was searched and nothing was found, then the
    // search failed.
    if (vects.size() == 0)
      return {std::nullopt, table_tracker};
    return {vects, table_tracker};
  }

  bool contains(const Vect &q) {
    /*
     * Check if any partitions contain q.
     */
    auto any_true = [q](bool accum, const Table<Vect> &x) {
      return accum || x.contains(q);
    };
    return std::accumulate(tables.begin(), tables.end(), false, any_true);
  }

  void print_stats() {
    int64_t table_id = 1;
    for (auto &table : tables) {
      std::cout << "table " << table_id << '\n';
      table.print_stats();
      ++table_id;
    }
  }

  const Table<Vect> &at(size_t idx) const {
    if (!(idx < size())) {
      throw std::out_of_range("Tables::at(idx) idx out of bounds.");
    }
    return (*this)[idx];
  }

  const Table<Vect> &operator[](size_t idx) const { return tables[idx]; }

  size_t size() const { return tables.size(); }

  typename std::vector<Table<Vect>>::iterator begin() { return tables.begin(); }
  typename std::vector<Table<Vect>>::iterator end() { return tables.end(); }
};

} // namespace nr
