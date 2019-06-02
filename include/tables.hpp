
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
    auto parts = partitioner(data, tables.size());
    auto normal_data_and_U = normalizer(data, parts);
    auto normal_data = normal_data_and_U.first;
    this->normalizers = normal_data_and_U.second;
    auto indices = simple_LSH_partitions<decltype(normal_data), Component>(
        normal_data, hash, num_buckets);

    std::vector<std::vector<Vect>> parted_data(parts.size());

    // each thread inserts into its own partitions, so this should be
    // safe even though there is a push_back.
    for (size_t p = 0; p < parts.size(); ++p) {
      for (size_t i = 0; i < parts.at(p).size(); ++i) {
        // parted_data.at(p).push_back(data.at(parts.at(p).at(i)));
        parted_data.at(p).push_back(normal_data.at(p).at(i));
      }
    }

    for (size_t p = 0; p < tables.size(); ++p) {
      /*for (auto idx : indices.at(p)) {
        std::cout << idx << ' ';
      }
      std::cout << '\n';*/
      tables.at(p).fill(parted_data.at(p), indices.at(p), parts.at(p),
                        normalizers.at(p), is_normalized);
    }
  }

  std::pair<bool, KV> MIPS(const Vect &q) const {
    /*
     * Search partitions in each table for exact MIP with q.
     */

    // tracks the tables that successfuly found a MIP
    std::vector<bool> success(tables.size(), false);
    std::vector<KV> check(tables.size());

    for (size_t idx = 0; idx < tables.size(); ++idx) {
      std::pair<bool, KV> check_j = tables.at(idx).MIPS(q);
      if (check_j.first) {
        success.at(idx) = true;
        check.at(idx) = check_j.second;
      }
    }

    std::vector<KV> x(0);
    for (size_t idx = 0; idx < tables.size(); ++idx) {
      if (success.at(idx)) {
        x.push_back(check.at(idx));
      }
    }

    if (x.size() == 0) {
      return std::make_pair(false, KV());
    }

    // return pair of:
    // true for success (duh!)
    // and the element of x the maximizes the dot product w/ q.
    return std::make_pair(
        true, *std::max_element(x.begin(), x.end(), [q](KV y, KV z) {
          return q.dot(y.first) < q.dot(z.first);
        }));
  }

  std::pair<std::optional<KV>, StatTracker> probe(const Vect &q,
                                                  int64_t n_to_probe) const {
    std::vector<KV> x(0);

    StatTracker table_tracker;

    for (auto it = tables.begin(); it != tables.end(); ++it) {
      // std::optional<KV> xj = (*it).probe(q, n_to_probe);
      auto found = (*it).probe(q, n_to_probe);
      std::optional<KV> xj = found.first;
      table_tracker += found.second;
      table_tracker.incr_partitions_probed();
      if (xj) {
        x.push_back(xj.value());
      }
    }
    // if x.size() == 0, nothing was found, so nothing should be returned.
    // this must happen before finding the max element.
    if (x.size() == 0) {
      return std::make_pair(std::nullopt, table_tracker);
    }
    KV ret = *std::max_element(x.begin(), x.end(), [&](KV y, KV z) {
      return q.dot(y.first) < q.dot(z.first);
    });
    return std::make_pair(ret, table_tracker);
  }

  std::pair<std::optional<std::vector<KV>>, StatTracker>
  k_probe(int64_t k, const Vect &q, int64_t n_to_probe)  {
    if (k < 0) {
      throw std::runtime_error(
          "tables::k_probe. k must be non-negative");
    }
    StatTracker table_tracker;

    using mp = boost::multiprecision::cpp_int;
    mp mp_hash = hash(q);
    mp residue = mp_hash % num_buckets;
    int64_t idx = residue.convert_to<int64_t>();
    auto rankings = sub_tables_rankings(idx);

    std::vector<KV> vects(0);
    // look through the top n_to_probe ranked buckets.
    for (size_t col = 0; col < static_cast<size_t>(n_to_probe); ++col) {
      for (size_t t = 0; t < rankings.size(); ++t) {
      
        // k largest items in highly ranked bucket rankings[t][col].
        auto found = tables[t].topk_in_bucket(k, rankings[t][col], q);
        if(found.size() > 0) {
          vects.insert(vects.end(), found.begin(), found.end());
        }
      }
    }
    if(vects.size() == 0) {
      return std::make_pair(std::nullopt, table_tracker);
    }
    else {
      return std::make_pair(vects, table_tracker);
    }
  }

  std::vector<std::vector<int64_t>> sub_tables_rankings(int64_t idx) const {
    std::vector<std::vector<int64_t>> rankings(tables.size());
    for (size_t i = 0; i < tables.size(); ++i) {
      rankings.at(i) = tables.at(i).probe_ranking(idx);
    }
    return rankings;
  }

  std::pair<std::optional<KV>, StatTracker>
  probe_approx(const Vect &q, Component c, int64_t adj) const {
    StatTracker table_tracker;

    using mp = boost::multiprecision::cpp_int;
    mp mp_hash = hash(q);
    mp residue = mp_hash % num_buckets;
    int64_t idx = residue.convert_to<int64_t>();

    auto rankings = sub_tables_rankings(idx);
    // iterate column major though rankings until dot(q, x) > c is found.
    // look through adj other buckets. Should be the top ranked ones.
    // probes the best bucket of each table first.
    for (int64_t col = 0; col < adj; ++col) {
      for (size_t t = 0; t < tables.size(); ++t) {

        std::pair<std::optional<KV>, StatTracker> found =
            tables.at(t).look_in(rankings[t][col], q, c);
        table_tracker += found.second; // add partition's stats to total
        if (found.first) {
          // probed t partitions before value was found.
          table_tracker.k_partitions_probed(t);
          return std::make_pair(found.first.value(), table_tracker);
        }
      }
    }
    // If nothing found, it looked through the top adj buckets.
    table_tracker.k_partitions_probed(rankings.size());
    return std::make_pair(std::nullopt, table_tracker);
  }

  std::pair<std::optional<std::vector<KV>>, StatTracker>
  k_probe_approx(int64_t k, const Vect &q, Component c, size_t adj) const {
    if (k < 0) {
      throw std::runtime_error(
          "tables::k_probe_approx. k must be non-negative");
    }
    StatTracker table_tracker;

    using mp = boost::multiprecision::cpp_int;
    mp mp_hash = hash(q);
    mp residue = mp_hash % num_buckets;
    int64_t idx = residue.convert_to<int64_t>();

    auto rankings = sub_tables_rankings(idx);
    std::vector<KV> vects(0);
    for (size_t col = 0; col < adj; ++col) {
      for (size_t t = 0; t < rankings.size(); ++t) {

        auto found =
            tables[t].look_in_until(rankings[t][col], q, c, k - vects.size());

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
          return std::make_pair(vects, table_tracker);
        }
      }
    }
    // if everything was searched, and at least one thing was found, return
    // success, if everything was searched and nothing was found, then the
    // search failed.
    if (vects.size() == 0) {
      return std::make_pair(std::optional<std::vector<KV>>{}, table_tracker);
    }
    return std::make_pair(vects, table_tracker);
  }

  void print_stats() {
    int64_t table_id = 1;
    for (auto &table : tables) {
      std::cout << "table " << table_id << '\n';
      table.print_stats();
      ++table_id;
    }
  }

  const Table<Vect> &at(int idx) const {
    if (!(idx < size())) {
      throw std::out_of_range("Tables::at(idx) idx out of bounds.");
    }
    return (*this)[idx];
  }

  const Table<Vect> &operator[](int idx) const { return tables[idx]; }

  size_t size() const { return tables.size(); }

  typename std::vector<Table<Vect>>::iterator begin() { return tables.begin(); }
  typename std::vector<Table<Vect>>::iterator end() { return tables.end(); }
};

} // namespace nr
