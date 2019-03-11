
#pragma once

#include <algorithm>
#include <iostream>
#include <iterator>
#include <omp.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "comp_counter.hpp"
#include "index_builder.hpp"
#include "simple_lsh.hpp"
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
  Tables() : num_partitions(0), hash(0, 0) {} // empty constructor.

  Tables(int64_t num_partitions, int64_t bits, int64_t dim, size_t num_buckets)
      : num_partitions(num_partitions), num_buckets(num_buckets),
        hash(bits, dim),
        tables(num_partitions, Table<Vect>(hash, num_buckets)) {}

  template <typename Cont> void fill(const Cont &data, bool is_normalized) {
    /*
     * partition data and put partitions into different tables.
     */
    auto parts = partitioner(data, tables.size());
    auto normal_data_and_U = normalizer(data, parts);
    auto normal_data = normal_data_and_U.first;
    normalizers = normal_data_and_U.second;
    auto indices = simple_LSH_partitions<decltype(normal_data), Component>(
        normal_data, hash);

    std::vector<std::vector<Vect>> parted_data(parts.size());

    // each thread inserts into its own partitions, so this should be
    // safe even though there is a push_back.
#pragma omp parallel for
    for (size_t p = 0; p < parts.size(); ++p) {
      for (size_t i = 0; i < parts.at(p).size(); ++i) {
        parted_data.at(p).push_back(data.at(parts.at(p).at(i)));
      }
    }

#pragma omp parallel for
    for (size_t p = 0; p < tables.size(); ++p) {
      tables.at(p).fill(parted_data.at(p), indices.at(p), parts.at(p),
                        normalizers.at(p), is_normalized);
    }
  }

  std::pair<bool, KV> MIPS(const Vect &q) const {
    /*
     * Search partitions in each table for exact MIP with q.
     */
    /*
    std::vector<KV> x(0);
    for (auto t : tables) {
      std::pair<bool, KV> xj = t.MIPS(q);
      if (xj.first) {
        x.push_back(xj.second);
      }
    }*/

    // tracks the tables that successfuly found a MIP
    std::vector<bool> success(tables.size(), false);
    std::vector<KV> check(tables.size());

#pragma omp parallel for
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

  std::pair<bool, KV> probe(const Vect &q, int64_t n_to_probe) {
    std::vector<KV> x(0);

    for (auto it = tables.begin(); it != tables.end(); ++it) {
      std::pair<bool, KV> xj = (*it).probe(q, n_to_probe);
      if (xj.first) {
        x.push_back(xj.second);
      }
    }

    KV ret = *std::max_element(x.begin(), x.end(), [&](KV y, KV z) {
      return q.dot(y.first) < q.dot(z.first);
    });

    // if x.size() == 0, nothing was found so it should return false
    return std::make_pair(x.size() != 0, ret);
  }

  std::vector<std::vector<int64_t>> sub_tables_rankings(int64_t idx) {

    std::vector<std::vector<int64_t>> rankings(tables.size());

    for (auto it = tables.begin(); it < tables.end(); ++it) {
      rankings[it - tables.begin()] = (*it).probe_ranking(idx);
    }

    return rankings;
  }

  std::pair<bool, KV> probe_approx(const Vect &q, Component c) {
    auto rankings = sub_tables_rankings(hash(q) % num_buckets);

    // iterate column major until dot(q, x) > c is found.
    for (size_t col = 0; col < num_buckets; ++col) {
      for (size_t t = 0; t < rankings.size(); ++t) {
        auto found = tables[t].look_in(rankings[t][col], q, c);
        if (found.first) {
          return found;
        }
      }
    }

    return std::make_pair(false, KV());
  }

  std::pair<bool, std::vector<KV>>
  k_probe_approx(int64_t k, const Vect &q, Component c,
                 CompCounter *counter = nullptr) {
    auto rankings = sub_tables_rankings(hash(q) % num_buckets);

    std::vector<KV> vects(0);

    for (size_t col = 0; col < num_buckets; ++col) {
      for (size_t t = 0; t < rankings.size(); ++t) {
        auto found = tables[t].look_in(rankings[t][col], q, c);
        if (found.first) {
          vects.push_back(found.second);
        }
        if (vects.size() == k) {
          return std::make_pair(true, vects);
        }
      }
    }

    return std::make_pair(vects.size() != 0, vects);
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
