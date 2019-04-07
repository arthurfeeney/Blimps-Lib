
#pragma once

#include <cmath>
#include <limits>
#include <list>
#include <map>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <utility>
#include <vector>

#include <iostream>

#include "simple_lsh.hpp"
#include "stat_tracker.hpp"
#include "stats.hpp"

namespace nr {

template <typename Vect> class Table {
private:
  using Component = typename Vect::value_type;
  using KV = std::pair<Vect, int64_t>;

  size_t num_buckets;
  std::vector<std::list<KV>> table;
  SimpleLSH<Component> hash;
  typename Vect::value_type normalizer; // this partitions Up normalizer

public:
  Table(SimpleLSH<Component> hash, size_t num_buckets)
      : num_buckets(num_buckets), table(num_buckets), hash(hash),
        normalizer(0) {}

  void fill(const std::vector<Vect> &normalized_partition,
            const std::vector<int64_t> &indices,
            const std::vector<int64_t> &ids, const Component Up,
            bool is_normalized) {
    for (size_t i = 0; i < indices.size(); ++i) {
      KV to_insert = std::make_pair(normalized_partition.at(i), ids.at(i));
      if (is_normalized) {
        // use % to make sure all indices in range.
        table.at(indices.at(i) % table.size()).push_back(to_insert);
      } else {
        KV fix_to_insert =
            std::make_pair(to_insert.first * Up, to_insert.second);

        table.at(indices.at(i) % table.size()).push_back(fix_to_insert);
      }
    }
    normalizer = Up;
  }

  size_t first_non_empty_bucket() const {
    for (size_t bucket = 0; bucket < num_buckets; ++bucket) {
      if (table.at(bucket).size() > 0) {
        return bucket;
      }
    }
    return -1;
  }

  std::pair<bool, KV> MIPS(const Vect &q) const {
    size_t start_bucket = first_non_empty_bucket();
    if (start_bucket == -1) {
      /*
       * it is required that data.size() > num partitions, so
       * it should be impossible for a partition to be totally empty.
       */
      throw std::runtime_error("table::MIPS, All buckets empty.");
    }
    KV max = *table.at(start_bucket).begin();
    double big_dot = q.dot(max.first);
    for (size_t idx = start_bucket; idx < num_buckets; ++idx) {
      for (auto &current : table[idx]) {
        // KV current = *iter;
        double dot = q.dot(current.first);
        if (dot > big_dot) {
          big_dot = dot;
          max = current;
        }
      }
    }
    return std::make_pair(true, max);
  }

  std::pair<bool, KV> probe(const Vect &q, int64_t n_to_probe) {
    int64_t idx = hash(q) % table.size();
    std::vector<int64_t> rank = probe_ranking(idx);
    // initialize to impossible values
    KV max = std::make_pair(Vect(1), -1);
    double big_dot = std::numeric_limits<Component>::min();
    for (int64_t r = 0; r < n_to_probe; ++r) {
      for (const auto &current : table.at(rank.at(r))) {
        double dot = q.dot(current.first);
        if (dot > big_dot) {
          big_dot = dot;
          max = current;
        }
      }
    }
    if (max.second < 0) { // no large inner products were found.
      return std::make_pair(false, KV());
    }
    return std::make_pair(true, max);
  }

  double sim(int64_t idx, int64_t other) const {
    /*
     * Similar inputs result in POSITIBE output.
     * Disimilar inputs result in NEGATIVE output.
     * sort of similar inputs result in an output closer to zero.
     */
    constexpr double PI = 3.141592653589;
    constexpr double e = 0.001;
    double l = static_cast<double>(__builtin_popcount(idx & other));
    double L = static_cast<double>(hash.bit_count());

    return normalizer * std::cos(PI * (1.0 - e) * (1 - (l / L)));
  }

  std::vector<int64_t> probe_ranking(int64_t idx) const {
    std::vector<int64_t> rank(table.size(), 0);
    std::iota(rank.begin(), rank.end(), 0);

    // So this should be in descending order.
    std::sort(rank.begin(), rank.end(),
              [&](int64_t x, int64_t y) { return sim(idx, x) > sim(idx, y); });

    return rank;
  }

  std::tuple<bool, KV, StatTracker> look_in(int64_t bucket, const Vect &q,
                                            double c) {
    // returns first vector x of this bucket with dot(q, x) > c
    StatTracker partition_tracker;
    for (auto it = table[bucket].begin(); it != table[bucket].end(); ++it) {
      partition_tracker.incr_comparisons();
      if (q.dot((*it).first) > c) {
        return std::make_tuple(true, *it, partition_tracker);
      }
    }
    return std::make_tuple(false, KV(), partition_tracker);
  }

  std::tuple<bool, std::vector<KV>, StatTracker>
  look_in_until(int64_t bucket, const Vect &q, double c, size_t limit) {
    // searches this bucket until it finds <limit> vectors x where dot(q, x) > c
    // or it reaches the end of the bucket. It returns success as long as at
    // least one such x is found.
    StatTracker partition_tracker;
    std::vector<KV> successful(0);
    for (const auto &x : table[bucket]) {
      partition_tracker.incr_comparisons();
      if (q.dot(x.first) > c) {
        successful.push_back(x);
      }
      if (successful.size() == limit) {
        // if limit vectors were found, the search can stop early
        return std::make_tuple(true, successful, partition_tracker);
      }
    }
    // return true if at least one vector was found. False otherwise
    return std::make_tuple(successful.size() > 0, successful,
                           partition_tracker);
  }

  void print_stats() {
    std::vector<size_t> bucket_sizes(table.size(), 0);
    size_t num_empty_buckets = 0;

#pragma omp parallel for
    for (size_t i = 0; i < bucket_sizes.size(); ++i) {
      bucket_sizes.at(i) = table.at(i).size();
      if (bucket_sizes.at(i) == 0) {
        ++num_empty_buckets;
      }
    }

    auto most_content =
        std::max_element(bucket_sizes.begin(), bucket_sizes.end());

    size_t max = *most_content;
    size_t min = *std::min_element(bucket_sizes.begin(), bucket_sizes.end());

    auto var = stats::variance(bucket_sizes);
    auto stdev = std::sqrt(var);

    std::cout << "\tmean:       " << stats::mean(bucket_sizes) << '\n';
    std::cout << "\tmax:        " << max << '\n';
    std::cout << "\tmax bucket: " << most_content - bucket_sizes.begin()
              << '\n';
    std::cout << "\tmin:        " << min << '\n';
    std::cout << "\tvar:        " << var << '\n';
    std::cout << "\tstdev:      " << stdev << '\n';
    std::cout << "\tmedian:     " << stats::median(bucket_sizes) << '\n';
    std::cout << "\tempty:      " << num_empty_buckets << '\n';
    std::cout << "\tnon-empty:  " << num_buckets - num_empty_buckets << '\n';
  }

  const std::list<KV> &at(int idx) const {
    if (!(idx < size())) {
      throw std::out_of_range("Table::at(idx) idx out of bounds.");
    }
    return (*this)[idx];
  }

  const std::list<KV> &operator[](int idx) const { return table[idx]; }

  size_t size() const { return num_buckets; }
}; // namespace nr

} // namespace nr
