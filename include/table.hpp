
#pragma once

#include <boost/multiprecision/cpp_int.hpp>

#include <cmath>
#include <iterator>
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
#include "stats/stats.hpp"

namespace nr {

template <typename Vect> class Table {
private:
  using Component = typename Vect::value_type;
  using KV = std::pair<Vect, int64_t>;

  size_t num_buckets;
  std::vector<std::list<KV>> table;
  SimpleLSH<Component> hash;
  typename Vect::value_type normalizer; // this partitions Up normalizer
  bool data_is_normalized = false;

public:
  Table(SimpleLSH<Component> hash, size_t num_buckets)
      : num_buckets(num_buckets), table(num_buckets), hash(hash),
        normalizer(0) {}

  void fill(const std::vector<Vect> &normalized_partition,
            const std::vector<int64_t> &indices,
            const std::vector<int64_t> &ids, const Component Up,
            bool is_normalized) {
    for (size_t i = 0; i < indices.size(); ++i) {
      KV to_insert = {normalized_partition.at(i), ids.at(i)};

      // modulo is done in the index builder. Don't need to repeat it.
      size_t bucket_idx = indices.at(i);

      if (is_normalized) {
        table.at(bucket_idx).push_back(to_insert);
      } else {
        // this UN-normalizes before inserting.
        KV fix_to_insert = {to_insert.first * Up, to_insert.second};
        table.at(bucket_idx).push_back(fix_to_insert);
      }
    }
    normalizer = Up;
    data_is_normalized = is_normalized;
  }

  int64_t first_non_empty_bucket() const {
    for (size_t bucket = 0; bucket < num_buckets; ++bucket) {
      if (table.at(bucket).size() > 0) {
        return static_cast<int64_t>(bucket);
      }
    }
    return -1;
  }

  std::pair<bool, KV> MIPS(const Vect &q) const {
    int64_t start_bucket = first_non_empty_bucket();
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
    return {true, max};
  }

  std::pair<std::optional<KV>, StatTracker> probe(const Vect &q,
                                                  int64_t n_to_probe) const {
    /*
     * Searches through the best n_to_probe buckets, ranked by sim,
     * returning the KV pair that results in
     * the largest inner product with the query point.
     * */
    using mp = boost::multiprecision::cpp_int;
    mp mp_hash = hash(q);
    mp residue = mp_hash % table.size();
    int64_t idx = residue.convert_to<int64_t>();

    StatTracker partition_tracker;

    std::vector<int64_t> rank = probe_ranking(idx, n_to_probe);
    // initialize to impossible values
    KV max = {Vect(1), -1};
    Component big_dot = std::numeric_limits<Component>::min();
    for (int64_t r = 0; r < n_to_probe; ++r) {
      for (const auto &current : table.at(rank.at(r))) {
        partition_tracker.incr_comparisons();
        Component dot = q.dot(current.first);
        if (dot > big_dot) {
          big_dot = dot;
          max = current;
        }
      }
    }
    if (max.second < 0) // no large inner products were found.
      return {std::nullopt, partition_tracker};
    return {max, partition_tracker};
  }

  inline double sim(size_t idx, size_t other) const {
    /*
     * Similar inputs result in POSITIBE output.
     * Disimilar inputs result in NEGATIVE output.
     * sort of similar inputs result in an output closer to zero.
     */
    constexpr double PI = 3.141592653589;
    constexpr double eps = 1e-3;
    const size_t bit_lim = std::floor(std::log2(num_buckets)) + 1;
    const double l = static_cast<double>(stats::same_bits(idx, other, bit_lim));
    const double L = static_cast<double>(hash.bit_count());
    return normalizer * std::cos(PI * (1.0 - eps) * (1.0 - (l / L)));
  }

  std::vector<int64_t> probe_ranking(int64_t idx, int64_t adj) const {
    /*
     * finds the similarity rankings of buckets using this tables sim
     * It retuns the top-adj most similar indices.
     */
    return stats::topk<int64_t>(
        adj, static_cast<int64_t>(0), static_cast<int64_t>(num_buckets),
        [&](const int64_t x, const int64_t y) {
          return sim(idx, x) > sim(idx, y);
        },
        [&](const int64_t x, const int64_t y) {
          return sim(idx, x) < sim(idx, y);
        });
  }

  std::pair<std::optional<KV>, StatTracker>
  look_in(int64_t bucket, const Vect &q, double c) const {
    /*
     * returns the first vector x (if any) where q.dot(x) > c
     */
    StatTracker partition_tracker;
    partition_tracker.incr_buckets_probed();
    for (auto &x : table.at(bucket)) {
      partition_tracker.incr_comparisons();
      if (q.dot(x.first) > c)
        return {x, partition_tracker};
    }
    return {std::nullopt, partition_tracker};
  }

  std::pair<std::optional<std::vector<KV>>, StatTracker>
  look_in_until(int64_t bucket, const Vect &q, double c, size_t limit) const {
    /*
     * searches this bucket until it finds <limit> vectors x where dot(q, x) > c
     * or it reaches the end of the bucket. It returns success as long as at
     * least one such x is found. Otherwise, it returns nullopt.
     */
    StatTracker partition_tracker;
    std::vector<KV> successful(0);
    for (const auto &x : table.at(bucket)) {
      partition_tracker.incr_comparisons();
      if (q.dot(x.first) > c)
        successful.push_back(x);
      if (successful.size() == limit) // return if limit vectors found
        return {successful, partition_tracker};
    }
    if (successful.size() == 0)
      return {std::nullopt, partition_tracker};
    return {successful, partition_tracker};
  }

  bool contains(Vect q) const {
    /*
     * Checks if this partition contains q.
     * If the data is normalized, it searches for the normalized vector
     * Otherwise, it searches for the unnormalized input.
     */
    Vect norm_q = q / normalizer;
    if (norm_q.norm() > 1) // norm_q is longer than longest in partition
      return false;
    const size_t idx = hash.hash_max(norm_q, num_buckets);
    const std::list<KV> &bucket = table.at(idx);
    q /= query_scalar();
    auto vect_equality = [q](const KV &x) { return x.first.isApprox(q); };
    auto q_iter = std::find_if(bucket.begin(), bucket.end(), vect_equality);
    return q_iter != bucket.end();
  }

  Component query_scalar() const {
    /*
     * When calling contains on a query, it must be scaled depending on how
     * the data in the table is stored. If it is stored as it was input, the
     * scalar is one. Otherwise, it must use the tables scalar.
     */
    return data_is_normalized ? normalizer : 1;
  }

  void print_stats() {
    std::vector<size_t> bucket_sizes(table.size(), 0);
    size_t num_empty_buckets = 0;

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

    std::cout << "Bucket Stats" << '\n';
    std::cout << "\tmean:       " << stats::mean(bucket_sizes) << '\n';
    std::cout << "\tmax:        " << max << '\n';
    std::cout << "\tmax bucket: " << most_content - bucket_sizes.begin()
              << '\n';
    std::cout << "\tmin:        " << min << '\n';
    std::cout << "\tvar:        " << var << '\n';
    std::cout << "\tstdev:      " << stdev << '\n';
    std::cout << "\tlow median: " << stats::lower_median(bucket_sizes) << '\n';
    std::cout << "\tempty:      " << num_empty_buckets << '\n';
    std::cout << "\tnon-empty:  " << num_buckets - num_empty_buckets << '\n';

    std::vector<size_t> non_empty_sizes = stats::nonzero(bucket_sizes);

    std::cout << "Non-empty Buckets" << '\n';
    std::cout << "\tmean:       " << stats::mean(non_empty_sizes) << '\n';
    std::cout << "\tvar:        " << stats::variance(non_empty_sizes) << '\n';
    std::cout << "\tstdev       " << stats::stdev(non_empty_sizes) << '\n';
    std::cout << "\tlow median: " << stats::lower_median(non_empty_sizes)
              << '\n';
  }

  const std::list<KV> &at(size_t idx) const {
    if (!(idx < size())) {
      throw std::out_of_range("Table::at(idx) idx out of bounds.");
    }
    return (*this)[idx];
  }

  const std::list<KV> &operator[](size_t idx) const { return table[idx]; }

  size_t size() const { return num_buckets; }
}; // namespace nr

} // namespace nr
