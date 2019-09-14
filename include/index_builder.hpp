
#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

#include "index_builder.hpp"
#include "simple_lsh.hpp"

/*
 * Contains operations for "index-building" for NR-LSH
 */

namespace mp = boost::multiprecision;
namespace nr {

struct IndexBuilder {
  template <typename VectCont, typename Hash>
  static auto build(const VectCont &data, int64_t m, int64_t num_buckets,
                    Hash hash) {
    /*
     * The main "building" function. This is really the only one that should
     * be called.
     */
    using Component = typename VectCont::value_type::value_type;
    auto parts = IndexBuilder::partitioner(data, m);
    auto normal_data_and_U = IndexBuilder::normalizer(data, parts);
    auto normal_data = normal_data_and_U.first;
    auto normalizers = normal_data_and_U.second;
    auto indices =
        IndexBuilder::simple_LSH_partitions<decltype(normal_data), Component>(
            normal_data, hash, num_buckets);
    return std::make_tuple(parts, normal_data, normalizers, indices);
  }

  template <typename VectCont, typename IntCont>
  static auto max_norm(const VectCont &data, const IntCont &partition) {
    /*
     * Finds the largest norm in a partition of data.
     * does not use max_element because it only applies to specific partition
     */
    using Scalar = typename VectCont::value_type::value_type;
    Scalar max = -1; // by definition, norm is positive.
    for (size_t i = 0; i < partition.size(); ++i) {
      Scalar norm = data.at(partition.at(i)).norm();
      if (norm > max) {
        max = norm;
      }
    }
    return max;
  }

  template <template <typename Vect> typename VectCont, typename Vect>
  static std::vector<int64_t> rank_by_norm(const VectCont<Vect> &dataset) {
    using Component = typename Vect::value_type;
    std::vector<Component> norms(dataset.size());
    for (size_t i = 0; i < norms.size(); ++i) {
      norms.at(i) = dataset.at(i).norm();
    }
    std::vector<int64_t> ranking(dataset.size());
    std::iota(ranking.begin(), ranking.end(), 0);
    // rank the vectors in dataset based on their norms
    // ascending order
    std::sort(ranking.begin(), ranking.end(), [norms](int64_t x, int64_t y) {
      return norms.at(x) < norms.at(y);
    });
    return ranking;
  }

  template <typename VectCont>
  static std::vector<std::vector<int64_t>> partitioner(const VectCont &dataset,
                                                       int64_t m) {
    std::vector<int64_t> ranking = rank_by_norm(dataset);
    std::vector<std::vector<int64_t>> partitions(m, std::vector<int64_t>(0));
    // put into partition based on norms.
    // vectors with smallest norms go in the first partition
    // vectors with largest norms go in the last partition
    int64_t current_partition = 0;
    for (size_t i = 0; i < dataset.size(); ++i) {
      // start adding to a new partition every m elements
      if (i != 0 && (i % (dataset.size() / m) == 0))
        ++current_partition;

      // add to partitions
      if (current_partition < m)
        partitions.at(current_partition).push_back(ranking.at(i));
      else
        // put any overflow into the last partition.
        partitions.at(m - 1).push_back(ranking.at(i));
    }
    return partitions;
  }

  template <typename VectCont>
  static std::pair<std::vector<std::vector<typename VectCont::value_type>>,
                   std::vector<typename VectCont::value_type::value_type>>
  normalizer(const VectCont &dataset,
             const std::vector<std::vector<int64_t>> &partitions) {
    /*
     * normalize partition using the largest norm in that partition.
     * returns normalized data and the normalizer U
     */
    using Vect = typename VectCont::value_type;
    std::vector<std::vector<Vect>> normalized_dataset(partitions.size(),
                                                      std::vector<Vect>(0));
    // stores normalizers, Up, for each partition
    std::vector<typename Vect::value_type> U(partitions.size());

    for (size_t p = 0; p < partitions.size(); ++p) {
      auto Up = IndexBuilder::max_norm(dataset, partitions.at(p));
      U.at(p) = Up;
      for (size_t i = 0; i < partitions.at(p).size(); ++i) {
        auto normalized = dataset.at(partitions.at(p).at(i)) / Up;
        normalized_dataset.at(p).push_back(normalized);
      }
    }
    return {normalized_dataset, U};
  }

  template <typename PartCont, typename Component>
  static std::vector<std::vector<int64_t>>
  simple_LSH_partitions(const PartCont &partitioned_dataset,
                        SimpleLSH<Component> hash, int64_t num_buckets) {
    const size_t m = partitioned_dataset.size();
    std::vector<std::vector<int64_t>> indices(m, std::vector<int64_t>(0));

    for (size_t j = 0; j < partitioned_dataset.size(); ++j) {
      size_t partition_size = partitioned_dataset.at(j).size();
      for (size_t p_idx = 0; p_idx < partition_size; ++p_idx) {
        const auto &item = partitioned_dataset.at(j).at(p_idx);
        const int64_t idx = hash.hash_max(item, num_buckets);
        indices.at(j).push_back(idx);
      }
    }
    return indices;
  }
};
} // namespace nr
