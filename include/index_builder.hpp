
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
 * Contains operations for "index-building"
 *
 * Needs to be refactored, badly.
 */

namespace mp = boost::multiprecision;
namespace nr {

template <typename VectCont, typename IntCont>
auto max_norm(const VectCont &data, const IntCont &partition) {
  /*
   * Finds the largest norm in a partition of data.
   */
  using Scalar = typename VectCont::value_type::value_type;

  Scalar max = -1;
  for (size_t i = 0; i < partition.size(); ++i) {
    Scalar norm = data.at(partition.at(i)).norm();
    if (norm > max) {
      max = norm;
    }
  }

  return max;
}

template <template <typename Vect> typename VectCont, typename Vect>
std::vector<int64_t> rank_by_norm(const VectCont<Vect> &dataset) {
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
std::vector<std::vector<int64_t>> partitioner(const VectCont &dataset,
                                              int64_t m) {

  std::vector<int64_t> ranking = rank_by_norm(dataset);

  std::vector<std::vector<int64_t>> partitions(m, std::vector<int64_t>(0));

  // put into partition based on norms.
  // vectors with smallest norms go in the first partition
  // vectors with largest norms go in the last partition
  int64_t current_partition = 0;
  for (size_t i = 0; i < dataset.size(); ++i) {
    if (i != 0 && (i % (dataset.size() / m) == 0)) {
      ++current_partition;
    }

    if (current_partition < m) {
      partitions.at(current_partition).push_back(ranking.at(i));
    } else {
      // put any overflow into the last partition.
      partitions.at(m - 1).push_back(ranking.at(i));
    }
  }
  return partitions;
}

template <typename VectCont>
std::pair<std::vector<std::vector<typename VectCont::value_type>>,
          std::vector<typename VectCont::value_type::value_type>>
normalizer(const VectCont &dataset,
           const std::vector<std::vector<int64_t>> &partitions) {
  /*
   * normalize partition using the largest norm in that partition.
   * returns normalized data and the normalizer U
   *  */
  using Vect = typename VectCont::value_type;

  std::vector<std::vector<Vect>> normalized_dataset(partitions.size(),
                                                    std::vector<Vect>(0));

  // initialize partitions of normalized dataset.
  // done here b/c the last partition may be a different size.
  for (size_t p = 0; p < partitions.size(); ++p) {
    normalized_dataset.at(p) = std::vector<Vect>(partitions.at(p).size());
  }

  // stores normalizers, Up, for each partitions
  std::vector<typename Vect::value_type> U(partitions.size());

  for (size_t p = 0; p < partitions.size(); ++p) {

    auto Up = max_norm(dataset, partitions.at(p));

    U.at(p) = Up;

    for (size_t i = 0; i < partitions.at(p).size(); ++i) {
      auto normalized = dataset.at(partitions.at(p).at(i)) / Up;
      normalized_dataset.at(p).at(i) = normalized;
    }
  }
  return std::make_pair(normalized_dataset, U);
}

template <typename PartCont, typename Component>
std::vector<std::vector<int64_t>>
simple_LSH_partitions(const PartCont &partitioned_dataset,
                      SimpleLSH<Component> hash, int64_t num_buckets) {
  const size_t m = partitioned_dataset.size();

  std::vector<std::vector<int64_t>> indices(m, std::vector<int64_t>(0));

  for (size_t i = 0; i < m; ++i) {
    indices.at(i) = std::vector<int64_t>(partitioned_dataset.at(i).size());
  }

  // using mp = boost::multiprecision::cpp_int;
  for (size_t j = 0; j < partitioned_dataset.size(); ++j) {
    for (size_t p_idx = 0; p_idx < partitioned_dataset.at(j).size(); ++p_idx) {
      mp::cpp_int mp_hash = hash(partitioned_dataset.at(j).at(p_idx));
      mp::cpp_int residue = mp_hash % num_buckets;
      std::cout << mp_hash << '\n';
      int64_t idx = residue.convert_to<int64_t>();
      indices.at(j).at(p_idx) = idx;
    }
  }
  return indices;
}

} // namespace nr
