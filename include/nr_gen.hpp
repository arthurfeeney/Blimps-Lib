
#include <cmath>
#include <iostream>
#include <utility>

#include "../include/nr_multiprobe.hpp"

namespace nr {

std::pair<int64_t, int64_t> sizes_from_probs(const size_t num_to_insert,
                                             const double p1, const double p2) {
  /*
   * Determines good values for k (number of bits) and L (number of tables)
   * based on the number of input vectors and the two threshold probabilities
   * p1 and p2.
   * p1 is the min probability that near vectors are inserted into the same
   * bucket and p2 is the max probability that distant vectors are inserted
   * into the same bucket.
   */
  const double n = num_to_insert;
  const int64_t bits = std::round(std::log2(n) / std::log2(1.0 / p2));
  const double row = std::log2(p1) / std::log2(p2);
  const int64_t num_tables = std::round(std::pow(n, row));
  return std::make_pair(bits, num_tables);
}

template <typename Vect>
NR_MultiProbe<Vect> nr_multiprobe_from_probs(int64_t num_partitions,
                                             int64_t dim, int64_t num_buckets,
                                             size_t num_to_insert, double p1,
                                             double p2) {
  std::pair<int64_t, int64_t> sizes = sizes_from_probs(num_to_insert, p1, p2);
  int64_t bits = sizes.first;
  int64_t num_tables = sizes.second;
  return NR_MultiProbe<Vect>(num_tables, num_partitions, bits, dim,
                             num_buckets);
}

} // namespace nr
