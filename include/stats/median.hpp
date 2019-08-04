
#pragma once

#include <algorithm>
#include <random>

/*
 * Implementations of median that has expected runtime of O(n)
 * Has worst-case time of O(n^2), but meh.
 * Basically from CLRS CH. 9
 */

namespace nr {
namespace stats {

template <typename Cont>
int64_t partition(Cont &c, const int64_t p, const int64_t r) {
  const auto x = c.at(r);
  int64_t i = p - 1;
  for (int64_t j = p; j <= r - 1; ++j) {
    if (c[j] <= x) {
      ++i;
      std::swap(c.at(i), c.at(j));
    }
  }
  std::swap(c.at(i + 1), c.at(r));
  return i + 1;
}

template <typename Cont>
int64_t random_partition(Cont &c, int64_t p, int64_t r) {
  // selects a random element of c to be the pivot used by partition.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int64_t> dis(p, r);
  int64_t i = dis(gen);
  std::swap(c.at(i), c.at(r));
  return partition(c, p, r);
}

template <typename Cont>
int64_t random_select(Cont &c, const int64_t p, const int64_t r,
                      const int64_t i) {
  // returns the ith smallest item in c.
  if (p == r) {
    return c[p];
  }
  const int64_t q = random_partition(c, p, r);
  const int64_t k = q - p + 1;
  if (i == k) {
    return c[q];
  } else if (i < k) {
    return random_select(c, p, q - 1, i);
  } else {
    return random_select(c, q + 1, r, i - k);
  }
}

template <template <typename Sub> typename Cont, typename Sub>
Sub lower_median(Cont<Sub> c) {
  // if there are an even number of elements, this function returns the
  // "lower" median. It returns this instead of the average of the two medians.
  // finds the floor(c.size()/2) smallest element
  int64_t i = c.size() / 2;
  if (c.size() % 2 == 1) {
    ++i;
  }
  return random_select(c, 0, c.size() - 1, i);
}

template <template <typename Sub> typename Cont, typename Sub>
Sub upper_median(Cont<Sub> c) {
  // if there are an even number of elements, this function returns the
  // "lower" median. It returns this instead of the average of the two medians.
  // finds the floor(c.size()/2) smallest element
  int64_t i = c.size() / 2 + 1;
  return random_select(c, 0, c.size() - 1, i);
}

template <template <typename Sub> typename Cont, typename Sub>
double median(Cont<Sub> &c) {
  /*
   * If there is an even number of elements, find mean of upper
   * and lower medians. Even though it does two operations, this
   * is still O(n). If there's an odd number of elements, the
   * upper and lower medians are the same.
   */
  if (c.size() % 2 == 0) {
    double n1 = static_cast<double>(lower_median(c));
    double n2 = static_cast<double>(upper_median(c));
    return (n1 + n2) / 2.0;
  }
  return static_cast<double>(lower_median(c));
}

} // namespace stats
} // namespace nr
