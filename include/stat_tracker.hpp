
#pragma once

/*
 * used to track of some info during probing.
 * Just used to help judge the quality of MIPS.
 * For instance, the number of comparisons should generally be very low.
 */

#include <utility>

class StatTracker {
private:
  size_t comparisons = 0;
  // size_t buckets_probed = 0;
  // size_t partitions_probed = 0;

public:
  StatTracker() {}

  StatTracker &operator+=(StatTracker &other) {
    comparisons += other.comparisons;
    return *this;
  }

  void incr_comparisons() { ++comparisons; }

  std::tuple<size_t> get_stats() const { return comparisons; }
};
