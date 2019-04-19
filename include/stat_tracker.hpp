
#pragma once

/*
 * used to track of some info during probing.
 * Just used to help judge the quality of MIPS.
 * For instance, the number of comparisons should generally be very low.
 */

#include <utility>

namespace nr {

struct Tracked {
  // utility class.
  // Allows checking with t.comparisons, rather than std::get<0>(stats);
  const size_t comparisons;
  const size_t buckets_probed;
  const size_t partitions_probed;
  const size_t tables_probed;

  Tracked(size_t c, size_t b, size_t p, size_t t)
      : comparisons(c), buckets_probed(b), partitions_probed(p),
        tables_probed(t) {}
};

class StatTracker {
private:
  size_t comparisons = 0;
  size_t buckets_probed = 0;
  size_t partitions_probed = 0;
  size_t tables_probed = 0;

public:
  StatTracker() {}

  StatTracker &operator+=(StatTracker &other) {
    comparisons += other.comparisons;
    buckets_probed += other.buckets_probed;
    partitions_probed += other.partitions_probed;
    tables_probed += other.tables_probed;
    return *this;
  }

  void incr_comparisons() { ++comparisons; }
  void incr_buckets_probed() { ++buckets_probed; }
  void incr_partitions_probed() { ++partitions_probed; }
  void incr_tables_probed() { ++tables_probed; }

  void k_partitions_probed(size_t k) { partitions_probed += k; }

  std::tuple<size_t, size_t, size_t, size_t> get_stats() const {
    return {comparisons, buckets_probed, partitions_probed, tables_probed};
  }

  Tracked tracked_stats() const {
    return Tracked(comparisons, buckets_probed, partitions_probed,
                   tables_probed);
  }
};
} // namespace nr
