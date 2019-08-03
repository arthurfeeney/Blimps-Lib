
#pragma once

#include <iostream>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "kv_comparator.hpp"
#include "simple_lsh.hpp"
#include "stat_tracker.hpp"
#include "stats/stats.hpp"
#include "stats/topk.hpp"
#include "tables.hpp"

/*
 * Abstract base class for multiprobe tables.
 */

namespace nr {

template <typename Vect> class MultiProbe {
private:
  using Component = typename Vect::value_type;
  using KV = std::pair<Vect, int64_t>;

public:
  template <typename Cont> void fill(const Cont &data, bool is_normalized);

  virtual std::pair<std::optional<KV>, StatTracker> probe(const Vect &q,
                                                          int64_t adj) = 0;

  virtual std::pair<std::optional<std::vector<KV>>, StatTracker>
  k_probe(int64_t k, const Vect &q, size_t adj) = 0;

  virtual std::pair<std::optional<KV>, StatTracker>
  probe_approx(const Vect &q, Component c, int64_t adj) = 0;

  virtual std::pair<std::optional<std::vector<KV>>, StatTracker>
  k_probe_approx(int64_t k, const Vect &q, double c, size_t adj) = 0;

  virtual KV find_max_inner(const Vect &q) = 0;

  virtual void print_stats() = 0;

  virtual size_t num_tables() const = 0;
};

} // namespace nr
