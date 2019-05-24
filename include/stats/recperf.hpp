
#pragma once

#include <algorithm>

// functions to analyze the performance of recommendation systems.

namespace nr {
namespace stats {

template <template <typename Vect> typename Cont, typename Vect>
bool hit(size_t k, Vect i, const Cont<Vect> &others) {
  // return true if i is in topk of others.
}

} // namespace stats
} // namespace nr
