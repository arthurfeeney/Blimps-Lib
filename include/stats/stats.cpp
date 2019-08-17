
#include "stats.hpp"

namespace nr {
namespace stats {

short same_bits(size_t m, size_t n, size_t bits) {
  /*
   * checks which bits of m and n are the same.
   * the bits paramter is how many bits to compare.
   */
  size_t one_bits = m & n;    // both ones.
  size_t zero_bits = ~m & ~n; // both zero
  size_t match_bits = one_bits | zero_bits;

  // iterate through first 'bits' of the match_bits to find the count
  short count = 0;
  for (size_t i = 0; i < bits; ++i) {
    count += match_bits & 1;
    match_bits >>= 1;
  }
  return count;
}

} // namespace stats
} // namespace nr
