
#include "stats.hpp"

namespace nr{
namespace stats{

short same_bits(size_t m, size_t n, size_t bits) {
  // if both have 0's and 1's in the same spot, similarity incremented.

  size_t one_bits = m & n; // both ones.
  size_t zero_bits = ~m & ~n; // both zero (including those past desired bits.)
  size_t match_bits = one_bits | zero_bits;

  short count = 0;
  // iterate through first 'bits' of the match_bits.
  for(size_t i = 0; i < bits; ++i) {
    count += match_bits & 1;
    match_bits >>= 1;
  }
  return count;
}

} // stats
} // nr
