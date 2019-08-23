

#pragma once

#include <cmath>
#include <vector>

namespace nr {

// may not be totally necessary, but it shouldn't slow it down either...
inline size_t low_bits(const size_t bit_lim) {
  /*
   * returns a number with bit_lim 1's in binary representation
   */
  if (bit_lim == 0) {
    return 0;
  }
  return static_cast<size_t>(std::pow(2, bit_lim) - 1);
}

inline size_t zero_high_bits(const size_t n, const size_t bit_lim) {
  return n & low_bits(bit_lim);
}

inline std::vector<size_t> fast_sim_1bit(const size_t n, const size_t bit_lim) {
  /*
   * Given a number, this computes all numbers that differ by 1 bit up to
   * bit_lim.
   */
  std::vector<size_t> out(bit_lim + 1, 0);
  out.at(0) = zero_high_bits(n, bit_lim);
  for (size_t i = 0; i < bit_lim; ++i) {
    out.at(i + 1) = zero_high_bits((n ^ (0b1 << i)), bit_lim);
  }
  return out;
}

inline std::vector<size_t> fast_sim_2bit(const size_t n, const size_t bit_lim) {
  /*
   * Given a number, this computes all numbers that differs by at most 2 bits up
   * to bit_lim
   */
  std::vector<size_t> out;
  // reserve memory now so it doesn't reallocate.
  // definitely over allocating, but not a huge deal.
  out.reserve(std::pow(bit_lim, 2));
  out.push_back(zero_high_bits(n, bit_lim));
  for (size_t i = 0; i < bit_lim; ++i) {
    for (size_t j = i; j < bit_lim; ++j) {
      out.push_back(zero_high_bits(n ^ ((0b1 << i) | (0b1 << j)), bit_lim));
    }
  }
  return out;
}

} // namespace nr
