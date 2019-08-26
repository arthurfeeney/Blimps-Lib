
#pragma once

#include <Eigen/Core>
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <random>
#include <type_traits>

#include "lsh_family.hpp"
#include "normal_matrix.hpp"
#include "xf_or_xd.hpp"

/*
 * Sign LSH family.
 * Similar to SimpleLSH, but does not use P(), and a isn't dim + 1
 */

namespace mp = boost::multiprecision;
namespace nr {

template <typename Component> class SignLSH : public LSH_Family<Component> {
private:
  // use the proper matrix and vector type for component.
  using Matrix = typename MatrixXf_or_Xd<Component>::type;
  using Vect = typename VectorXf_or_Xd<Component>::type;

  Matrix a;
  int64_t bits;
  int64_t dim;
  std::vector<mp::cpp_int> bit_mask;

public:
  SignLSH(int64_t bits, int64_t dim)
      : a(Matrix(bits, dim)), bits(bits), dim(dim), bit_mask(bits) {
    NormalMatrix<Component> nm;
    nm.fill_matrix(a);
    fill_bit_mask();
  }

  int64_t bit_count() const { return bits; }

  int64_t dimension() const { return dim; }

  mp::cpp_int operator()(const Vect &input) const { return hash(input); }

  mp::cpp_int hash(const Vect &input) const {
    // with a large number of hashes, it can become larger than 64 bit.
    // have to use multiprecision.
    Vect prods = a * input;
    Vect bit_vect = numerals_to_bits(prods);
    return bits_to_num(bit_vect);
  }

  size_t hash_max(const Vect &input, size_t max) const {
    mp::cpp_int mp_hash = hash(input);
    mp::cpp_int residue = mp_hash % max;
    size_t idx = residue.convert_to<size_t>();
    return idx;
  }

  void fill_bit_mask() {
    // fill this->bit_mask with powers of 2.
    for (size_t i = 0; i < static_cast<size_t>(bits); ++i) {
      bit_mask.at(i) = mp::pow(mp::cpp_int(2), i);
    }
  }

  std::vector<mp::cpp_int> get_bit_mask() const { return bit_mask; }

  Vect numerals_to_bits(Vect input) const {
    // if a value is positive, it's bit is 1, otherwise 0.
    // can be in place because input arg is a copy.
    for (int64_t i = 0; i < input.rows(); ++i) {
      input(i) = input(i) >= 0 ? 1 : 0;
    }
    return input;
  }

  mp::cpp_int bits_to_num(const Vect &bits) const {
    // convert a string of bits to an integer.
    mp::cpp_int sum = 0;
    for (Eigen::Index i = 0; i < bits.size(); ++i) {
      mp::cpp_int bit = bits(i) - 1 >= 0 ? 1 : 0;
      mp::cpp_int val = 0;
      val = mp::multiply(val, bit, bit_mask.at(i));
      sum = mp::add(sum, sum, val);
    }
    return sum;
  }
};

} // namespace nr
