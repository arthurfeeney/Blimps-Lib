
#pragma once

#include <Eigen/Core>
#include <boost/multiprecision/cpp_int.hpp>
#include <random>
#include <type_traits>

#include "normal_matrix.hpp"
#include "xf_or_xd.hpp"

/*
 * SimpleLSH family.
 */

namespace mp = boost::multiprecision;
namespace nr {

template <typename Component> class SimpleLSH {
private:
  // use the proper matrix and vector type for component.
  using Matrix = typename MatrixXf_or_Xd<Component>::type;
  using Vect = typename VectorXf_or_Xd<Component>::type;

  Matrix a;
  int64_t bits;
  int64_t dim;
  std::vector<mp::cpp_int> bit_mask;

public:
  SimpleLSH(int64_t bits, int64_t dim)
      : a(Matrix(bits, dim + 1)), bits(bits), dim(dim), bit_mask(bits) {
    NormalMatrix<Component> nm;
    nm.fill_matrix(a);
    fill_bit_mask();
  }

  int64_t bit_count() const { return bits; }

  int64_t dimension() const { return dim; }

  void fill_bit_mask() {
    // fill this->bit_mask with powers of 2.
    for (size_t i = 0; i < bits; ++i) {
      bit_mask.at(i) = mp::pow(mp::cpp_int(2), i);
    }
  }

  std::vector<mp::cpp_int> get_bit_mask() const { return bit_mask; }

  Vect P(Vect input) const {
    // symmetric transform that appends sqrt(1 - ||input||) to input
    Vect append(dim + 1);
    Component norm = input.norm();
    if (norm - 1 > .001) {
      std::cout << norm << '\n';
      throw std::logic_error("SimpleLSH::P, Cannot take sqrt of negative");
    }
    append << input, std::sqrt(1 - norm); // append sqrt to input.
    return append;
  }

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
    for (size_t i = 0; i < bits.size(); ++i) {
      mp::cpp_int bit = bits(i) - 1 >= 0 ? 1 : 0;
      mp::cpp_int val = 0;
      val = mp::multiply(val, bit, bit_mask.at(i));
      sum = mp::add(sum, sum, val);
    }
    return sum;
  }

  mp::cpp_int operator()(Vect input) const {
    // with a large number of hashes, it can become larger than 64 bit.
    // have to use multiprecision.
    Vect simple = P(input);
    Vect prods = a * simple;
    Vect bit_vect = numerals_to_bits(prods);
    std::cout << "bits" << bit_vect << '\n';
    return bits_to_num(bit_vect);
  }
};

} // namespace nr
