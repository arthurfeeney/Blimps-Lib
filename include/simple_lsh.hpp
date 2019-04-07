
#pragma once

#include <Eigen/Core>
#include <random>
#include <type_traits>

#include "normal_matrix.hpp"
#include "xf_or_xd.hpp"

/*
 * SimpleLSH family.
 */

namespace nr {

template <typename Component> class SimpleLSH {
private:
  // use the proper matrix and vector type for component.
  using Matrix = typename MatrixXf_or_Xd<Component>::type;
  using Vect = typename VectorXf_or_Xd<Component>::type;

  Matrix a;
  int64_t bits;
  int64_t dim;
  Vect bit_mask;

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
    for (int64_t i = 0; i < bits; ++i) {
      bit_mask(i) = std::pow(2, i);
    }
  }

  Vect get_bit_mask() const { return bit_mask; }

  Vect P(Vect input) const {
    // symmetric transform that appends sqrt(1 - ||input||) to input
    Vect append(dim + 1);
    Component norm = input.norm();
    if (norm - 1 > .0001) {
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

  int64_t operator()(Vect input) const {
    Vect simple = P(input);
    Vect prods = a * simple;
    Vect bit_vect = numerals_to_bits(prods);
    return bit_vect.dot(bit_mask);
  }
};

} // namespace nr
