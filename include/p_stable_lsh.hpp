
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
 * p-stable LSH family.
 * Similar to SimpleLSH, but does not use P(), and a isn't dim + 1
 */

namespace mp = boost::multiprecision;
namespace nr {

template <typename Component> class PStableLSH : public LSH_Family<Component> {
private:
  // use the proper matrix and vector type for component.
  using Matrix = typename MatrixXf_or_Xd<Component>::type;
  using Vect = typename VectorXf_or_Xd<Component>::type;

  // The a_i must be drawn from a p-stable distribution.
  // a normal distribution satisfies this and is convenient.
  Vect a;
  Component b;
  Component r;
  int64_t dim;

public:
  PStableLSH(int64_t dim) : a(dim), dim(dim) {
    NormalMatrix<Component> nm;
    nm.fill_vector(a);
  }

  int64_t bit_count() const { return 1; }

  int64_t dimension() const { return dim; }

  mp::cpp_int operator()(const Vect &input) const { return hash(input); }

  mp::cpp_int hash(const Vect &input) const {
    // with a large number of hashes, it can become larger than 64 bit.
    // have to use multiprecision.
    Component h = floor((a.dot(input) + b) / r);
    return mp::cpp_int(h);
  }

  size_t hash_max(const Vect &input, size_t max) const {
    using mp = boost::multiprecision::cpp_int;
    mp mp_hash = hash(input);
    mp residue = mp_hash % max;
    size_t idx = residue.convert_to<size_t>();
    return idx;
  }
};

} // namespace nr
