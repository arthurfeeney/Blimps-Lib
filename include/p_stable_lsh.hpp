
#pragma once

#include <Eigen/Core>
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <random>

#include "lsh_family.hpp"
#include "normal_matrix.hpp"
#include "xf_or_xd.hpp"

/*
 * p-stable LSH family.
 */

namespace mp = boost::multiprecision;
namespace nr {

template <typename Component> class PStableLSH : public LSH_Family<Component> {
private:
  // use the proper vector type for component.
  using Vect = typename VectorXf_or_Xd<Component>::type;

  // The a_i must be drawn from a p-stable distribution.
  // a normal distribution satisfies this and is convenient.
  Vect a;
  int64_t dim;
  Component r;
  Component b;

public:
  PStableLSH(Component r, int64_t dim) : a(dim), dim(dim), r(r) {
    NormalMatrix<Component> nm;
    nm.fill_vector(a);
    // b is selected uniformly from [0, r]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Component> dis(0.0, r);
    b = dis(gen);
  }

  int64_t bit_count() const { return 1; }

  int64_t dimension() const { return dim; }

  mp::cpp_int operator()(const Vect &input) const { return hash(input); }

  mp::cpp_int hash(const Vect &input) const {
    Component h = floor((a.dot(input) + b) / r);
    return mp::cpp_int(h);
  }

  size_t hash_max(const Vect &input, size_t max) const {
    mp::cpp_int mp_hash = hash(input);
    mp::cpp_int residue = mp_hash % max;
    if (residue < 1) {
      residue *= -1;
    }
    size_t idx = residue.convert_to<size_t>();
    return idx;
  }
};

} // namespace nr
