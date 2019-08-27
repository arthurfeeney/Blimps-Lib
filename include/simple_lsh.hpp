
#pragma once

#include <Eigen/Core>
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <random>
#include <type_traits>

#include "lsh_family.hpp"
#include "normal_matrix.hpp"
#include "sign_lsh.hpp"
#include "xf_or_xd.hpp"

/*
 * SimpleLSH family.
 */

namespace mp = boost::multiprecision;
namespace nr {

template <typename Component> class SimpleLSH : public LSH_Family<Component> {
private:
  // use the proper matrix and vector type for component.
  using Matrix = typename MatrixXf_or_Xd<Component>::type;
  using Vect = typename VectorXf_or_Xd<Component>::type;

  SignLSH<Component> sign_hash;

public:
  SimpleLSH(int64_t bits, int64_t dim) : sign_hash(bits, dim + 1) {}

  int64_t bit_count() const { return sign_hash.bit_count(); }

  int64_t dimension() const {
    // returns dimension of the vectors hashed.
    // Since P increases dimension by one, the input dimension is
    // sign_hashes dim minus one.
    return sign_hash.dimension() - 1;
  }

  mp::cpp_int operator()(const Vect &input) const { return hash(input); }

  mp::cpp_int hash(const Vect &input) const {
    // with a large number of hashes, it can become larger than 64 bit.
    // have to use multiprecision.
    Vect simple = P(input);
    return sign_hash(simple);
  }

  size_t hash_max(const Vect &input, size_t max) const {
    Vect simple = P(input);
    return sign_hash.hash_max(simple, max);
  }

  std::vector<mp::cpp_int> get_bit_mask() const {
    return sign_hash.get_bit_mask();
  }

  Vect P(const Vect &input) const {
    // symmetric transform that appends sqrt(1 - ||input||) to input
    Vect append(sign_hash.dimension());
    Component norm = input.norm();
    if (norm - 1 >= 1e-4)
      throw std::logic_error("SimpleLSH::P, Cannot take sqrt of negative");
    append << input, std::sqrt(1 - std::pow(norm, 2)); // append sqrt to input.
    return append;
  }

}; // namespace nr

} // namespace nr
