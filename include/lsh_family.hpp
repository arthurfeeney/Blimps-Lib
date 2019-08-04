
#pragma once

#include <Eigen/Core>
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <random>
#include <type_traits>

#include "normal_matrix.hpp"
#include "xf_or_xd.hpp"

/*
 * Abstract base class for LSH and familes.
 */

namespace mp = boost::multiprecision;
namespace nr {

template <typename Component> class LSH_Family {
private:
  // use the proper matrix and vector type for component.
  using Matrix = typename MatrixXf_or_Xd<Component>::type;
  using Vect = typename VectorXf_or_Xd<Component>::type;

public:
  virtual int64_t bit_count() const = 0;

  virtual int64_t dimension() const = 0;

  virtual mp::cpp_int operator()(const Vect &input) const = 0;

  virtual mp::cpp_int hash(const Vect &input) const = 0;

  virtual size_t hash_max(const Vect &input, size_t max) const = 0;
};

} // namespace nr
