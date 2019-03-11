
#pragma once

#include <random>

/*
 * fills an Eigen matrix with random values from normal distribution.
 * Eigen seems to only support uniform random.
 */

namespace nr {

template <typename Component> class NormalMatrix {
private:
  std::random_device rd;
  std::mt19937 gen;
  std::normal_distribution<Component> d;

public:
  NormalMatrix(Component mean = 0.0, Component stddev = 1.0)
      : rd(), gen(rd()), d(0, 1) {}

  template <typename Matr> void fill_matrix(Matr &A) {
    for (int c = 0; c < A.cols(); ++c) {
      for (int r = 0; r < A.rows(); ++r) {
        A(r, c) = d(gen);
      }
    }
  }
};

} // namespace nr
