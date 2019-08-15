
#include "catch.hpp"

#include <Eigen/Core>

#include "../include/p_stable_lsh.hpp"

TEST_CASE("simple hash", "p_stable_lsh") {
  nr::PStableLSH<float> h(3, 5.0);
  Eigen::VectorXf v(3);
  v << 1, 1, 1;

  size_t hash_value = h.hash_max(v, 3);
  REQUIRE(hash_value < 3);
  REQUIRE(hash_value >= 0);
}
