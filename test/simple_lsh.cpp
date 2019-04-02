
#include "catch.hpp"

#include <Eigen/Core>

#include "../include/simple_lsh.hpp"

TEST_CASE("construct SimpleLSH", "simple_lsh") {
  nr::SimpleLSH<float> a(3, 4);
  CHECK(a.bit_count() == 3);
  CHECK(a.dimension() == 4);
}

TEST_CASE("proper bit mask", "simple_lsh") {
  nr::SimpleLSH<float> a(5, 4);

  Eigen::VectorXf mask1 = a.get_bit_mask();
  REQUIRE(mask1(0) == 1);
  REQUIRE(mask1(1) == 2);
  REQUIRE(mask1(2) == 4);
  REQUIRE(mask1(3) == 8);
  REQUIRE(mask1(4) == 16);
}

TEST_CASE("P() - pre-processing function correct", "simple_lsh") {
  nr::SimpleLSH<float> a(3, 3);

  Eigen::VectorXf input(3);
  input << 1, 1, 1;

  input *= .75 / input.norm(); // force input to be < unit magnitude

  Eigen::VectorXf output(4);
  output = a.P(input);
  REQUIRE(output(0) == input(0));
  REQUIRE(output(1) == input(1));
  REQUIRE(output(2) == input(2));
  REQUIRE(output(3) == std::sqrt(1 - input.norm()));
}

TEST_CASE("numerals to bits", "simple_lsh") {
  nr::SimpleLSH<float> a(3, 3);

  Eigen::VectorXf input(5);
  input << 1, -1, 2, 0, 1;
  Eigen::VectorXf output = a.numerals_to_bits(input);
  REQUIRE(output(0) == 1);
  REQUIRE(output(1) == 0);
  REQUIRE(output(2) == 1);
  REQUIRE(output(3) == 1);
  REQUIRE(output(4) == 1);

  Eigen::VectorXf input2(7);
  input2 << 1, -1, -999999, .0001, -.1, -.001, 99999;
  Eigen::VectorXf output2 = a.numerals_to_bits(input2);
  REQUIRE(output2(0) == 1);
  REQUIRE(output2(1) == 0);
  REQUIRE(output2(2) == 0);
  REQUIRE(output2(3) == 1);
  REQUIRE(output2(4) == 0);
  REQUIRE(output2(5) == 0);
  REQUIRE(output2(6) == 1);
}
