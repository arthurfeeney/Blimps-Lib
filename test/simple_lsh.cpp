
#include "catch.hpp"

#include <Eigen/Core>
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>

#include "../include/simple_lsh.hpp"

TEST_CASE("construct SimpleLSH", "simple_lsh") {
  nr::SimpleLSH<float> a(3, 4);
  CHECK(a.bit_count() == 3);
  CHECK(a.dimension() == 4);
}

TEST_CASE("proper bit mask", "simple_lsh") {
  nr::SimpleLSH<float> a(5, 4);

  auto mask1 = a.get_bit_mask();
  REQUIRE(mask1.at(0) == 1);
  REQUIRE(mask1.at(1) == 2);
  REQUIRE(mask1.at(2) == 4);
  REQUIRE(mask1.at(3) == 8);
  REQUIRE(mask1.at(4) == 16);
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
  REQUIRE(output(3) == Approx(std::sqrt(1 - std::pow(input.norm(), 2))));
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

TEST_CASE("many bit hash", "simple_lsh") {
  using mp = boost::multiprecision::cpp_int;

  nr::SimpleLSH<float> hash(128, 2);
  Eigen::VectorXf input(2);
  input << .3, .3;

  mp idx = hash(input);
  mp res = idx % 5;
  int out = res.convert_to<int>();
  REQUIRE(out < 5);
}

TEST_CASE("P(x) should be unit length, 2-dim", "simple_lsh") {
  nr::SimpleLSH<float> hash(32, 2);

  Eigen::VectorXf a(2);
  Eigen::VectorXf b(2);
  Eigen::VectorXf c(2);

  a << .3, .3;
  b << .1, -.1;
  c << -.5, -.2;

  auto append1 = hash.P(a);
  auto append2 = hash.P(b);
  auto append3 = hash.P(c);
  REQUIRE(append1.norm() == Approx(1));
  REQUIRE(append2.norm() == Approx(1));
  REQUIRE(append3.norm() == Approx(1));
}
