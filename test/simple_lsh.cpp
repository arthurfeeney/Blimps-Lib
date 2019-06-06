
#include "catch.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <vector>
#include <unordered_map>

#include "../include/nr_lsh.hpp"
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

short popcount(int64_t n) {
  // replacement for __builtin_popcount available in gcc.
  // returns short because there aren't many digits.
  short count = 0;
  while (n) {
    count += n & 1; // adds 1 if bit is positive.
    n >>= 1;        // shift right 1.
  }
  return count;
}

short same_bits(size_t m, size_t n, int64_t bits) {
  // if both have 0's and 1's in the same spot, similarity incremented.
  short count = 0;

  size_t one_bits = m & n; // both ones.
  size_t zero_bits = ~m & ~n; // zero bits. (including those past true range.)

  size_t match_bits = one_bits | zero_bits;

  // iterate through first 'bits' of the matching bits.
  for(int i = 0; i < bits; ++i) {
    count += match_bits & 1;
    match_bits >>= 1;
  }
  return count;
}

TEST_CASE("same bits", "simple_lsh") {
  int a = 0b1010;
  int b = 0b1000;
  REQUIRE(same_bits(a, b, 4) == 3);

  a = 0b0010;
  b = 0b1011;
  REQUIRE(same_bits(a, b, 4) == 2);

  a = 0b001001001;
  b = 0b101010101;
  REQUIRE(same_bits(a, b, 9) == 5);
}


TEST_CASE("large inner have similar hash", "simple_lsh") {

  using mp = boost::multiprecision::cpp_int;

  nr::SimpleLSH<float> hash(4, 3);

  Eigen::VectorXf a(3);
  Eigen::VectorXf b(3);
  Eigen::VectorXf c(3);

  a << .3, .3, -.3;
  b << .3, .3, -.4;
  c << -.3, -.2, .5;


  // large inner product should be similar.
  a /= a.norm(); // normalize query vector
  b /= (b.norm() + .03);

  mp h1 = hash(a);
  mp residue = h1;
  size_t idx1 = residue.convert_to<size_t>();

  mp h2 = hash(b);
  mp residue2 = h2;
  size_t idx2 = residue.convert_to<size_t>();

  mp h3 = hash(c);
  residue = h3;
  size_t idx3 = residue.convert_to<size_t>();


  REQUIRE(same_bits(idx1, idx2, 4) > same_bits(idx1, idx3, 4));
}

TEST_CASE("vectors w/ large inner have similar hashes", "simpe_lsh") {
  nr::SimpleLSH<float> hash(8, 50);
  nr::NormalMatrix<float> nm;

  // make a random query vector
  Eigen::VectorXf q(50);
  nm.fill_vector(q);
  q /= q.norm();

  size_t count = 3;

  // generate data.
  std::vector<Eigen::VectorXf> data(count, Eigen::VectorXf(50));
  for(auto &datum : data) {
    nm.fill_vector(datum);
    datum /= datum.norm() + .1;
  }

  std::vector<size_t> before_inner(count);
  std::iota(before_inner.begin(), before_inner.end(), 0);

  // sort indices by inner product
  std::sort(before_inner.begin(), before_inner.end(),
            [&data, &q](size_t x, size_t y) {
              return q.dot(data.at(x)) < q.dot(data.at(y));
            });
  // just check its in ascending order.
  REQUIRE(q.dot(data.at(before_inner.at(0))) < q.dot(data.at(before_inner.at(1))));

  // sort indices by hash similarity. (sim of matching bits.)
  std::vector<size_t> before_hash_sim(count);
  std::iota(before_hash_sim.begin(), before_hash_sim.end(), 0);
  std::sort(before_hash_sim.begin(),
            before_hash_sim.end(),
            [&hash, &q, &data](size_t x, size_t y) {
              using mp = boost::multiprecision::cpp_int;
              mp h1 = hash(q);
              mp residue = h1;
              size_t idx1 = residue.convert_to<size_t>();

              mp h2 = hash(data.at(x));
              residue = h2;
              size_t idx2 = residue.convert_to<size_t>();

              mp h3 = hash(data.at(y));
              residue = h3;
              size_t idx3 = residue.convert_to<size_t>();

              return same_bits(idx1, idx2, 8) < same_bits(idx1, idx3, 8);
            });


}
