
#include "catch.hpp"

#include <Eigen/Core>

#include "../include/table.hpp"

TEST_CASE("Table constructor", "table") {
  nr::SimpleLSH<float> a(3, 3);
  nr::Table<Eigen::VectorXf> t(a, 10);

  REQUIRE(t.size() == 10);
}

TEST_CASE("Table fill with non-normalized data", "table") {
  nr::SimpleLSH<float> hash(3, 3);
  nr::Table<Eigen::VectorXf> t(hash, 3);

  std::vector<Eigen::VectorXf> partition(3, Eigen::VectorXf(3));
  partition[0] << .3, .3, .3;
  partition[1] << 0, 0, 1;
  partition[2] << -.3, -.3, -.3;
  std::vector<int64_t> indices{0, 1, 2};
  std::vector<int64_t> ids{0, 1, 2};
  float Up = .75;

  t.fill(partition, indices, ids, Up, false);

  // 1 thing in each bucket
  REQUIRE(t.at(0).size() == 1);
  REQUIRE(t.at(1).size() == 1);
  REQUIRE(t.at(2).size() == 1);

  // things have the correct ids
  REQUIRE((*t.at(0).begin()).second == 0);
  REQUIRE((*t.at(1).begin()).second == 1);
  REQUIRE((*t.at(2).begin()).second == 2);

  // vectors properly normalized
  REQUIRE((*t.at(0).begin()).first == partition[0] * Up);
  REQUIRE((*t.at(1).begin()).first == partition[1] * Up);
  REQUIRE((*t.at(2).begin()).first == partition[2] * Up);
}

TEST_CASE("Table sim function", "table") {
  nr::SimpleLSH<float> hash(3, 3);
  nr::Table<Eigen::VectorXf> t(hash, 3);
  std::vector<Eigen::VectorXf> partition(3, Eigen::VectorXf(3));
  partition[0] << .3, .3, .3;
  partition[1] << 0, 0, 1;
  partition[2] << -.3, -.3, -.3;
  std::vector<int64_t> indices{0, 1, 2};
  std::vector<int64_t> ids{0, 1, 2};
  float Up = .75;
  t.fill(partition, indices, ids, Up, false);

  // ALL CODES SHOULD BE OF LENGTH 3 FOR THIS!!!
  // same or similar should be negative.
  CHECK(t.sim(1, 1) > 0);
  CHECK(t.sim(0b111, 0b111) > 0);

  // disimilar
  CHECK(t.sim(0b000, 0b000) > 0);
  CHECK(t.sim(0b100, 0b001) < 0);
}

TEST_CASE("high(er)-dim similarity test", "table") {
  nr::SimpleLSH<float> hash(6, 6);
  nr::Table<Eigen::VectorXf> t(hash, 6);
  std::vector<Eigen::VectorXf> partition(0, Eigen::VectorXf(6));
  std::vector<int64_t> indices(0);
  std::vector<int64_t> ids(0);
  float Up = .75;
  t.fill(partition, indices, ids, Up, false);

  // similar
  REQUIRE(t.sim(0b111000, 0b111000) > 0);

  // disimilar
  REQUIRE(t.sim(0b111000, 0b000111) < 0);
  REQUIRE(t.sim(0b111001, 0b100001) > 0);
}

TEST_CASE("Test table ranking", "table") {
  nr::SimpleLSH<float> hash(3, 3);
  nr::Table<Eigen::VectorXf> t(hash, 3);

  std::vector<Eigen::VectorXf> partition(3, Eigen::VectorXf(3));
  partition[0] << .3, .3, .3;
  partition[1] << 0, 0, 1;
  partition[2] << -.3, -.3, -.3;
  std::vector<int64_t> indices{0, 1, 2};
  std::vector<int64_t> ids{0, 1, 2};
  float Up = .75;
  t.fill(partition, indices, ids, Up, false);

  REQUIRE(t.probe_ranking(1, 3) == std::vector<int64_t>{1, 0, 2});

  REQUIRE(t.probe_ranking(2, 3) == std::vector<int64_t>{2, 0, 1});
}
