
#include "catch.hpp"

#include "../include/tables.hpp"

#include <Eigen/Core>

TEST_CASE("sub tables rankings", "tables") {
  // 2 partitions, 2 bits, 3 dim, 2 buckets.
  nr::Tables<Eigen::VectorXf> tables(2, 2, 3, 2);

  std::vector<Eigen::VectorXf> data(10, Eigen::VectorXf(3));
  data[0] << .3, .3, .3;
  data[1] << 0, 0, 1;
  data[2] << -3, .3, -.78;
  data[3] << -.28, -.69, -.45;
  data[4] << .67, -.42, -.664;
  data[5] << .45, -.1, .3453;
  data[6] << 2, 2.1, -.363;
  data[7] << -.3, -3, -.324;
  data[8] << 1, -.78, -.3;
  data[9] << .5, .9, -.67;

  tables.fill(data, false);

  REQUIRE(tables.size() == 2);

  // 1 is more similar to 1 than 0.
  std::vector<std::vector<int64_t>> rank = tables.sub_tables_rankings(1, 2);
  CHECK(rank.at(0).at(0) == 1);
  CHECK(rank.at(0).at(1) == 0);
  CHECK(rank.at(1).at(0) == 1);
  CHECK(rank.at(1).at(1) == 0);
}

TEST_CASE("tables contains", "tables") {
  nr::Tables<Eigen::VectorXf> tables(2, 2, 3, 2);

  std::vector<Eigen::VectorXf> data(10, Eigen::VectorXf(3));
  data[0] << .3, .3, .3;
  data[1] << 0, .2, .9;
  data[2] << -.3, .3, -.78;
  data[3] << -.28, -.69, -.45;
  data[4] << .67, -.42, -.66;
  data[5] << .45, -.1, .345;
  data[6] << .2, .21, -.363;
  data[7] << -.3, -.3, -.324;
  data[8] << .1, -.78, -.3;
  data[9] << .5, .9, -.67;

  tables.fill(data, false);

  REQUIRE(tables.size() == 2);

  for (auto &datum : data) {
    REQUIRE(tables.contains(datum));
  }
}
