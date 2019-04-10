
#include "catch.hpp"

#include "../include/index_builder.hpp"
#include "../include/simple_lsh.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <vector>

TEST_CASE("finds obvious max norm", "index_builder") {
  std::vector<Eigen::VectorXf> data(3, Eigen::VectorXf(3));
  data[0] << 1, 1, 1;
  data[1] << .5, .1, .1;
  data[2] << -.3, -.3, .3;

  std::vector<int> partition(3);
  std::iota(partition.begin(), partition.end(), 0);

  float norm = nr::max_norm(data, partition);
  REQUIRE(norm == data[0].norm());
}

TEST_CASE("find max norm in subset", "index_buidler") {
  std::vector<Eigen::VectorXf> data(5, Eigen::VectorXf(3));
  data[0] << 1, 1, 1;
  data[1] << .5, .1, .1;
  data[2] << -.3, -.3, .3;
  data[3] << .1, .1, .1;
  data[4] << -.5, -.5, .5;

  std::vector<int> partition{1, 2, 4};

  float norm = nr::max_norm(data, partition);
  REQUIRE(norm == data[4].norm());
}

TEST_CASE("Properly rank 3 vectors by norm", "index_builder") {
  std::vector<Eigen::VectorXf> data(3, Eigen::VectorXf(3));
  data[0] << 1, 1, 1;
  data[1] << .1, .1, .1;
  data[2] << -.3, -.3, .3;

  std::vector<int64_t> ranking = nr::rank_by_norm(data);

  REQUIRE(ranking == std::vector<int64_t>{1, 2, 0});
}

TEST_CASE("rank more vectors by norm", " index_builder") {
  std::vector<Eigen::VectorXf> data(10, Eigen::VectorXf(3));
  data[0] << .4, .8, .9;  // 8
  data[1] << .3, .6, .3;  // 7
  data[2] << .3, -.5, .3; // 6
  data[3] << .2, .5, .3;  // 5
  data[4] << .1, .5, .3;  // 4
  data[5] << .2, -.2, .2; // 3
  data[6] << .1, .1, .1;  // 1
  data[7] << .1, .2, .1;  // 2
  data[8] << -1, -1, 1;   // 9
  data[9] << .1, 0, 0;    // 0

  std::vector<int64_t> ranking = nr::rank_by_norm(data);

  std::vector<int64_t> expected{9, 6, 7, 5, 4, 3, 2, 1, 0, 8};

  REQUIRE(ranking == expected);
}

TEST_CASE("Partition two vectors", "index_builder") {
  std::vector<Eigen::VectorXf> data(2, Eigen::VectorXf(3));
  data[0] << 1, 1, 1;
  data[1] << 2, 2, 2;

  std::vector<std::vector<int64_t>> p = nr::partitioner(data, 2);

  CHECK(p.size() == 2);
  CHECK(p.at(0).size() == 1);
  CHECK(p.at(1).size() == 1);
  REQUIRE(p.at(0).at(0) == 0);
  REQUIRE(p.at(1).at(0) == 1);
}

TEST_CASE("Partition 10 vectors into 2 partitions", "index_builder") {
  std::vector<Eigen::VectorXf> data(10, Eigen::VectorXf(3));
  data[0] << .4, .8, .9;  // 8
  data[1] << .3, .6, .3;  // 7
  data[2] << .3, -.5, .3; // 6
  data[3] << .2, .5, .3;  // 5
  data[4] << .1, .5, .3;  // 4
  data[5] << .2, -.2, .2; // 3
  data[6] << .1, .1, .1;  // 1
  data[7] << .1, .2, .1;  // 2
  data[8] << -1, -1, 1;   // 9
  data[9] << .1, 0, 0;    // 0

  auto parts = nr::partitioner(data, 2);

  std::vector<std::vector<int64_t>> expected{{9, 6, 7, 5, 4}, {3, 2, 1, 0, 8}};

  REQUIRE(parts == expected);
}

TEST_CASE("Partition 10 vectors into 3 partitions", "index_builder") {
  std::vector<Eigen::VectorXf> data(10, Eigen::VectorXf(3));
  data[0] << .4, .8, .9;  // 8
  data[1] << .3, .6, .3;  // 7
  data[2] << .3, -.5, .3; // 6
  data[3] << .2, .5, .3;  // 5
  data[4] << .1, .5, .3;  // 4
  data[5] << .2, -.2, .2; // 3
  data[6] << .1, .1, .1;  // 1
  data[7] << .1, .2, .1;  // 2
  data[8] << -1, -1, 1;   // 9
  data[9] << .1, 0, 0;    // 0

  auto parts = nr::partitioner(data, 3);

  // overflow should go in the last partition.
  std::vector<std::vector<int64_t>> expected{
      {9, 6, 7}, {5, 4, 3}, {2, 1, 0, 8}};

  REQUIRE(parts == expected);
}

TEST_CASE("Normalize two partitions", "index_builder") {
  std::vector<Eigen::VectorXf> data(2, Eigen::VectorXf(3));
  data[0] << 1, 1, 1;
  data[1] << 2, 2, 2;

  std::vector<std::vector<int64_t>> p = nr::partitioner(data, 2);

  auto n_U = nr::normalizer(data, p);

  std::vector<std::vector<Eigen::VectorXf>> norm = n_U.first;
  std::vector<float> U = n_U.second;

  // normalizer used by partition is the only thing in the partition!
  REQUIRE(U.at(0) == data.at(0).norm());
  REQUIRE(U.at(1) == data.at(1).norm());

  REQUIRE(norm.at(0).at(0) == data.at(0) / U.at(0));
}

TEST_CASE("normalize ten vectors in 3 partitions", "index_builder") {
  std::vector<Eigen::VectorXf> data(10, Eigen::VectorXf(3));
  data[0] << .4, .8, .9;  // 8
  data[1] << .3, .6, .3;  // 7
  data[2] << .3, -.5, .3; // 6
  data[3] << .2, .5, .3;  // 5
  data[4] << .1, .5, .3;  // 4
  data[5] << .2, -.2, .2; // 3
  data[6] << .1, .1, .1;  // 1
  data[7] << .1, .2, .1;  // 2
  data[8] << -1, -1, 1;   // 9
  data[9] << .1, 0, 0;    // 0

  auto parts = nr::partitioner(data, 3);

  auto n_U = nr::normalizer(data, parts);
  auto norm = n_U.first;
  auto U = n_U.second;

  REQUIRE(norm.at(2).at(2) == data.at(0) / U.at(2));
  REQUIRE(norm.at(2).at(3) == data.at(8) / U.at(2));

  REQUIRE(norm.at(0).at(0) == data.at(9) / U.at(0));
  REQUIRE(norm.at(0).at(1) == data.at(6) / U.at(0));
}

TEST_CASE("hash two single-element partitions", "index_builder") {
  std::vector<Eigen::VectorXf> data(2, Eigen::VectorXf(3));
  data[0] << 1, 1, 1;
  data[1] << -2, -2, -2;

  std::vector<std::vector<int64_t>> p = nr::partitioner(data, 2);

  auto n_U = nr::normalizer(data, p);

  std::vector<std::vector<Eigen::VectorXf>> norm = n_U.first;
  std::vector<float> U = n_U.second;

  nr::SimpleLSH<float> hash(2, 3);

  size_t num_buckets = 8;

  int64_t h1 = static_cast<int64_t>(hash(norm.at(0).at(0)) % num_buckets);
  int64_t h2 = static_cast<int64_t>(hash(norm.at(1).at(0)) % num_buckets);
  REQUIRE(h1 < 4);
  REQUIRE(h1 >= 0);
  REQUIRE(h2 < 4);
  REQUIRE(h2 >= 0);

  std::vector<std::vector<int64_t>> idx =
      nr::simple_LSH_partitions(norm, hash, num_buckets);

  REQUIRE(idx.at(0).at(0) == h1);
  REQUIRE(idx.at(1).at(0) == h2);
}

TEST_CASE("hash 3 ten element partitions", "index_builder") {
  std::vector<Eigen::VectorXf> data(10, Eigen::VectorXf(3));
  data[0] << .4, .8, .9;  // 8
  data[1] << .3, .6, .3;  // 7
  data[2] << .3, -.5, .3; // 6
  data[3] << .2, .5, .3;  // 5
  data[4] << .1, .5, .3;  // 4
  data[5] << .2, -.2, .2; // 3
  data[6] << .1, .1, .1;  // 1
  data[7] << .1, .2, .1;  // 2
  data[8] << -1, -1, 1;   // 9
  data[9] << .1, 0, 0;    // 0

  auto parts = nr::partitioner(data, 3);

  auto n_U = nr::normalizer(data, parts);
  auto norm = n_U.first;
  auto U = n_U.second;

  nr::SimpleLSH<float> hash(6, 3);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      REQUIRE(hash(norm.at(i).at(j)) < 64);
      REQUIRE(hash(norm.at(i).at(j)) >= 0);
    }
  }
  REQUIRE(hash(norm.at(2).at(3)) < 64);
  REQUIRE(hash(norm.at(2).at(3)) >= 0);

  std::vector<std::vector<int64_t>> idx =
      nr::simple_LSH_partitions(norm, hash, 64);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      REQUIRE(hash(norm.at(i).at(j)) == idx.at(i).at(j));
    }
  }
  REQUIRE(hash(norm.at(2).at(3)) == idx.at(2).at(3));
}
