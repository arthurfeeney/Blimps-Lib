
#include "catch.hpp"

#include <Eigen/Core>

#include "../include/lsh_multi.hpp"

using namespace Eigen;

std::vector<VectorXf> make_data() {
  std::vector<VectorXf> data(10, VectorXf(3));

  data.at(0) << 1, 1, 1;
  data.at(1) << .3, -1, .4;
  data.at(2) << -.3, -1, .4;
  data.at(3) << .5, .4, -.05;
  data.at(4) << .5, .4, -.05;
  data.at(5) << 0, .1, 0;
  data.at(6) << .3, .3, .7;
  data.at(7) << .4, .1, -.8;
  data.at(8) << -1, -.4, -.6;
  data.at(9) << .2, .8, -.09;

  return data;
}

TEST_CASE("lsh multi table construction", "") {
  nr::LSH_MultiProbe_MultiTable<VectorXf> lsh(2, 10, 10, 10);
  REQUIRE(lsh.num_tables() == 2);
}

TEST_CASE("lsh multi table fill", "lsh_multi") {
  nr::LSH_MultiProbe_MultiTable<VectorXf> lsh(2, 10, 3, 10);
  REQUIRE(lsh.num_tables() == 2);
  auto data = make_data();
  lsh.fill(data);
  // <= 10 since things may go into the same bucket.
  // definitely less than 10 though.
  REQUIRE(lsh.data().at(0).size() <= 10);
}

TEST_CASE("lsh multi simple probe", "lsh_multi") {
  nr::LSH_MultiProbe_MultiTable<VectorXf> lsh(2, 10, 3, 10);
  REQUIRE(lsh.num_tables() == 2);
  auto data = make_data();
  lsh.fill(data);

  VectorXf query(3);
  query << .3, -1, .4;

  auto out = lsh.probe(query, 1);
  REQUIRE(out.first.value().first == data.at(1));

  // should find a near neighbor
  VectorXf query2(3);
  query2 << .3, -.9, .4;
  out = lsh.probe(query2, 1);
  REQUIRE(out.first.value().first == data.at(1));
}

TEST_CASE("lsh multi simple probe approx", "lsh_multi") {
  nr::LSH_MultiProbe_MultiTable<VectorXf> lsh(2, 10, 3, 10);
  REQUIRE(lsh.num_tables() == 2);
  auto data = make_data();
  lsh.fill(data);

  VectorXf query(3);
  query << .3, -1, .4;

  auto out = lsh.probe_approx(query, 0.1, 4);
  REQUIRE(out.first.value().first == data.at(1));

  // should find a near neighbor
  VectorXf query2(3);
  query2 << .3, -.9, .4;
  out = lsh.probe_approx(query2, .2, 4);
  REQUIRE(out.first.value().first == data.at(1));
}
