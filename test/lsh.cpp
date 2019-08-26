

#include "catch.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <list>
#include <utility>
#include <vector>

#include "../include/lsh.hpp"
#include "../include/p_stable_lsh.hpp"

using namespace nr;

TEST_CASE("build lsh multiprobe", "lsh") {
  LSH_MultiProbe<Eigen::VectorXf> l(10, 10, 100);
  REQUIRE(l.num_tables() == 1);
}

TEST_CASE("fill lsh multiprobe", "lsh") {
  /*
   * put everything into one bucket and make sure everything is there.
   */
  LSH_MultiProbe<Eigen::VectorXf> l(5, 3, 1);
  REQUIRE(l.num_tables() == 1);
  std::vector<Eigen::VectorXf> data{Eigen::VectorXf(3), Eigen::VectorXf(3),
                                    Eigen::VectorXf(3)};
  data.at(0) << .1, .1, .1;
  data.at(1) << .2, .3, .1;
  data.at(2) << .1, .3, .1;

  l.fill(data);
  auto bucket = l.data().at(0);
  for (const auto &item : bucket) {
    REQUIRE(std::find(data.begin(), data.end(), item.first) != data.end());
  }
}

TEST_CASE("single probe", "lsh") {
  /*
   * Since the query is (.1,.1,.1) and it searches both buckets, it
   * should return (.1, .1, .1) as the neighbor.
   */
  LSH_MultiProbe<Eigen::VectorXf> l(5, 3, 2);
  std::vector<Eigen::VectorXf> data{Eigen::VectorXf(3), Eigen::VectorXf(3),
                                    Eigen::VectorXf(3)};
  data.at(0) << .1, .1, .1;
  data.at(1) << .2, .3, .1;
  data.at(2) << .1, .3, .1;
  l.fill(data);
  Eigen::VectorXf query(3);
  query << .1, .1, .1;
  auto out = l.probe(query, 2);
  auto kv_opt = out.first;
  if (kv_opt) {
    auto kv = kv_opt.value();
    REQUIRE(kv.first == query);
  }
}

TEST_CASE("simple probe_approx", "lsh") {
  LSH_MultiProbe<Eigen::VectorXf> l(2, 3, 2);
  std::vector<Eigen::VectorXf> data{Eigen::VectorXf(3), Eigen::VectorXf(3),
                                    Eigen::VectorXf(3)};
  data.at(0) << .1, .1, .1;
  data.at(1) << .2, .3, .1;
  data.at(2) << .1, .3, .1;
  l.fill(data);
  Eigen::VectorXf query(3);
  query << .1, .2, .1;
  auto out = l.probe_approx(query, .1, 1);
  auto kv_opt = out.first;

  Eigen::VectorXf expected_out(3);
  expected_out << .1, .1, .1;

  if (kv_opt) {
    auto kv = kv_opt.value();
    REQUIRE(kv.first == expected_out);
  }
}

TEST_CASE("simple k_probe", "lsh") {
  LSH_MultiProbe<Eigen::VectorXf> l(10, 3, 2);
  std::vector<Eigen::VectorXf> data{Eigen::VectorXf(3), Eigen::VectorXf(3),
                                    Eigen::VectorXf(3)};
  data.at(0) << .1, .1, .1;
  data.at(1) << .1, -1, -.1;
  data.at(2) << .1, .3, .1;
  l.fill(data);
  Eigen::VectorXf query(3);
  query << .1, .2, .1;
  auto out = l.k_probe(2, query, 1);

  auto kv_opt = out.first;

  if (kv_opt) {
    auto kv_vect = kv_opt.value();
    REQUIRE(kv_vect.size() == 2);
    for (auto &kv : kv_vect) {
      bool good = kv.first == data.at(0) || kv.first == data.at(2);
      REQUIRE(good);
    }
  }
}

TEST_CASE("simple k_probe_approx", "lsh") {
  /*
   * This is a probability, so it can fail.
   * Depends on how the hash is initialized.
   * data.at(1) is longer so it's less likely to be in the same
   * bucket as the others.
   */
  LSH_MultiProbe<Eigen::VectorXf> l(2, 3, 2);
  std::vector<Eigen::VectorXf> data{Eigen::VectorXf(3), Eigen::VectorXf(3),
                                    Eigen::VectorXf(3)};
  data.at(0) << .1, .1, .1;
  data.at(1) << 1, -1, -1;
  data.at(2) << .1, .15, .1;
  l.fill(data);
  Eigen::VectorXf query(3);
  query << .1, .2, .1;
  auto out = l.k_probe_approx(2, query, .5, 1);

  auto kv_opt = out.first;

  if (kv_opt) {
    auto kv_vect = kv_opt.value();
    REQUIRE(kv_vect.size() == 2);
    for (auto &kv : kv_vect) {
      bool good = kv.first == data.at(0) || kv.first == data.at(2);
      REQUIRE(good);
    }
  }
}

TEST_CASE("use p_stable_lsh", "lsh") {
  LSH_MultiProbe<Eigen::VectorXf, PStableLSH<float>> l(2, 3, 2);

  std::vector<Eigen::VectorXf> data{Eigen::VectorXf(3), Eigen::VectorXf(3),
                                    Eigen::VectorXf(3)};
  data.at(0) << .1, .1, .1;
  data.at(1) << 1, -1, -1;
  data.at(2) << .1, .15, .1;

  l.fill(data);

  REQUIRE(l.num_tables() == 1);
}
