
#include "catch.hpp"

#include <cmath>
#include <list>
#include <vector>

#include "../include/stats/stats.hpp"

TEST_CASE("mean", "stats") {
  std::vector<int> a{1, 2, 3};
  REQUIRE(nr::stats::mean(a) == 2);

  std::vector<double> b{3, 4, 3.5};
  REQUIRE(nr::stats::mean(b) == 3.5);

  std::list<int> c{1, 1, 1, 1, 1, 1, 1};
  REQUIRE(nr::stats::mean(c) == 1);
}

TEST_CASE("variance", "stats") {
  std::vector<int> a{1, 2, 3, 4, 5};
  REQUIRE(nr::stats::variance(a) == 2);

  std::vector<int> b{11, 61, 72, 420, 520, 1000};
  REQUIRE(nr::stats::variance(b) - 121997.2 < .2);

  std::vector<double> c{-1234, -1000, -521, -242, .3,  .5,  1,    11,
                        61,    72,    99,   340,  420, 520, 9999, 23423.1};
  REQUIRE(nr::stats::variance(c) - 36765990.4 < .5);
}

TEST_CASE("stdev", "stats") {
  std::vector<int> a{1, 2, 3, 4, 5};
  REQUIRE(nr::stats::stdev(a) - std::pow(nr::stats::variance(a), 2) < .5);

  std::vector<double> b{-1234, -1000, -521, -242, .3,  .5,  1,    11,
                        61,    72,    99,   340,  420, 520, 9999, 23423.1};
  REQUIRE(nr::stats::stdev(b) - 6063 < .5);
}

TEST_CASE("median", "stats") {
  std::vector<int> a{1, 2, 3};
  REQUIRE(nr::stats::median(a) == 2);

  std::vector<int> b{1, 2, 3, 4};
  REQUIRE(nr::stats::median(b) == 2.5);

  std::vector<double> c{2.0, 1.3, -420, 11, 99};
  REQUIRE(nr::stats::median(c) - 2.0 < 1e-5);

  std::vector<double> d{2.0, 1.3, -420, 11, 99, 2.5};
  REQUIRE(nr::stats::median(d) - 2.25 < 1e-5);
}

TEST_CASE("histogram", "stats") {
  // histogram only works for integral values
  std::vector<int> a{1, 1, 1, 1, 1, 1, 1, 1, 2};
  REQUIRE(nr::stats::histogram(a) == std::vector<int64_t>{0, 8, 1});

  std::vector<int> b{1, 2, 1, 2, 1};
  REQUIRE(nr::stats::histogram(b) == std::vector<int64_t>{0, 3, 2});

  std::vector<int> c{10, 100, 10};

  std::vector<int64_t> expected(101, 0);
  expected[10] = 2;
  expected[100] = 1;
  REQUIRE(nr::stats::histogram(c) == expected);
}

TEST_CASE("mode", "stats") {
  // mode only works for integral values
  std::vector<int> a{1, 1, 1, 1, 2, 2, 2, 1, 1};
  REQUIRE(nr::stats::mode(a) == 1);

  // if more than one thing is most frequent, return the smallest
  std::vector<int> b{1, 2, 3};
  REQUIRE(nr::stats::mode(b) == 1);

  std::vector<int> b2{2, 1, 3};
  REQUIRE(nr::stats::mode(b) == 1);

  std::vector<int> c{10,  11, 12, 12, 11, 12, 9, 8,  8, 3, 2, 12, 9,
                     100, 1,  0,  4,  4,  12, 9, 12, 8, 7, 6, 5};
  REQUIRE(nr::stats::mode(c) == 12);
}

TEST_CASE("nonzero", "stats") {
  std::vector<int> a{1, 2, -3, 4, 0};
  REQUIRE(nr::stats::nonzero(a) == std::vector<int>{1, 2, -3, 4});

  std::vector<float> b{1.0, .001, 0, -0, -.001};
  REQUIRE(nr::stats::nonzero(b) == std::vector<float>{1.0, .001, -.001});
}

TEST_CASE("topk", "stats") {
  std::vector<int> a{1, 2, 3, 4, 5};
  REQUIRE(nr::stats::topk(3, a).first == std::vector{3, 4, 5});
  REQUIRE(nr::stats::topk(3, a).second == std::vector<size_t>{2, 3, 4});

  REQUIRE(nr::stats::topk(6, a).first == std::vector{1,2,3,4,5});
  REQUIRE(nr::stats::topk(6, a).second == std::vector<size_t>{0,1,2,3,4});

  std::vector<size_t> b{5, 600, 1, 0, 100, 20, 6, 99, 420};
  auto t1 = nr::stats::topk(1, b);
  auto t3 = nr::stats::topk(3, b);
  auto t5 = nr::stats::topk(5, b);

  REQUIRE(t1.first == std::vector<size_t>{600});
  REQUIRE(t1.second == std::vector<size_t>{1});

  REQUIRE(t3.first == std::vector<size_t>{100, 420, 600});
  REQUIRE(t3.second == std::vector<size_t>{4, 8, 1});

  REQUIRE(t5.first == std::vector<size_t>{20, 99, 100, 420, 600});
  REQUIRE(t5.second == std::vector<size_t>{5, 7, 4, 8, 1});
}
