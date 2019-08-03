
#include "catch.hpp"

#include <cmath>
#include <list>
#include <utility>
#include <vector>

#include "../include/stats/stats.hpp"
//#include "../include/nr_lsh.hpp"

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

  REQUIRE(nr::stats::topk(6, a).first == std::vector{1, 2, 3, 4, 5});
  REQUIRE(nr::stats::topk(6, a).second == std::vector<size_t>{0, 1, 2, 3, 4});

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

TEST_CASE("topk with comp", "stats") {
  std::vector<int> a{1, 2, 3, 4, 5};
  auto out1 = nr::stats::topk(
      3, a, [](int a, int b) { return a < b; },
      [](int a, int b) { return a > b; });
  REQUIRE(out1.first == std::vector{3, 4, 5});
  REQUIRE(out1.second == std::vector<size_t>{2, 3, 4});

  std::vector<int> b{5, 4, 2, 6, 7, 1, 2};
  auto out2 = nr::stats::topk(
      3, b, [](int a, int b) { return a < b; },
      [](int a, int b) { return a > b; });
  REQUIRE(out2.first == std::vector{5, 6, 7});
  REQUIRE(out2.second == std::vector<size_t>{0, 3, 4});

  // find topk of pairs with custom comparators.
  using id = std::pair<int, double>;

  std::vector<id> c{std::make_pair(2, 0.0), std::make_pair(1, 1.0),
                    std::make_pair(3, .5), std::make_pair(0, .7)};
  // by first element in ascending order.
  auto out3 = nr::stats::topk(
                  2, c, [](id a, id b) { return a.first < b.first; },
                  [](id a, id b) { return a.first > b.first; })
                  .first;
  std::vector<id> expected3{std::make_pair(2, 0.0), std::make_pair(3, 0.5)};
  REQUIRE(out3.at(0).first == expected3.at(0).first);
  REQUIRE(out3.at(0).second == expected3.at(0).second);
  REQUIRE(out3.at(1).first == expected3.at(1).first);
  REQUIRE(out3.at(0).second == expected3.at(0).second);

  // by second element in ascending order.
  auto out4 = nr::stats::topk(
                  2, c, [](id a, id b) { return a.second < b.second; },
                  [](id a, id b) { return a.second > b.second; })
                  .first;
  std::vector<id> expected4{std::make_pair(0, 0.7), std::make_pair(1, 1.0)};
  REQUIRE(out4.at(0).first == expected4.at(0).first);
  REQUIRE(out4.at(0).second == expected4.at(0).second);
  REQUIRE(out4.at(1).first == expected4.at(1).first);
  REQUIRE(out4.at(0).second == expected4.at(0).second);

  std::vector<int> d{};
  REQUIRE(nr::stats::topk(
              3, d, [](int a, int b) { return a < b; },
              [](int a, int b) { return a > b; })
              .first == std::vector<int>{});

  std::vector<int> e{1};
  REQUIRE(nr::stats::topk(
              3, e, [](int a, int b) { return a < b; },
              [](int a, int b) { return a > b; })
              .first == std::vector{1});
}

TEST_CASE("unique", "stats") {
  std::vector<int> a{1, 1, 3, 2, 2, 3, 5};
  REQUIRE(nr::stats::unique(a).first == std::vector{1, 3, 2, 5});
  std::list<int> b{5, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 3, 3, 5, -1};
  REQUIRE(nr::stats::unique(b).first == std::list{5, 1, 2, 3, -1});
  std::list<size_t> c{1};
  REQUIRE(nr::stats::unique(c).first == std::list<size_t>{1});
  std::list<size_t> d{};
  REQUIRE(nr::stats::unique(d).first == std::list<size_t>{});

  // test if returned indices are correct
  REQUIRE(nr::stats::unique(a).second == std::vector<size_t>{0, 2, 3, 6});
  REQUIRE(nr::stats::unique(c).second == std::list<size_t>{0});
  REQUIRE(nr::stats::unique(d).second == std::list<size_t>{});

  std::vector<int> e{1, 2, 3, 5, 5, 5, 5, 6, 1};
  REQUIRE(nr::stats::unique(e).second == std::vector<size_t>{0, 1, 2, 3, 7});
}

TEST_CASE("unique with comparator", "stats") {
  using P = std::pair<int, double>;
  std::vector<P> a{std::make_pair(1, 0), std::make_pair(2, -1),
                   std::make_pair(2, 0)};
  std::vector<P> un_a =
      nr::stats::unique(a, [](P x, P y) { return x.first == y.first; }).first;
  std::vector<P> out_a{
      std::make_pair(1, 0),
      std::make_pair(2, -1),
  };
  REQUIRE(un_a.size() == out_a.size());
  REQUIRE(un_a.at(0).first == out_a.at(0).first);
  REQUIRE(un_a.at(0).second == out_a.at(0).second);
  REQUIRE(un_a.at(1).first == out_a.at(1).first);
  REQUIRE(un_a.at(1).second == out_a.at(1).second);
}

TEST_CASE("same bits", "simple_lsh") {
  int a = 0b1010;
  int b = 0b1000;
  REQUIRE(nr::stats::same_bits(a, b, 4) == 3);

  a = 0b0010;
  b = 0b1011;
  REQUIRE(nr::stats::same_bits(a, b, 4) == 2);

  a = 0b001001001;
  b = 0b101010101;
  REQUIRE(nr::stats::same_bits(a, b, 9) == 5);
}

TEST_CASE("partition simple", "stats") {
  std::vector<int> a{1, 3, 2};
  REQUIRE(nr::stats::partition(a, 0, a.size() - 1) == 1);

  std::vector<int> b{1, 2, 4, 3};
  REQUIRE(nr::stats::partition(b, 0, b.size() - 1) == 2);

  std::vector<int> c{1, 2, 4, 3, 5};
  REQUIRE(nr::stats::partition(c, 0, c.size() - 1) == 4);

  // 1.0 is the pivot and smallest, so pivot is at 0.
  std::vector<double> d{2.0, 3.1, 5, 3.9, 1.0};
  REQUIRE(nr::stats::partition(d, 0, c.size() - 1) == 0);
}

bool correct_partition(int64_t q, std::vector<int> &v) {
  for (size_t i = 0; i < v.size(); ++i) {
    if (i <= q && v[i] > v[q])
      return false;
    if (i > q && v[i] <= v[q])
      return false;
  }
  return true;
}

TEST_CASE("random partition", "stats") {
  std::vector<int> a{1, 2, 3};
  int64_t q = nr::stats::random_partition(a, 0, a.size() - 1);
  REQUIRE(correct_partition(q, a));

  for (size_t i = 0; i < 15; ++i) {
    std::vector<int> b{3, 2, 0, 5, 6, 7, 1, 10, 11, 19, 15, 23};
    q = nr::stats::random_partition(b, 0, b.size() - 1);
    REQUIRE(correct_partition(q, b));
  }
}

TEST_CASE("lower median simple", "stats") {
  std::vector<int> a{1, 2, 3};
  CHECK(nr::stats::lower_median(a) == 2);

  std::vector<int> b{2, 1, 4, 3, 10, 8, 7, 5, 9, 6};
  CHECK(nr::stats::lower_median(b) == 5);

  std::vector<int> c{1, 2, 3, 4};
  CHECK(nr::stats::lower_median(c) == 2);

  std::vector<int> d{1, 4, 3, 2};
  CHECK(nr::stats::lower_median(d) == 2);

  std::vector<int> e{1, 2, 3, 4, 5};
  CHECK(nr::stats::lower_median(e) == 3);

  std::vector<int> f{5, 4, 1, 2, 3};
  CHECK(nr::stats::lower_median(f) == 3);
}

TEST_CASE("upper median simple", "stats") {
  std::vector<int> a{1, 2, 3};
  CHECK(nr::stats::upper_median(a) == 2);

  std::vector<int> b{2, 1, 4, 3, 10, 8, 7, 5, 9, 6};
  CHECK(nr::stats::upper_median(b) == 6);

  std::vector<int> c{1, 2, 3, 4};
  CHECK(nr::stats::upper_median(c) == 3);

  std::vector<int> d{1, 4, 3, 2};
  CHECK(nr::stats::upper_median(d) == 3);

  std::vector<int> e{1, 2, 3, 4, 5};
  CHECK(nr::stats::upper_median(e) == 3);

  std::vector<int> f{5, 4, 1, 2, 3};
  CHECK(nr::stats::upper_median(f) == 3);
}

TEST_CASE("lower median = upper median with odd # of elements", "stats") {
  std::vector<int> a{1, 2, 3};
  CHECK(nr::stats::lower_median(a) == nr::stats::upper_median(a));

  std::vector<int> b{5, 4, 1, 2, 3};
  CHECK(nr::stats::lower_median(b) == nr::stats::upper_median(b));

  std::vector<int> c{5, 4, -9, 1, 2, 3, 11, 10, 22, 15, -1, 23, 99, 50, 51};
  CHECK(nr::stats::lower_median(c) == nr::stats::upper_median(c));
}
