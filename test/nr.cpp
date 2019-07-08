
#include <cmath>

#include "../include/nr_gen.hpp"
#include "catch.hpp"

TEST_CASE("sets sizes based on probability", "NR-LSH tests") {
  std::pair<int64_t, int64_t> sizes;
  sizes = nr::sizes_from_probs(10, .9, .2);
  REQUIRE(sizes.first == std::floor(std::log2(10) / 2.322));
  double p = std::pow(10, std::log2(.9) / std::log2(.2));
  REQUIRE(sizes.second == std::floor(p));

  sizes = nr::sizes_from_probs(std::pow(2, 3), .9, .2);
  REQUIRE(sizes.first == 1);
  sizes = nr::sizes_from_probs(std::pow(2, 4), .9, .2);
  REQUIRE(sizes.first == 1);
  sizes = nr::sizes_from_probs(std::pow(2, 5), .9, .2);
  REQUIRE(sizes.first == 2);
  sizes = nr::sizes_from_probs(std::pow(2, 6), .9, .2);
  REQUIRE(sizes.first == 2);
  sizes = nr::sizes_from_probs(std::pow(2, 7), .9, .2);
  REQUIRE(sizes.first == 3);
  sizes = nr::sizes_from_probs(std::pow(2, 8), .9, .2);
  REQUIRE(sizes.first == 3);
  sizes = nr::sizes_from_probs(std::pow(2, 15), .9, .1);

  REQUIRE(sizes.second > 1);
}
