
#include "catch.hpp"

#include <vector>

#include "../include/fast_sim.hpp"

TEST_CASE("simple zero high bits", "fast_sim") {
  REQUIRE(nr::zero_high_bits(5, 10) == 5);
  REQUIRE(nr::zero_high_bits(5, 0) == 0);
  REQUIRE(nr::zero_high_bits(8, 2) == 0);
  REQUIRE(nr::zero_high_bits(0b01101, 3) == 0b00101);
  REQUIRE(nr::zero_high_bits(0b110101011, 5) == 0b000001011);
}

TEST_CASE("simple fast_sim", "fast_sim") {
  auto sim = nr::fast_sim_1bit(1, 1);
  REQUIRE(sim == std::vector<size_t>{1, 0});
}

TEST_CASE("empty fast_sim", "fast_sim") {
  auto sim = nr::fast_sim_1bit(0b01110101101, 0);
  REQUIRE(sim == std::vector<size_t>{0});
}

TEST_CASE("harder fast_sim", "fast_sim") {
  auto sim = nr::fast_sim_1bit(0b01101, 5);
  REQUIRE(sim == std::vector<size_t>{0b01101, 0b01100, 0b01111, 0b01001,
                                     0b00101, 0b11101});
}

TEST_CASE("simple fast_sim_2bit", "fast_sim") {
  auto sim = nr::fast_sim_2bit(1, 1);
  REQUIRE(sim == std::vector<size_t>{1, 0});
}

TEST_CASE("empty fast_sim 2bit", "fast_sim") {
  auto sim = nr::fast_sim_1bit(0b01110101101, 0);
  REQUIRE(sim == std::vector<size_t>{0});
}

TEST_CASE("harder fast_sim 2bit", "fast_sim") {
  auto sim = nr::fast_sim_2bit(0b11, 2);
  std::vector<size_t> expected{0b11, 0b10, 0b00, 0b01};
  REQUIRE(sim == expected);
}

TEST_CASE("second harder fast_sim 2bit", "fast_sim") {
  auto sim = nr::fast_sim_2bit(0b10010111, 2);
  std::vector<size_t> expected{0b11, 0b10, 0b00, 0b01};
  REQUIRE(sim == expected);
}

TEST_CASE("third harder fast_sim 2bit", "fast_sim") {
  auto sim = nr::fast_sim_2bit(0b0101010101, 3);
  std::vector<size_t> expected{0b101, 0b100, 0b110, 0b000, 0b111, 0b011, 0b001};
  REQUIRE(sim == expected);
}
