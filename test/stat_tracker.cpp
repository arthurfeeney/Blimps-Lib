
#include "catch.hpp"
#include "utility"

#include "../include/stat_tracker.hpp"

size_t get_comps(StatTracker s) {
  std::tuple<size_t> stats = s.get_stats();
  size_t comp = std::get<0>(stats);
  return comp;
}

TEST_CASE("simple StatTracker", "stat_tracker") {
  StatTracker st;
  st.incr_comparisons();
  size_t comp = get_comps(st);
  REQUIRE(comp == 1);
  st.incr_comparisons();
  st.incr_comparisons();
  st.incr_comparisons();
  st.incr_comparisons();
  size_t comp2 = get_comps(st);
  REQUIRE(comp2 == 5);
}

TEST_CASE("add StatTrackers", "stat_tracker") {
  StatTracker s1;
  StatTracker s2;

  s1.incr_comparisons();
  s1.incr_comparisons();
  s1.incr_comparisons();

  REQUIRE(get_comps(s1) == 3);

  s2.incr_comparisons();
  s2.incr_comparisons();
  s2.incr_comparisons();
  s2.incr_comparisons();
  s2.incr_comparisons();

  REQUIRE(get_comps(s2) == 5);

  s1 += s2;
  REQUIRE(get_comps(s1) == 8);
}
