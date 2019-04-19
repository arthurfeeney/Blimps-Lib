
#include "catch.hpp"
#include "utility"

#include "../include/stat_tracker.hpp"

using namespace nr;

size_t get_comps(StatTracker s) {
  auto stats = s.get_stats();
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

TEST_CASE("track probes", "stat_tracker") {
  StatTracker st;
  for (size_t i = 0; i < 20; ++i) {
    st.incr_comparisons();
  }
  st.incr_buckets_probed();
  st.incr_buckets_probed();
  st.incr_buckets_probed();

  st.incr_partitions_probed();
  st.incr_partitions_probed();

  st.incr_tables_probed();
  st.incr_tables_probed();
  st.incr_tables_probed();
  st.incr_tables_probed();

  auto counts = st.get_stats();

  REQUIRE(std::get<0>(counts) == 20);
  REQUIRE(std::get<1>(counts) == 3);
  REQUIRE(std::get<2>(counts) == 2);
  REQUIRE(std::get<3>(counts) == 4);

  st.k_partitions_probed(4);

  auto new_counts = st.get_stats();

  REQUIRE(std::get<0>(new_counts) == 20);
  REQUIRE(std::get<1>(new_counts) == 3);
  REQUIRE(std::get<2>(new_counts) == 6);
  REQUIRE(std::get<3>(new_counts) == 4);

  Tracked t = st.tracked_stats();

  REQUIRE(std::get<0>(new_counts) == t.comparisons);
  REQUIRE(std::get<1>(new_counts) == t.buckets_probed);
  REQUIRE(std::get<2>(new_counts) == t.partitions_probed);
  REQUIRE(std::get<3>(new_counts) == t.tables_probed);
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
