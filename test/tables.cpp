
#include "catch.hpp"

#include "../include/tables.hpp"

#include <Eigen/Core>

TEST_CASE("sub tables rankings", "tables") {
  nr::Tables<Eigen::VectorXf> tables(2, 2, 3, 1);

  REQUIRE(tables.size() == 2);
}
