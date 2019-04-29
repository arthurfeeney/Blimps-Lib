
#include <Eigen/Core>
#include <iostream>
#include <utility>

#include "../include/nr_lsh.hpp"
#include "../include/stat_tracker.hpp"

using namespace Eigen;

std::vector<VectorXf> gen_data(size_t num, size_t dim);

int main() {
  std::vector<VectorXf> data = gen_data(10000, 40);
  std::vector<VectorXf> queries = gen_data(100, 40);

  // probe(num_tables, num_partitions, bits, dim, num_buckets)
  nr::NR_MultiProbe<VectorXf> probe(4, 128, 32, 40, 100);
  probe.fill(data, false);

  for (VectorXf &query : queries) {
    query /= query.norm();
    auto op_and_tracker = probe.probe_approx(query, 30, 20);
    auto op = op_and_tracker.first;
    nr::StatTracker t = op_and_tracker.second;
    nr::Tracked tr = t.tracked_stats();
    if (op) {
      auto kv = op.value();
      VectorXf mip = kv.first;
      std::cout << mip.dot(query) << '\n';
    }
    std::cout << tr.comparisons << '\n';
  }

  return 0;
}

std::vector<VectorXf> gen_data(size_t num, size_t dim) {
  // Random is uniform [-1,1].
  std::vector<VectorXf> data(num, VectorXf(dim));
  nr::NormalMatrix<float> nm;
  for (auto &datum : data) {
    nm.fill_vector(datum); // changed in-place
  }
  return data;
}
