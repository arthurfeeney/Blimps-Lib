
#include <Eigen/Core>
#include <iostream>
#include <utility>

#include "../include/nr_lsh.hpp"
#include "../include/stat_tracker.hpp"
#include "../include/stats.hpp"

using namespace Eigen;

std::vector<VectorXf> gen_data(size_t num, size_t dim);
VectorXf exact_MIPS(VectorXf query, const std::vector<VectorXf> &data);

int main() {
  std::vector<VectorXf> data = gen_data(10000, 40);
  std::vector<VectorXf> queries = gen_data(100, 40);

  // probe(num_tables, num_partitions, bits, dim, num_buckets)
  nr::NR_MultiProbe<VectorXf> probe(1, 1, 32, 40, 10000);
  probe.fill(data, false);

  std::vector<size_t> comp_list(0);
  for (VectorXf &query : queries) {

    // query should be unit length!!
    query /= query.norm();

    auto found = probe.probe(query, 1000);
    if (found) {
      auto kv = found.value();
      auto mip = kv.first;
      std::cout << mip.dot(query) << ' ';
    }

    auto op_and_tracker = probe.probe_approx(query, 2, 300);
    auto op = op_and_tracker.first;
    nr::StatTracker t = op_and_tracker.second;
    nr::Tracked tr = t.tracked_stats();
    if (op) {
      auto kv = op.value();
      VectorXf mip = kv.first;
      VectorXf exact = exact_MIPS(query, data);
      std::cout << mip.dot(query) << ' ' << exact.dot(query) << ' ';
    }
    std::cout << tr.comparisons << '\n';
    comp_list.push_back(tr.comparisons);
  }
  float avg_comps = nr::stats::mean(comp_list);

  std::cout << "avg comps: " << avg_comps << '\n';

  probe.print_stats();

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

VectorXf exact_MIPS(VectorXf query, const std::vector<VectorXf> &data) {
  VectorXf max_found = data.at(0);
  float max_inner = -1;
  for (const VectorXf &v : data) {
    float inner = query.dot(v);
    if (inner > max_inner) {
      max_found = v;
      max_inner = inner;
    }
  }
  return max_found;
}
