
#include <Eigen/Core>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <utility>

#include "../include/nr_lsh.hpp"
#include "../include/stat_tracker.hpp"
#include "../include/stats.hpp"

using namespace Eigen;

std::vector<VectorXf> gen_data(size_t num, size_t dim);
VectorXf exact_MIPS(VectorXf query, const std::vector<VectorXf> &data);

int main() {

  /*
   * Generate data and fill NR-LSH table.
   */

  std::vector<VectorXf> data = gen_data(10000, 40);
  std::vector<VectorXf> queries = gen_data(100, 40);

  // probe(num_tables, num_partitions, bits, dim, num_buckets)
  nr::NR_MultiProbe<VectorXf> probe(20, 1, 32, 40, 15000);
  probe.fill(data, false);

  /*
   * Perform MIPS in a variety of ways
   *
   */

  std::cout << std::setprecision(2) << std::fixed;
  std::cout << '\n';

  std::vector<size_t> comp_list(0);
  for (VectorXf &query : queries) {

    // query should be unit length!!
    query /= query.norm();

    // probe a constant number of buckets.
    // returns large IP found in buckets searched
    auto start = std::chrono::high_resolution_clock::now();
    auto found = probe.probe(query, 1000);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (found) {
      auto kv = found.value();
      auto mip = kv.first;
      std::cout << mip.dot(query) << ' ';
    }
    std::cout << duration.count() << "ms \t";

    // probe constant number of buckets until a MIP > constant is found
    auto op_and_tracker = probe.probe_approx(query, 3, 200);
    auto op = op_and_tracker.first;
    nr::StatTracker t = op_and_tracker.second;
    nr::Tracked tr = t.tracked_stats();
    if (op) {
      auto kv = op.value();
      VectorXf mip = kv.first;
      std::cout << mip.dot(query);
    } else {
      std::cout << "    "; // 4 spaces for blank.
    }
    std::cout << '\t';

    auto max = probe.find_max_inner(query);
    std::cout << max.first.dot(query) << ' ';

    // the EXACT MIPS with the query vector.
    VectorXf exact = exact_MIPS(query, data);
    std::cout << exact.dot(query) << '\t';

    // # of comparisions during approx probing.
    std::cout << "Approx Comps: " << tr.comparisons << '\t';
    comp_list.push_back(tr.comparisons);

    // probe constant number of buckets until 5 MIP > constant are found.
    start = std::chrono::high_resolution_clock::now();
    auto topk_and_tracker = probe.k_probe_approx(5, query, 3, 200);
    end = std::chrono::high_resolution_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto opt_topk = topk_and_tracker.first;

    std::cout << duration.count() << "ms" << ' ';
    if (opt_topk) {
      for (auto &kv : opt_topk.value()) {
        std::cout << kv.first.dot(query) << ' ';
      }
    }
    std::cout << '\n';
  }

  float avg_comps = nr::stats::mean(comp_list);

  std::cout << "avg comps: " << avg_comps << '\n';

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
