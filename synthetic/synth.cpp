
#include <Eigen/Core>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <utility>

#include "../include/nr_lsh.hpp"
#include "../include/stat_tracker.hpp"
#include "../include/stats/stats.hpp"

using namespace Eigen;

std::vector<VectorXf> gen_data(size_t num, size_t dim);
VectorXf exact_MIPS(VectorXf query, const std::vector<VectorXf> &data);
std::vector<VectorXf> k_exact_MIPS(VectorXf query,
                                   const std::vector<VectorXf> &data);

int main() {

  /*
   * Generate data and fill NR-LSH table.
   */

   size_t dim = 10;

  std::vector<VectorXf> data = gen_data((size_t)std::pow(2, 13), dim);
  std::vector<VectorXf> queries = gen_data(100, dim);

  // probe(num_tables, num_partitions, bits, dim, num_buckets)
  nr::NR_MultiProbe<VectorXf> probe(10, 1, 20, dim, std::pow(2, 13));
  probe.fill(data, false);

  /*
   * Perform MIPS in a variety of ways
   */


  std::cout << std::setprecision(2) << std::fixed;
  std::cout << '\n';


  std::vector<float> recalls(0);
  for (VectorXf &query : queries) {
    // query should be unit length!!
    query /= query.norm();

    auto topk_vects = nr::stats::topk(5, data,
                                  [&query](VectorXf x, VectorXf y) {
                                    return query.dot(x) < query.dot(y);
                                  },
                                  [&query](VectorXf x, VectorXf y) {
                                    return query.dot(x) > query.dot(y);
                                  }).first;

    for(auto& v : topk_vects) {
      std::cout << v.dot(query) << ' ';
    }
    std::cout << '\t';


    auto topk_and_tracker = probe.k_probe(5, query, 300);
    auto opt_topk = topk_and_tracker.first.value();

    for (auto &kv : opt_topk) {
      std::cout << kv.first.dot(query) << ' ';
    }
    std::cout << '\t';

    std::vector<VectorXf> predicted_topk(opt_topk.size());
    for(size_t i = 0; i < opt_topk.size(); ++i) {
      predicted_topk.at(i) = opt_topk.at(i).first;
    }


    float recall = nr::stats::recall(topk_vects, predicted_topk);

    std::cout << recall;
    recalls.push_back(recall);

    std::cout << '\n';
  }

  std::cout << "average recall: " << nr::stats::mean(recalls) << '\n';

  return 0;
}

std::vector<VectorXf> gen_data(size_t num, size_t dim) {
  std::vector<VectorXf> data(num, VectorXf(dim));
  nr::NormalMatrix<float> nm;
  for (auto &datum : data) {
    nm.fill_vector(datum); // datum is changed in-place
  }
  // vectors get normalized during insertion, before hashing.
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

std::vector<VectorXf> k_exact_MIPS(size_t k, VectorXf query,
                                   const std::vector<VectorXf> &data) {
  std::vector<float> prods(data.size());

  // compute all inner products
  for (size_t i = 0; i < data.size(); ++i) {
    prods.at(i) = query.dot(data.at(i));
  }

  auto tk = nr::stats::topk(k, prods);
  std::vector<size_t> indices = tk.second;

  std::vector<VectorXf> topk_vects(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    topk_vects.at(i) = data.at(indices.at(i));
  }

  return topk_vects;
}
