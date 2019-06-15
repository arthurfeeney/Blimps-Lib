
#pragma once

#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <utility>

#include "../include/nr_lsh.hpp"
#include "../include/stat_tracker.hpp"
#include "../include/stats/stats.hpp"

using namespace Eigen;

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
