
#include <Eigen/Core>
#include <matplotlibcpp.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <utility>

#include "../include/nr_lsh.hpp"
#include "../include/stat_tracker.hpp"
#include "../include/stats/stats.hpp"

using namespace Eigen;
namespace plt = matplotlibcpp;

std::vector<VectorXf> gen_data(size_t num, size_t dim);
VectorXf exact_MIPS(VectorXf query, const std::vector<VectorXf> &data);
std::vector<VectorXf> k_exact_MIPS(VectorXf query,
                                   const std::vector<VectorXf> &data);

int main() {

  /*
   * Generate data and fill NR-LSH table.
   */

  const size_t dim = 30;



  const std::vector<size_t> ks {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
  };
  const std::vector<double> rs {2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4.0};

  std::vector<std::vector<double>> ks_to_plot;
  std::vector<std::vector<double>> rs_to_plot;
  std::vector<std::vector<double>> recall_to_plot;

  for(size_t k : ks) {
    std::vector<double> ks_row, rs_row, recall_row;
    for(double r : rs) {


      std::vector<VectorXf> data = gen_data(std::pow(2, 15), dim);
      std::vector<VectorXf> queries = gen_data(100, dim);

      // probe(num_tables, num_partitions, bits, dim, num_buckets)
      nr::NR_MultiProbe<VectorXf> probe(20, 1, 32, dim, std::pow(2, 15));
      probe.fill(data, false);

      std::cout << std::setprecision(2) << std::fixed;
      std::cout << '\n';


      std::vector<float> recalls(0);
      for (VectorXf &query : queries) {
        // query should be unit length!!
        query /= query.norm();

        /*
        * Find the true topk vectors. Print their products with q.
        */
        auto topk_vects = nr::stats::topk(k, data,
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


        /*
        * Find some decent vectors and print their products with q.
        */
        auto topk_and_tracker = probe.k_probe_approx(k, query, r, 100);
        auto opt_topk = topk_and_tracker.first.value();
        for (auto &kv : opt_topk) {
          std::cout << kv.first.dot(query) << ' ';
        }
        std::cout << '\t';

        /*
        * Find the Recall - the fraction of decent vectors in the true topk.
        */
        std::vector<VectorXf> predicted_topk(opt_topk.size());
        for(size_t i = 0; i < opt_topk.size(); ++i) {
          predicted_topk.at(i) = opt_topk.at(i).first;
        }
        float recall = nr::stats::recall(topk_vects, predicted_topk);

        std::cout << recall;
        recalls.push_back(recall);

        std::cout << '\n';
      }
      ks_row.push_back(k);
      rs_row.push_back(r);
      recall_row.push_back(nr::stats::mean(recalls));
    }
    ks_to_plot.push_back(ks_row);
    rs_to_plot.push_back(rs_row);
    recall_to_plot.push_back(recall_row);
  }

  plt::title("Recall of k_probe_approx");
  plt::xlabel("min");
  plt::ylabel("k");
  plt::plot_surface(rs_to_plot, ks_to_plot, recall_to_plot);
  plt::show();
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
