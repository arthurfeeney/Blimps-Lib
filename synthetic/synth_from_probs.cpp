
#include <Eigen/Core>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <matplotlibcpp.h>
#include <utility>

#include "../include/nr_gen.hpp"
#include "../include/nr_lsh.hpp"
#include "../include/stat_tracker.hpp"
#include "../include/stats/stats.hpp"
#include "synth.hpp"

using namespace Eigen;
namespace plt = matplotlibcpp;

int main() {

  /*
   * Generate data and fill NR-LSH table.
   */

  const size_t dim = 30;

  const std::vector<size_t> ks{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                               11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  const std::vector<double> rs{2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4.0};

  std::vector<std::vector<double>> ks_to_plot;
  std::vector<std::vector<double>> rs_to_plot;
  std::vector<std::vector<double>> recall_to_plot;

  for (size_t k : ks) {
    std::vector<double> ks_row, rs_row, recall_row;
    for (double r : rs) {

      std::vector<VectorXf> data = gen_data(std::pow(2, 15), dim);
      std::vector<VectorXf> queries = gen_data(100, dim);

      // table with:
      //  1. num_partitions, 2. dim, 3. num_buckets,
      //  4. num_to_insert, 5. p1, 6. p2
      auto sizes = nr::sizes_from_probs(data.size(), .9, .5);
      int64_t bits = sizes.first;
      int64_t num_tables = sizes.second;
      std::cout << '\n' << "(k, L): " << bits << ' ' << num_tables << "\n\n";
      nr::NR_MultiProbe<VectorXf> probe(num_tables, 1, bits, dim,
                                        std::pow(2, 15));
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
        auto topk_vects = nr::stats::topk(
                              k, data,
                              [&query](VectorXf x, VectorXf y) {
                                return query.dot(x) < query.dot(y);
                              },
                              [&query](VectorXf x, VectorXf y) {
                                return query.dot(x) > query.dot(y);
                              })
                              .first;
        for (auto &v : topk_vects) {
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
        for (size_t i = 0; i < opt_topk.size(); ++i) {
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
