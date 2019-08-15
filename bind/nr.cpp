
#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "../include/lsh.hpp"
#include "../include/nr_multiprobe.hpp"
#include "../include/p_stable_lsh.hpp"
#include "../include/sign_lsh.hpp"
#include "../include/simple_lsh.hpp"
#include "../include/stats/stats.hpp"

namespace py = pybind11;
using namespace nr;
using namespace Eigen;

PYBIND11_MODULE(nr_binding, m) {
  // hello

  m.def("same_bits", &stats::same_bits);

  // Binding for SimpleLSH is primarly for testing it and plotting.
  // hash_max takes an extra argument to
  py::class_<SimpleLSH<float>>(m, "SimpleLSH")
      .def(py::init<int64_t, int64_t>())
      .def("bit_count", &SimpleLSH<float>::bit_count)
      .def("dimension", &SimpleLSH<float>::dimension)
      .def("hash", &SimpleLSH<float>::hash_max);

  py::class_<SignLSH<float>>(m, "SignLSH")
      .def(py::init<int64_t, int64_t>())
      .def("bit_count", &SignLSH<float>::bit_count)
      .def("dimension", &SignLSH<float>::dimension)
      .def("hash", &SignLSH<float>::hash_max);

  py::class_<PStableLSH<float>>(m, "PStableLSH")
      .def(py::init<int64_t, float>())
      .def("bit_count", &PStableLSH<float>::bit_count)
      .def("dimension", &PStableLSH<float>::dimension)
      .def("hash", &PStableLSH<float>::hash_max);

  // Utility class to allow users to stats.comparisons.
  py::class_<Tracked>(m, "Tracked")
      .def(py::init<size_t, size_t, size_t, size_t>())
      .def_readonly("comps", &Tracked::comparisons)
      .def_readonly("bucks", &Tracked::buckets_probed)
      .def_readonly("parts", &Tracked::partitions_probed)
      .def_readonly("tables", &Tracked::tables_probed);

  // binding for StatTracker so Python users have access to it!
  py::class_<StatTracker>(m, "StatTracker")
      .def(py::init<>())
      .def("get_stats", &StatTracker::get_stats)
      .def("tracked_stats", &StatTracker::tracked_stats);

  // double tables.
  py::class_<NR_MultiProbe<VectorXd>>(m, "MultiProbeDouble")
      .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t>())
      .def("fill", &NR_MultiProbe<VectorXd>::fill<std::vector<VectorXd>>)
      .def("probe", &NR_MultiProbe<VectorXd>::probe)
      .def("k_probe", &NR_MultiProbe<VectorXd>::k_probe)
      .def("probe_approx", &NR_MultiProbe<VectorXd>::probe_approx)
      .def("k_probe_approx", &NR_MultiProbe<VectorXd>::k_probe_approx)
      .def("find_max_inner", &NR_MultiProbe<VectorXd>::find_max_inner)
      .def("stats", &NR_MultiProbe<VectorXd>::print_stats);

  // float tables.
  py::class_<NR_MultiProbe<VectorXf>>(m, "MultiProbeFloat")
      .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t>())
      .def("fill", &NR_MultiProbe<VectorXf>::fill<std::vector<VectorXf>>)
      .def("probe", &NR_MultiProbe<VectorXf>::probe)
      .def("k_probe", &NR_MultiProbe<VectorXf>::k_probe)
      .def("probe_approx", &NR_MultiProbe<VectorXf>::probe_approx)
      .def("k_probe_approx", &NR_MultiProbe<VectorXf>::k_probe_approx)
      .def("find_max_inner", &NR_MultiProbe<VectorXf>::find_max_inner)
      .def("stats", &NR_MultiProbe<VectorXf>::print_stats);

  // double lsh
  py::class_<LSH_MultiProbe<VectorXd>>(m, "LSHProbeDouble")
      .def(py::init<int64_t, int64_t, size_t>())
      .def("fill", &LSH_MultiProbe<VectorXd>::fill<std::vector<VectorXd>>)
      .def("probe", &LSH_MultiProbe<VectorXd>::probe)
      .def("k_probe", &LSH_MultiProbe<VectorXd>::k_probe)
      .def("probe_approx", &LSH_MultiProbe<VectorXd>::probe_approx)
      .def("k_probe_approx", &LSH_MultiProbe<VectorXd>::k_probe_approx)
      .def("stats", &LSH_MultiProbe<VectorXd>::print_stats);

  // float lsh
  py::class_<LSH_MultiProbe<VectorXf>>(m, "LSHProbeFloat")
      .def(py::init<int64_t, int64_t, size_t>())
      .def(py::init<int64_t, int64_t>())
      .def("fill", &LSH_MultiProbe<VectorXf>::fill<std::vector<VectorXf>>)
      .def("probe", &LSH_MultiProbe<VectorXf>::probe)
      .def("k_probe", &LSH_MultiProbe<VectorXf>::k_probe)
      .def("probe_approx", &LSH_MultiProbe<VectorXf>::probe_approx)
      .def("k_probe_approx", &LSH_MultiProbe<VectorXf>::k_probe_approx)
      .def("stats", &LSH_MultiProbe<VectorXf>::print_stats);
}
