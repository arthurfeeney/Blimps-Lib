
#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "../include/nr_multiprobe.hpp"
#include "../include/nr_multitable.hpp"

namespace py = pybind11;
using namespace nr;
using namespace Eigen;

PYBIND11_MODULE(nr_binding, m) {

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
  py::class_<NR_MultiTable<VectorXd>>(m, "MultiTableDouble")
      .def(py::init<int64_t, int64_t, int64_t, int64_t>())
      .def("fill", &NR_MultiTable<VectorXd>::fill<std::vector<VectorXd>>)
      .def("MIPS", &NR_MultiTable<VectorXd>::MIPS);

  py::class_<NR_MultiProbe<VectorXd>>(m, "MultiProbeDouble")
      .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t>())
      .def("fill", &NR_MultiProbe<VectorXd>::fill<std::vector<VectorXd>>)
      .def("probe", &NR_MultiProbe<VectorXd>::probe)
      .def("probe_approx", &NR_MultiProbe<VectorXd>::probe_approx)
      .def("k_probe_approx", &NR_MultiProbe<VectorXd>::k_probe_approx)
      .def("find_max_inner", &NR_MultiProbe<VectorXd>::find_max_inner)
      .def("stats", &NR_MultiProbe<VectorXd>::print_stats);

  // float tables.
  py::class_<NR_MultiTable<VectorXf>>(m, "MultiTableFloat")
      .def(py::init<int64_t, int64_t, int64_t, int64_t>())
      .def("fill", &NR_MultiTable<VectorXf>::fill<std::vector<VectorXf>>)
      .def("MIPS", &NR_MultiTable<VectorXf>::MIPS);

  py::class_<NR_MultiProbe<VectorXf>>(m, "MultiProbeFloat")
      .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t>())
      .def("fill", &NR_MultiProbe<VectorXf>::fill<std::vector<VectorXf>>)
      .def("probe", &NR_MultiProbe<VectorXf>::probe)
      .def("probe_approx", &NR_MultiProbe<VectorXf>::probe_approx)
      .def("k_probe_approx", &NR_MultiProbe<VectorXf>::k_probe_approx)
      .def("find_max_inner", &NR_MultiProbe<VectorXf>::find_max_inner)
      .def("stats", &NR_MultiProbe<VectorXf>::print_stats);
};
