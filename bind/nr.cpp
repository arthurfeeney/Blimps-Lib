
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <vector>

#include "../include/NR_multitable.hpp"
#include "../include/NR_multiprobe.hpp"

namespace py = pybind11;
using namespace nr;
using namespace Eigen;


PYBIND11_MODULE(nr_binding, m) {
    // double tables.
    py::class_<NR_MultiTable<VectorXd>>(m, "MultiTableDouble")
        .def(py::init<int64_t, int64_t, int64_t, int64_t>())
        .def("fill", &NR_MultiTable<VectorXd>::fill<std::vector<VectorXd>>)
        .def("MIPS", &NR_MultiTable<VectorXd>::MIPS);

    py::class_<NR_MultiProbe<VectorXd>>(m, "MultiProbeDouble")
        .def(py::init<int64_t, int64_t, int64_t, int64_t>())
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
        .def(py::init<int64_t, int64_t, int64_t, int64_t>())
        .def("fill", &NR_MultiProbe<VectorXf>::fill<std::vector<VectorXf>>)
        .def("probe", &NR_MultiProbe<VectorXf>::probe)
        .def("probe_approx", &NR_MultiProbe<VectorXf>::probe_approx)
        .def("k_probe_approx", &NR_MultiProbe<VectorXf>::k_probe_approx)
        .def("find_max_inner", &NR_MultiProbe<VectorXf>::find_max_inner)
        .def("stats", &NR_MultiProbe<VectorXf>::print_stats);
};
