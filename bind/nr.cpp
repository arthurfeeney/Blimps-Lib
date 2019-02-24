
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <vector>

#include "../include/NR_multitable.hpp"
#include "../include/NR_multiprobe.hpp"

namespace py = pybind11;
using namespace nr;

using Vect = Eigen::VectorXd;
using T = std::vector<Vect>;

PYBIND11_MODULE(nr, m) {
    py::class_<NR_MultiTable<Vect>>(m, "MultiTable")
        .def(py::init<int64_t, int64_t, int64_t, int64_t>())
        .def("fill", &NR_MultiTable<Vect>::fill<T>)
        .def("MIPS", &NR_MultiTable<Vect>::MIPS);

    py::class_<NR_MultiProbe<Vect>>(m, "MultiProbe")
        .def(py::init<int64_t, int64_t, int64_t, int64_t>())
        .def("fill", &NR_MultiProbe<Vect>::fill<T>)
        .def("probe", &NR_MultiProbe<Vect>::probe)
        .def("probe_approx", &NR_MultiProbe<Vect>::probe_approx)
        .def("k_probe_approx", &NR_MultiProbe<Vect>::k_probe_approx)
        .def("stats", &NR_MultiProbe<Vect>::print_stats);
};
