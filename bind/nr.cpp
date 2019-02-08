
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <vector>

#include "../include/NR_multitable.hpp"

namespace py = pybind11;

using Vect = Eigen::VectorXd;
using T = std::vector<Vect>;

PYBIND11_MODULE(nr, m) {
    py::class_<NR_MultiTable<Vect>>(m, "MultiTable")
        .def(py::init<int64_t, int64_t, int64_t, int64_t>())
        .def("fill", &NR_MultiTable<Vect>::fill<T>)
        .def("MIPS", &NR_MultiTable<Vect>::MIPS);

}
