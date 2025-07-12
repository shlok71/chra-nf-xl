#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pmr.h"

namespace py = pybind11;

PYBIND11_MODULE(router_py, m) {
    m.doc() = "Python bindings for the CHRA-NF-XL Router library";

    py::class_<ProbabilisticMaskRouter>(m, "ProbabilisticMaskRouter")
        .def(py::init<const std::string&>(), py::arg("model_path"))
        .def("get_mask", &ProbabilisticMaskRouter::get_mask,
             py::arg("bhv"),
             "Generates a 128-bit mask with 4 bits set to 1.");
}
