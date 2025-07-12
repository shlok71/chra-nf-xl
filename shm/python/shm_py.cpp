#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "shm_index.h"

namespace py = pybind11;

PYBIND11_MODULE(shm_py, m) {
    m.doc() = "Python bindings for the CHRA-NF-XL SHM library";

    py::class_<SHMIndex>(m, "SHMIndex")
        .def(py::init<>())
        .def("insert", &SHMIndex::insert,
             py::arg("bhv"),
             "Inserts a BHV into the index.")
        .def("query", &SHMIndex::query,
             py::arg("query_bhv"), py::arg("k"),
             "Queries the index for the k nearest neighbors.");
}
