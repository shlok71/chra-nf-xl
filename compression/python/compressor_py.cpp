#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "compressor.h"

namespace py = pybind11;

PYBIND11_MODULE(compressor_py, m) {
    py::class_<Compressor>(m, "Compressor")
        .def(py::init<>())
        .def("compress", &Compressor::compress)
        .def("decompress", &Compressor::decompress);
}
