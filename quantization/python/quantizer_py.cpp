#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "quantizer.h"

namespace py = pybind11;

PYBIND11_MODULE(quantizer_py, m) {
    py::class_<Quantizer>(m, "Quantizer")
        .def(py::init<>())
        .def("quantize", &Quantizer::quantize)
        .def("dequantize", &Quantizer::dequantize);
}
