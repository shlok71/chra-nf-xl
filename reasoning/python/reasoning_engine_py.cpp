#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "reasoning_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(reasoning_engine_py, m) {
    py::class_<ReasoningEngine>(m, "ReasoningEngine")
        .def(py::init<>())
        .def("reason", &ReasoningEngine::reason);
}
