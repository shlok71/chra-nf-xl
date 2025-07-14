#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "profiler.h"

namespace py = pybind11;

PYBIND11_MODULE(profiler_py, m) {
    py::class_<Profiler>(m, "Profiler")
        .def(py::init<>())
        .def("start_profiling", &Profiler::start_profiling)
        .def("stop_profiling", &Profiler::stop_profiling);
}
