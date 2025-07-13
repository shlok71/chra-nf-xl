#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "associative_memory.h"

namespace py = pybind11;

PYBIND11_MODULE(associative_memory_py, m) {
    py::class_<AssociativeMemory>(m, "AssociativeMemory")
        .def(py::init<>())
        .def("insert", &AssociativeMemory::insert)
        .def("query", &AssociativeMemory::query)
        .def("update", &AssociativeMemory::update);
}
