#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "paging_system.h"

namespace py = pybind11;

PYBIND11_MODULE(paging_system_py, m) {
    py::class_<PagingSystem>(m, "PagingSystem")
        .def(py::init<>())
        .def("page_out", &PagingSystem::page_out)
        .def("page_in", &PagingSystem::page_in);
}
