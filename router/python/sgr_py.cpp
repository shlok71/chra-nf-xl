#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sgr.h"

namespace py = pybind11;

PYBIND11_MODULE(sgr_py, m) {
    py::enum_<TaskType>(m, "TaskType")
        .value("TEXT", TaskType::TEXT)
        .value("OCR", TaskType::OCR)
        .value("CANVAS", TaskType::CANVAS)
        .value("RETRIEVAL", TaskType::RETRIEVAL)
        .export_values();

    py::class_<SGR>(m, "SGR")
        .def(py::init<>())
        .def("route", &SGR::route)
        .def("register_module", [](SGR &sgr, TaskType task_type, py::function load_function) {
            sgr.register_module(task_type, [load_function]() {
                load_function();
            });
        });
}
