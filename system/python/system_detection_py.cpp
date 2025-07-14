#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "system_detection.h"

namespace py = pybind11;

PYBIND11_MODULE(system_detection_py, m) {
    py::class_<SystemSpecs>(m, "SystemSpecs")
        .def(py::init<>())
        .def_readwrite("ram_gb", &SystemSpecs::ram_gb)
        .def_readwrite("cpu_cores", &SystemSpecs::cpu_cores);

    py::class_<SystemDetection>(m, "SystemDetection")
        .def(py::init<>())
        .def("get_system_specs", &SystemDetection::get_system_specs)
        .def("scale_architecture", &SystemDetection::scale_architecture);
}
