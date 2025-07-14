#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "overlay.h"

namespace py = pybind11;

PYBIND11_MODULE(overlay_py, m) {
    py::class_<Overlay>(m, "Overlay")
        .def(py::init<>())
        .def("show_window", &Overlay::show_window)
        .def("show_notification", &Overlay::show_notification)
        .def("get_input", &Overlay::get_input)
        .def("run_plugin", &Overlay::run_plugin);
}
