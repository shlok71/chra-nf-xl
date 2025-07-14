#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "nca_canvas.h"

namespace py = pybind11;

PYBIND11_MODULE(nca_canvas_py, m) {
    py::class_<NCACanvas>(m, "NCACanvas")
        .def(py::init<int, int>())
        .def("edit_pixel", &NCACanvas::edit_pixel)
        .def("draw_circle", &NCACanvas::draw_circle)
        .def("step", &NCACanvas::step)
        .def("to_bhv", &NCACanvas::to_bhv)
        .def("generate_image", &NCACanvas::generate_image)
        .def("generate_video", &NCACanvas::generate_video);
}
