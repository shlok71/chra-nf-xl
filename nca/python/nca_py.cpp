#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/nca.h"

namespace py = pybind11;

PYBIND11_MODULE(nca_py, m) {
    py::class_<NCA>(m, "NCA")
        .def(py::init<int, int>())
        .def("step", &NCA::step)
        .def("load_weights", &NCA::load_weights)
        .def("get_grid", [](NCA &nca) {
            const auto& grid = nca.getGrid();
            const auto& shape = {nca.height, nca.width, 4};
            return py::array_t<uint8_t>(shape, grid.data());
        })
        .def("set_pixel", &NCA::set_pixel)
        .def("draw_circle", &NCA::draw_circle)
        .def("undo", &NCA::undo)
        .def("redo", &NCA::redo)
        .def("settle", &NCA::settle)
        .def("transcribe_to_bhv", &NCA::transcribe_to_bhv);
}
