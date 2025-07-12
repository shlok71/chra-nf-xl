#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "avx2_bhv.h"

namespace py = pybind11;

PYBIND11_MODULE(bhv_py, m) {
    m.doc() = "Python bindings for the CHRA-NF-XL BHV library";

    py::class_<BHV>(m, "BHV")
        .def(py::init<>())
        .def("encode_text", &BHV::encode_text,
             py::arg("tokens"),
             "Encodes a list of string tokens into a BHV.")
        .def("bind", &BHV::bind,
             py::arg("a"), py::arg("b"),
             "Binds two BHVs together (XOR operation).")
        .def("hamming_distance", &BHV::hamming_distance,
             py::arg("a"), py::arg("b"),
             "Computes the Hamming distance between two BHVs.")
        // Allow property-like access to the raw data if needed, but make it read-only
        .def_property_readonly("data", [](const BHV& self) {
            return std::vector<uint64_t>(self.data, self.data + BHV_WORDS);
        });
}
