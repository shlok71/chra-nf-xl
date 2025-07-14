#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "scaling_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(scaling_engine_py, m) {
    py::enum_<ParamConfig>(m, "ParamConfig")
        .value("LOW", ParamConfig::LOW)
        .value("MEDIUM", ParamCofig::MEDIUM)
        .value("HIGH", ParamConfig::HIGH)
        .value("ULTRA", ParamConfig::ULTRA)
        .export_values();

    py::class_<ScalingEngine>(m, "ScalingEngine")
        .def(py::init<>())
        .def("get_param_config", &ScalingEngine::get_param_config)
        .def("set_param_config", &ScalingEngine::set_param_config);
}
