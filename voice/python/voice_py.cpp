#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "whisper_wrapper.h"
#include "tts_wrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(voice_py, m) {
    py::class_<WhisperWrapper>(m, "WhisperWrapper")
        .def(py::init<>())
        .def("transcribe", &WhisperWrapper::transcribe);

    py::class_<TTSWrapper>(m, "TTSWrapper")
        .def(py::init<>())
        .def("speak", &TTSWrapper::speak);
}
