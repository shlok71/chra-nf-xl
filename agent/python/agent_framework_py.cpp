#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "agent_framework.h"

namespace py = pybind11;

class PySubAgent : public SubAgent {
public:
    using SubAgent::SubAgent;
    void run() override {
        PYBIND11_OVERRIDE_PURE(void, SubAgent, run,);
    }
};

PYBIND11_MODULE(agent_framework_py, m) {
    py::class_<SubAgent, PySubAgent>(m, "SubAgent")
        .def(py::init<const std::string&>())
        .def("run", &SubAgent::run)
        .def("get_state", &SubAgent::get_state);

    py::class_<Agent>(m, "Agent")
        .def(py::init<const std::string&>())
        .def("add_sub_agent", &Agent::add_sub_agent)
        .def("run_serial", &Agent::run_serial)
        .def("run_parallel", &Agent::run_parallel);

    py::class_<AgentFramework>(m, "AgentFramework")
        .def(py::init<>())
        .def("add_agent", &Agent::add_agent)
        .def("run", &Agent::run);
}
