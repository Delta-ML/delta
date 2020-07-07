#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <climits>
#include <limits>
#include <unordered_map>

#include "delta_infer/cpp/export_utils.h"

namespace delta_infer {
namespace py = pybind11;
using namespace pybind11::literals;
using namespace tensorflow;

PYBIND11_MODULE(export_py, module) {
  module.doc() = "Python Interface for Delta Inference";
  // Submodule 'core'
  auto core = module.def_submodule("core", "Submodule of export_py");

  py::class_<grappler::Pattern>(core, "Pattern")
      .def(py::init<>())
      .def(py::init<const GraphDef*>())
      .def("LoadModel",
           delta_overload_cast<const char*>()(&grappler::Pattern::LoadModel),
           "Load tf model from pb files.")
      .def("LoadModelCT",
           delta_overload_cast<const std::string&>()(
               &grappler::Pattern::LoadModelCT),
           "Load tf model from GraphDef string buffer.")
      .def("get_hint_node", &grappler::Pattern::get_hint_node)
      .def("set_hint_node_type", &grappler::Pattern::set_hint_node_type)
      .def("nodes", &grappler::Pattern::nodes)
      .def("neighbors", &grappler::Pattern::neighbors)
      .def("pattern_ins", &grappler::Pattern::pattern_ins)
      .def("pattern_outs", &grappler::Pattern::pattern_outs)
      .def("graph", &grappler::Pattern::graph)
      .def("node_size", &grappler::Pattern::node_size)
      .def("mutable_node", &grappler::Pattern::mutable_node)
      .def("node",
           delta_overload_cast<int>()(&grappler::Pattern::node, py::const_),
           "Get Node by index.")
      .def("node",
           delta_overload_cast<const std::string&>()(&grappler::Pattern::node,
                                                     py::const_),
           "Get Node by name.")
      .def_property_readonly("is_input", &grappler::Pattern::is_input);
  //.def(py::self == py::self)
  //.def(py::self = py::self);

  py::class_<grappler::Pattern::Node>(core, "Node")
      .def(py::init(&grappler::Pattern::Node::null))
      .def("empty", &grappler::Pattern::Node::empty)
      .def("name", &grappler::Pattern::Node::name)
      .def("op", &grappler::Pattern::Node::op)
      .def("idx", &grappler::Pattern::Node::idx)
      .def("ins", &grappler::Pattern::Node::ins)
      .def("outs", &grappler::Pattern::Node::outs)
      .def("neighbors", &grappler::Pattern::Node::neighbors)
      .def("is_input", &grappler::Pattern::Node::is_input)
      .def("degenerate", &grappler::Pattern::Node::degenerate);
  //.def(py::self == py::self);

  // Submodule 'optimizer'
  auto optimizer = module.def_submodule("optimizer", "Submodule of export_py");

  // export base class LocalOptimizer
  py::class_<grappler::LocalOptimizer, PyLocalOptimizer> RealPyLocalOptimizer(
      optimizer, "LocalOptimizer");
  RealPyLocalOptimizer.def(py::init<const std::string&>())
      .def_property_readonly("name", &grappler::LocalOptimizer::GetName)
      .def("GenOptGraph", &grappler::LocalOptimizer::GenOptGraph)
      .def("_repr__", [](const grappler::LocalOptimizer& optim) {
        return "<[Pure Abstract c++ class]export_py.LocalOptimizer named '" +
               optim.GetName() + "'>";
      });

  // export pattern register
  optimizer.def("RegisterFusionPattern", &grappler::RegisterFusionPattern);

  py::class_<AutoOptimizer>(optimizer, "LocalAutoOptimizer")
      .def(py::init<const Pattern&>())
      .def("run", &AutoOptimizer::run)
      .def("serialization", &AutoOptimizer::serialization);

#if 0
    // export optimizer AutoFusion
    py::class_<grappler::AutoFusion>(optimizer, "AutoFusion", RealPyLocalOptimizer)
        .def(py::init<const std::string&>());

    // export pattern register
    optimizer.def("RegisterFusionPattern", &grappler::RegisterFusionPattern);

    py::class_<grappler::LocalOptimizerManager, 
               std::unique_ptr<grappler::LocalOptimizerManager, py::nodelete> >(optimizer, "OptimizerManager")
        .def_static("Instance", &grappler::LocalOptimizerManager::Instance)
        .def("Set", &grappler::LocalOptimizerManager::Set)
        .def("contain", &grappler::LocalOptimizerManager::contain)
        .def("Get", &grappler::LocalOptimizerManager::Get)
        .def("names", &grappler::LocalOptimizerManager::names);
#endif
}

}  // namespace delta_infer
