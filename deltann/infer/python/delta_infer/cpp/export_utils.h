#ifndef _DELTA_INFER_EXPORT_UTILS_H_
#define _DELTA_INFER_EXPORT_UTILS_H_

#include <fstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "delta_infer/core/pattern.h"
#include "delta_infer/custom_grappler/auto_fusion.h"
#include "delta_infer/custom_grappler/local_optimizer.h"

namespace delta_infer {
namespace py = pybind11;

using namespace pybind11::literals;
using namespace tensorflow;
using namespace tensorflow::grappler;

template <typename... Args>
using delta_overload_cast = pybind11::detail::overload_cast_impl<Args...>;

class PyLocalOptimizer: public LocalOptimizer {
public:
    using LocalOptimizer::LocalOptimizer;

    virtual GraphDef* GenOptGraph(const GraphDef* original_graph) {
        PYBIND11_OVERLOAD_PURE(
                GraphDef*,
                LocalOptimizer,
                GenOptGraph,
                original_graph
        );
    }
};

/**
 * @brief AutoOptimizer is a class for graph optimization automatically
 *
 */
class AutoOptimizer {
public:
    AutoOptimizer(const Pattern& pattern):_pattern(pattern) {}
    ~AutoOptimizer(){}

    /// run automatical optimization
    void run() {
        for(auto local_optimi_name : LocalOptimizerManager::Instance().names()) {
            auto* local_optimizer = LocalOptimizerManager::Instance().Get(local_optimi_name)();
            // condition check
            if(auto* auto_fusion_optimizer = dynamic_cast<AutoFusion*>(local_optimizer)) {
                optimized_graph_def = auto_fusion_optimizer->GenOptGraph(_pattern.graph()); 
                /// TODO
                //....
            } else {
                //// error
                printf("dynamic cast error on local optimizer %s !\n", local_optimi_name.c_str());
                exit(1);
            }
            /// change _pattern 's graph content to optimized_graph_def
            _pattern.LoadModel(*optimized_graph_def);
        }
    }

    void serialization(const char* path) {
        std::string out_graph;
        std::ofstream fout;
        _pattern.graph()->SerializeToString(&out_graph);
        fout.open (path, std::ios::binary | std::ios::out);
        fout<<out_graph;
        fout.close();
    }

private:
    Pattern _pattern;
    GraphDef* optimized_graph_def{nullptr};
};

} /* namespace delta_infer */

#endif
