#include "delta_infer/custom_grappler/auto_fusion.h"

namespace tensorflow {
namespace grappler {

struct FusionPatterns {
    static FusionPatterns& Instance() {
        static FusionPatterns _ins;
        return _ins;
    }
    
    void Set(const std::string& name, const Pattern& algo) {
        if(!contain(name)) {
            _pattern_map[name] = algo;
        }
    }

    void SetHintOp(const std::string& name, const std::string& op) {
        if(contain(name)) {
            _pattern_map[name].set_hint_node_type(op);
        }
    }

    bool contain(const std::string& name) {
        for(auto it = _pattern_map.begin(); it!=_pattern_map.end(); ++it) {
            if(it->first == name) {
                return true;
            }
        }
        return false;
    }

    Pattern* Get(const std::string& name) {
        if(!contain(name)) {
            return nullptr;
        }
        return &_pattern_map[name];
    }

    std::vector<std::string> names() {
        std::vector<std::string> ret;
        for(auto it = _pattern_map.begin(); it!=_pattern_map.end(); ++it) {
            ret.push_back(it->first);
        }
        return ret;
    }

private:
    FusionPatterns() {}

    std::unordered_map<std::string, Pattern> _pattern_map;
};

void RegisterFusionPattern(const std::string& name, const Pattern& graph, const std::string& hint_op="NULL") {
    FusionPatterns::Instance().Set(name, graph);
    if(hint_op != "NULL") {
        FusionPatterns::Instance().SetHintOp(name, hint_op);
    }
}

GraphDef* AutoFusion::GenOptGraph(const GraphDef* original_graph) {
    /// ( auto fusion algorithm )
    // get all patterns register by user
    GraphDef* temp_graph = new GraphDef();
    temp_graph->CopyFrom(*original_graph);
    auto pattern_vec = FusionPatterns::Instance().names();
    for(auto& pattern_name : pattern_vec) {
        auto* pattern = FusionPatterns::Instance().Get(pattern_name);
        if(pattern) {
            /// TODO
            // get algorithm by name
            auto* algo = AlgoManager::Instance().Get("ScatterSearch")();
            ScatterSearch* algo_scatter_search = dynamic_cast<ScatterSearch*>(algo);
            // set graph and pattern
            Pattern ori_graph_pattern(temp_graph);
            algo_scatter_search->set_ori_graph(&ori_graph_pattern);
            algo_scatter_search->set_pattern(pattern_name, pattern);
            // run
            delete temp_graph; // because temp_graph is copy in set_ori_graph
            temp_graph = algo_scatter_search->GenOptGraph();
        }
    }
    return temp_graph;
}

REGISTER_LOCAL_OPTIMIZER(AutoFusion);

} /* namespace grappler */
} /* namespace tensorflow */

