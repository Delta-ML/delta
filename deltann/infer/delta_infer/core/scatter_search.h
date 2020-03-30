#include "delta_infer/core/algorithm.h" 
#include "delta_infer/core/pattern.h" 
#include "delta_infer/core/debug.h" 

namespace tensorflow {
namespace grappler {

class ScatterSearch : public Algorithm {
public:
    ScatterSearch(const std::string& name):Algorithm(name) {}
    ~ScatterSearch() {}
    
    void set_ori_graph(const Pattern* graph) {
        _graph = *graph;
    }

    void set_pattern(const std::string& name, const Pattern* pattern) {
        _name = name;
        _pattern = *pattern;
    }

    Pattern& get_graph() { return _graph; }
    Pattern& get_pattern() { return _pattern; }

    void replace_outs(std::string replace_pattern_name, 
                      std::unordered_map<std::string, std::vector<Pattern::Node> >& outs, 
                      Pattern& graph) {
        for(auto it = outs.begin(); it!=outs.end();++it) {
            for(auto& node : it->second) {
                // get node from graph
                auto* ori_node_def = graph.mutable_node(node.idx());
                auto* inputs = ori_node_def->mutable_input();
                for(int i=0; i<inputs->size(); i++) {
                    if((*inputs)[i] == it->first) {
                        (*inputs)[i] = replace_pattern_name;
                    }
                }
            } 
        }
    }

    void set_attrs(NodeDef* target, Pattern::Node& node) {
        //auto* attr = target->mutable_attr();
        //(*attr)["T"].set_type(DT_FLOAT);
        //(*target->mutable_attr())["T"].set_type(DT_FLOAT);
        if(node.op() != std::string("Const")) {
            DELTA_PRINT("Detect input : %s(%s)\n", node.name().c_str(), node.op().c_str());
            target->add_input(node.name());
            //(*target->mutable_attr())["T"] = node.degenerate()->attr().at("T");
        } else {
            DELTA_PRINT("Detect attr : %s(%s)\n", node.name().c_str(), node.op().c_str());
            (*target->mutable_attr())[node.name()] = node.degenerate()->attr().begin()->second;
            // invalid the attr node which existed in graph
            _invalidated_nodes.emplace(node.degenerate());
        }
    }

    void invalid(absl::flat_hash_set<const NodeDef*>& invalid_nodes) {
        for(auto* node : invalid_nodes) {
            _invalidated_nodes.emplace(node);
        }
    }

    bool valid(const Pattern::Node& node) {
        if(_invalidated_nodes.contains(node.degenerate())) {
            return false;
        }
        return true;
    }

    void invalid_clear() { _invalidated_nodes.clear(); }

    virtual GraphDef* GenOptGraph() override;

private:
    Pattern _graph;
    Pattern _pattern; 
    std::string _name;
    absl::flat_hash_set<const NodeDef*> _invalidated_nodes;
    int accumulator{0};
};

} /* namespace grappler */
} /* namespace tensorflow */

