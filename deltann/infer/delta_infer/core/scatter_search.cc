#include "delta_infer/core/algorithm.h" 
#include "delta_infer/core/pattern.h" 
#include "delta_infer/core/scatter_search.h" 

#include <queue>

namespace tensorflow {
namespace grappler {

struct Compare {
    Compare() {
        std::vector<absl::flat_hash_set<std::string>> op_maps = {
            {"BatchMatMulV2", "BatchMatMul",},
            {"Const", "Shape"},
        };
        _equal_op_maps = op_maps;
    }

    bool operator()(const std::string& lhs, const std::string& rhs) {
        if(lhs == rhs) {
            return true;
        }
        for(auto& set : _equal_op_maps) {
            if(set.contains(lhs) && set.contains(rhs)) {
                return true;
            }
        }
        return false;
    }

private:    
    std::vector<absl::flat_hash_set<std::string>> _equal_op_maps;
};

class Scatter {
public:
    struct NodePair {
        Pattern::Node graph_node;
        Pattern::Node pattern_node;
    };
    explicit Scatter(const Pattern::Node& graph_node, const Pattern::Node& hint_node)
        :_graph_node(graph_node),_hint_node(hint_node) {}
    ~Scatter() {}

    void set_op_type_map(std::vector<Pattern::Node>& neighbors, 
                         std::unordered_map<string, int>& neighbors_ops_cnt) {
        for(int i=0; i<neighbors.size(); i++) {
            if(neighbors_ops_cnt.count(neighbors[i].op())) {
                neighbors_ops_cnt[neighbors[i].op()]++;
            } else {
                neighbors_ops_cnt[neighbors[i].op()] = 0;
            }
        }
    }

    std::vector<Pattern::Node>& inputs() { return _inputs; }
    std::unordered_map<std::string, std::vector<Pattern::Node> >& outs() { return _outs_map; }
    absl::flat_hash_set<const NodeDef*>& trace_nodes() { return _trace_nodes; }

    enum class RetCheck {
        OK=0,
        ERR,
        CONTINUE,
    };

    //  check if equal
    //  return: OK      : equal and continue
    //          ERR     : not equal and return 
    //          CONTINUE: not equal but continue
    RetCheck check_equal(NodePair& current_inspect_node_pair,
                         std::vector<Pattern::Node>& neighbors_lhs, 
                         std::vector<Pattern::Node>& neighbors_rhs) {
        if(neighbors_lhs.size() != neighbors_rhs.size()) {
            if(neighbors_lhs.size() < neighbors_rhs.size()) {
                // remove  all strided_slice before and after neighbors_rhs
                bool need_step_over = false;
                int j = 0;
                while(neighbors_rhs[j].op() == std::string("StridedSlice")) {
                    j++;
                }
                if(j>0) {
                    need_step_over = true;
                    for(auto it = neighbors_rhs.begin(); it != neighbors_rhs.end();) {
                        if(it->op() == std::string("StridedSlice")) {
                            _trace_nodes.emplace(it->degenerate());
                            it = neighbors_rhs.erase(it);
                        } else {
                            ++it;
                        }
                    }
                }
                /*for(auto it = neighbors_rhs.begin(); it != neighbors_rhs.end();) {
                    if(it->op() == std::string("StridedSlice")) {
                        // must put this StridedSlice node to _trace_node for invalid it lately
                        _trace_nodes.emplace(it->degenerate());
                        it = neighbors_lhs.erase(it);
                        break;
                    } else {
                        ++it;
                    }
                }*/
                // remove strided_slice after neighbors_rhs
                int i=0;
                for(; i < neighbors_lhs.size(); i++) {
                    if(!_cmp(neighbors_lhs[i].op(), neighbors_rhs[i].op())) {
                        return RetCheck::ERR;
                    }
                }
                // delete extra strided_slice
                int cnt =0;
                //bool need_step_over = false;
                for(auto it = neighbors_rhs.begin(); it != neighbors_rhs.end();) {
                    if(cnt < i) {
                        ++cnt;
                        ++it;
                    } else {
                        if(it->op() == std::string("StridedSlice")) {
                            // must put this shape node to _trace_node for invalid it lately
                            _trace_nodes.emplace(it->degenerate());
                            it = neighbors_rhs.erase(it);
                            need_step_over = true;
                        } else {
                            ++it;
                        }
                    }
                }
                if(need_step_over) {
                    return RetCheck::OK;
                }
                return RetCheck::ERR;
            }
            if(neighbors_lhs.size() > neighbors_rhs.size()) {
                int i=0, j=0;
                bool need_step_over = false;
                for(; j < neighbors_rhs.size();i++, j++) {
                    if(neighbors_lhs[i].op() == std::string("Shape") && !_cmp(neighbors_lhs[i].op(), neighbors_rhs[j].op())) {
                        i++; // step over shape
                        need_step_over = true;
                    }
                    if(!_cmp(neighbors_lhs[i].op(), neighbors_rhs[j].op()) && !neighbors_rhs[j].is_input()) {
                        return RetCheck::ERR;
                    }
                }
                for(;i < neighbors_lhs.size(); i++) {
                    _outs_map[current_inspect_node_pair.graph_node.name()].push_back(neighbors_lhs[i]);
                }
                if(need_step_over) {
                    // remove shape op in neighbors_lhs, to keep same size with neighbors_rhs
                    for(auto it = neighbors_lhs.begin(); it != neighbors_lhs.end();) {
                        if(it->op() == std::string("Shape")) {
                            // must put this shape node to _trace_node for invalid it lately
                            _trace_nodes.emplace(it->degenerate());
                            it = neighbors_lhs.erase(it);
                        } else {
                            ++it;
                        }
                    }
                    // if need step over, we should return ok for removing those op
                    return RetCheck::OK;
                }
                return RetCheck::CONTINUE;
            } else {
                DELTA_PRINT("        [ERROR size] %d *VS* %d \n", neighbors_lhs.size(), neighbors_rhs.size());
                return RetCheck::ERR;
            }
        }
        for(int i=0; i<neighbors_lhs.size(); i++) {
            if(neighbors_rhs[i].is_input() && (neighbors_lhs[i].op() != std::string("Shape"))) {
                // get patttern's input point
                _inputs.push_back(neighbors_lhs[i]);
            }
            if(!_cmp(neighbors_lhs[i].op(), neighbors_rhs[i].op()) && !neighbors_rhs[i].is_input()) {
                return RetCheck::ERR;
            }
        }
        return RetCheck::OK;
    }

    void push_queue(std::vector<Pattern::Node>& neighbors_lhs,
                    std::vector<Pattern::Node>& neighbors_rhs) {
        for(int i=0; i<neighbors_lhs.size(); i++) {
            if(!neighbors_rhs[i].is_input()) {
                _scatter_que.push({neighbors_lhs[i], neighbors_rhs[i]});
            }
        } 
    }

    void mask(std::vector<Pattern::Node>& neighbors_lhs,
              std::vector<Pattern::Node>& neighbors_rhs) {
        auto it = neighbors_rhs.begin();
        auto it_other = neighbors_lhs.begin();
        for(;it != neighbors_rhs.end();) {
            if(_backups.contains(*it)) {
                it_other = neighbors_lhs.erase(it_other);
                it = neighbors_rhs.erase(it);
            } else {
                ++it_other;
                ++it;
            }
        }
    }

    void back_up(std::vector<Pattern::Node>& neighbors) {
        for(auto& node : neighbors) {
            _backups.emplace(node);
        }
    }

    bool search() {
        std::vector<Pattern::Node> graph_neighbors;
        std::vector<Pattern::Node> hint_neighbors;
        graph_neighbors.push_back(_graph_node);
        hint_neighbors.push_back(_hint_node);
        back_up(graph_neighbors);
        back_up(hint_neighbors);
        push_queue(graph_neighbors, hint_neighbors);
        while(!_scatter_que.empty()) {
            auto& node_pair = _scatter_que.front();
            _trace_nodes.emplace(node_pair.graph_node.degenerate());
            DELTA_PRINT("   ==> get node %s <<==>> %s\n", node_pair.graph_node.name().c_str(), node_pair.pattern_node.name().c_str());
            auto tmp_graph_neighbors =  node_pair.graph_node.neighbors(); 
            auto tmp_hint_neighbors = node_pair.pattern_node.neighbors(); 
#ifdef BUILD_DEBUG
            auto info_neighbors = [&](const char* perfix, std::vector<Pattern::Node>& neighbors) {
                for(int i=0; i<neighbors.size(); i++) {
                    DELTA_PRINT("     %s get neghbors: %s(%s)\n", perfix, neighbors[i].name().c_str(), neighbors[i].op().c_str());
                }
            };
            info_neighbors(" ---> Graph ", tmp_graph_neighbors);
            info_neighbors(" ---> Pattern ", tmp_hint_neighbors);
#endif
            
            auto ret = check_equal(node_pair, tmp_graph_neighbors, tmp_hint_neighbors);
            if(ret == RetCheck::ERR) { 
                return false;
            } else if(ret == RetCheck::OK) {
                mask(tmp_graph_neighbors, tmp_hint_neighbors); 
                push_queue(tmp_graph_neighbors, tmp_hint_neighbors);
                back_up(tmp_graph_neighbors);
                back_up(tmp_hint_neighbors);
            } else {  // RetCheck::CONTINUE
                /// does nothing and continue to loop
            }
            _scatter_que.pop();
        }
        return true;
    }

private:
    std::queue<NodePair> _scatter_que;
    absl::flat_hash_set<Pattern::Node> _backups;

    Pattern::Node _graph_node;
    Pattern::Node _hint_node;
    Compare _cmp;
    // io of target graph when replacing pattern graph
    std::vector<Pattern::Node> _inputs;
    std::unordered_map<std::string, std::vector<Pattern::Node> >  _outs_map;
    // trace node for scatter algo
    absl::flat_hash_set<const NodeDef*> _trace_nodes;
};

GraphDef* ScatterSearch::GenOptGraph() { 
    Pattern::Node hint = _pattern.get_hint_node();
    Compare cmp;
    if(hint.empty()) {
        ///return optimized_graph with full graph;
    }

    //GraphDef* optimized_graph = new GraphDef();
    //optimized_graph->mutable_node()->Reserve(_graph.node_size());

    auto iterator_for_pattern_match = [&] (const GraphDef* graph_def, bool& find_subgraph) -> GraphDef* {
        invalid_clear();
        GraphDef* optimized_graph = new GraphDef();
        optimized_graph->mutable_node()->Reserve(graph_def->node_size());

        Pattern graph(graph_def);
        for(auto& node : graph.nodes()) {
            if(!valid(node)) {
                continue;
            }
            if(cmp(node.op(), hint.op())) {
                /// (scatter search and replace)
                Scatter scatter_search_engine(node, hint);
                /// find subgraph at graph node
                /// enable log scope
                DELTA_LOG_SCOPE_START();
                printf(">>>  start search graph node: %s <==> %s \n", node.name().c_str(), hint.name().c_str());
                //DELTA_LOG_SCOPE_END();
                if(scatter_search_engine.search()) {
                    // invalid the trace nodes
                    this->invalid(scatter_search_engine.trace_nodes());
    
                    DELTA_LOG_SCOPE_START();
                    DELTA_PRINT("@@@@@@@Delta find a subgraph at %s(%s)\n", node.name().c_str(), node.op().c_str());
                    /// replace pattern
                    NodeDef* replace_pattern = optimized_graph->add_node();
                    std::string replace_pattern_name = _name + "" + std::to_string(accumulator++);
                    replace_pattern->set_name(replace_pattern_name);
                    replace_pattern->set_op(_name);
    
                    // replace pattern inputs and outputs
                    absl::flat_hash_set<Pattern::Node> input_backups;
                    auto& inputs = scatter_search_engine.inputs();
                    // set const attr and inputs for replace_pattern
                    /*for(auto& node : inputs) {
                        if(!input_backups.contains(node)) {
                            set_attrs(replace_pattern, node);
                            input_backups.emplace(node);
                        }
                    }*/
                    for(auto& node : inputs) {
                        if(!input_backups.contains(node)) {
                            DELTA_PRINT("Detect input : %s(%s)\n", node.name().c_str(), node.op().c_str());
                            replace_pattern->add_input(node.name());
                            input_backups.emplace(node);
                        }
                    }
                    auto& outs_map = scatter_search_engine.outs();
                    this->replace_outs(replace_pattern_name, outs_map, graph);
#ifdef BUILD_DEBUG 
                    for(auto it = outs_map.begin(); it!=outs_map.end(); ++it) {
                        for(auto& node : it->second) {
                            DELTA_PRINT("Detect output: %s(%s)  from %s\n", node.name().c_str(), node.op().c_str(),it->first.c_str());
                        }
                    }
#endif
                    // fill T attributes from graph hint
                    (*replace_pattern->mutable_attr())["T"] = hint.degenerate()->attr().at("T");
                        
                    // close log scope
                    DELTA_LOG_SCOPE_END();
                    find_subgraph = true;
                    break;
                }
            }
        }
        // step over all invalid node
        for(auto& node : graph.nodes()) {
            if(!valid(node)) { continue; }
            optimized_graph->add_node()->CopyFrom(*(node.degenerate()));
        }
        optimized_graph->mutable_library()->CopyFrom(graph.graph()->library());
        optimized_graph->mutable_versions()->CopyFrom(graph.graph()->versions());
        return optimized_graph;
    };

    // loop iterator for graph
    bool find_subgraph = false;
    GraphDef* graph_temp = iterator_for_pattern_match(_graph.graph(), find_subgraph);
    while(find_subgraph) {
        find_subgraph = false;
        GraphDef* last_graph_ptr = graph_temp;
        auto temp = iterator_for_pattern_match(graph_temp, find_subgraph);
        if(find_subgraph) {
            graph_temp = temp;
            delete last_graph_ptr;
        }
    }
    return graph_temp;
}

REGISTER_ALGO(ScatterSearch);

} /* namespace grappler */
} /* namespace tensorflow */

