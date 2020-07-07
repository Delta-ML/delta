#ifndef _DELTA_INFER_PATTERN_H_
#define _DELTA_INFER_PATTERN_H_

#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/graph/tensor_id.h"

namespace tensorflow {
namespace grappler {

/**
 * @brief graph pattern for auto search algorithm.
 */
class Pattern {
 public:
  typedef GraphTopologyView GraphViewType;
  typedef absl::InlinedVector<int, 4> InsIdxType;
  typedef absl::InlinedVector<int, 2> OutsIdxType;

 public:
  struct Node {
    Node(const NodeDef* node, const Pattern* pattern)
        : node_p(node), pattern_p(pattern) {
      if (node != nullptr) {
        node_idx = pattern->_inner_graph_ptr->GetNodeIndex(*node);
      }
    }

    Node(const NodeDef& node, const Pattern* pattern)
        : node_p(&node), pattern_p(pattern) {
      node_idx = pattern->_inner_graph_ptr->GetNodeIndex(node);
    }

    Node(const Node& node)
        : node_p(node.node_p),
          pattern_p(node.pattern_p),
          node_idx(node.node_idx) {}

    ~Node() {}

    bool empty() { return node_p == nullptr; }

    static Node null(const Pattern* pattern) { return Node(nullptr, pattern); }

    std::string name() const { return node_p->name().c_str(); }

    std::string op() const { return node_p->op().c_str(); }

    int idx() const { return node_idx.value(); }

    InsIdxType ins() const;

    OutsIdxType outs() const;

    std::vector<Node> neighbors() const;

    bool is_input() const;

    const NodeDef* degenerate() const { return node_p; }

    bool operator==(const Node& node) const {
      return node_p == node.node_p && pattern_p == node.pattern_p;
    }

    template <typename H>
    friend H AbslHashValue(H h, const Node& p) {
      return H::combine(std::move(h), p.name());
    }

   private:
    absl::optional<int> node_idx{-1};
    const NodeDef* node_p{nullptr};
    const Pattern* pattern_p{nullptr};
  };

 public:
  Pattern() {}
  explicit Pattern(const GraphDef* graph_p) {
    // set graph pattern with self graphdef member to prevent mem leak.
    _graph_def_ptr = std::make_shared<GraphDef>(*graph_p);
    _inner_graph_ptr = std::make_shared<GraphViewType>();
    _inner_graph_ptr->InitializeFromGraph(*_graph_def_ptr.get());
    _inputs = pattern_ins();
  }
  Pattern(const Pattern& pattern) {
    _graph_def_ptr = pattern._graph_def_ptr;
    _inner_graph_ptr = pattern._inner_graph_ptr;
    _hint_op_type = pattern._hint_op_type;
    _inputs = pattern_ins();
  }
  ~Pattern() {}

  /// load model string content to construction a new pattern
  void LoadModelCT(const std::string& model_content);

  /// load pb model to construction a new pattern
  void LoadModel(const GraphDef& model_pb);

  /// load pb model file to construction a new pattern
  void LoadModel(const char* model_pb_path);

  bool is_input(const Node& node) { return node.is_input(); }

  /// get hint body node from pattern.
  /// note: this node isn't 'Placeholder'
  Node get_hint_node();
  bool set_hint_node_type(const std::string& op_type);

  std::vector<Node> nodes();

  std::vector<Node> neighbors(const Node& node) { return node.neighbors(); }

  /// detection and get pattern inputs
  absl::flat_hash_set<Node> pattern_ins();

  /// detection and get pattern outputs
  absl::flat_hash_set<Node> pattern_outs();

  inline const GraphDef* graph() const { return _inner_graph_ptr->graph(); }

  inline size_t node_size() { return _inner_graph_ptr->num_nodes(); }

  inline NodeDef* mutable_node(int node_idx) {
    return _graph_def_ptr->mutable_node(node_idx);
  }

  inline Node node(int node_idx) const {
    return Node(_inner_graph_ptr->GetNode(node_idx), this);
  }
  inline Node node(const std::string& name) const {
    return Node(_inner_graph_ptr->GetNode(absl::string_view(name.c_str())),
                this);
  }

  Pattern& operator==(const Pattern& pattern) {
    this->_inner_graph_ptr = pattern._inner_graph_ptr;
    return *this;
  }

  Pattern& operator=(const Pattern& pattern) {
    this->_graph_def_ptr = pattern._graph_def_ptr;
    this->_inner_graph_ptr = pattern._inner_graph_ptr;
    this->_hint_op_type = pattern._hint_op_type;
    this->_inputs = pattern_ins();
    return *this;
  }

 private:
  // add extra graph
  std::shared_ptr<GraphDef> _graph_def_ptr;
  std::shared_ptr<GraphViewType> _inner_graph_ptr{nullptr};
  // graph inputs
  absl::flat_hash_set<Node> _inputs;
  std::string _hint_op_type;
};

Pattern CreatePattern(const char* model_pb_path);

} /* namespace grappler */
} /* namespace tensorflow */

#endif
