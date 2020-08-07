#include "delta_infer/core/pattern.h"

namespace tensorflow {
namespace grappler {

Pattern::InsIdxType Pattern::Node::ins() const {
  return pattern_p->_inner_graph_ptr->GetFanin(this->idx());
}

Pattern::OutsIdxType Pattern::Node::outs() const {
  return pattern_p->_inner_graph_ptr->GetFanout(this->idx());
}

std::vector<Pattern::Node> Pattern::Node::neighbors() const {
  std::vector<Pattern::Node> results;
  const auto inputs = this->ins();
  for (auto& in : inputs) {
    results.push_back(pattern_p->node(in));
  }
  const auto outs = this->outs();
  for (auto& out : outs) {
    results.push_back(pattern_p->node(out));
  }
  return results;
}

bool Pattern::Node::is_input() const {
  if (this->op() == std::string("Placeholder") ||
      this->op() == std::string("Const")) {
    return true;
  } else {
    return false;
  }
}

void Pattern::LoadModelCT(const std::string& model_content) {
  _graph_def_ptr = std::make_shared<GraphDef>();
  _graph_def_ptr->ParseFromString(model_content);
  _inner_graph_ptr.reset(new GraphViewType());
  _inner_graph_ptr->InitializeFromGraph(*(_graph_def_ptr.get()));
  _inputs = pattern_ins();
}

void Pattern::LoadModel(const GraphDef& model_pb) {
  _graph_def_ptr = std::make_shared<GraphDef>();
  _graph_def_ptr->CopyFrom(model_pb);
  _inner_graph_ptr.reset(new GraphViewType());
  _inner_graph_ptr->InitializeFromGraph(*(_graph_def_ptr.get()));
  _inputs = pattern_ins();
  // return Status::OK();
}

void Pattern::LoadModel(const char* model_pb_path) {
  _graph_def_ptr = std::make_shared<GraphDef>();
  Status s =
      ReadBinaryProto(Env::Default(), model_pb_path, _graph_def_ptr.get());
  if (!s.ok()) {
    s = ReadTextProto(Env::Default(), model_pb_path, _graph_def_ptr.get());
  }
  if (!s.ok()) {
    LOG(ERROR) << "Could not create TensorFlow Graph: " << s;
    return;  // s;
  }
  _inner_graph_ptr.reset(new GraphViewType());
  _inner_graph_ptr->InitializeFromGraph(*(_graph_def_ptr.get()));
  _inputs = pattern_ins();
  // return Status::OK();
}

Pattern::Node Pattern::get_hint_node() {
  for (auto& node : nodes()) {
    if (!node.is_input() && node.op() == _hint_op_type) {
      return node;
    }
  }
  return Pattern::Node::null(this);
}

bool Pattern::set_hint_node_type(const std::string& op_type) {
  _hint_op_type = op_type;
}

std::vector<Pattern::Node> Pattern::nodes() {
  std::vector<Pattern::Node> results;
  for (const NodeDef& node : _inner_graph_ptr->graph()->node()) {
    results.push_back(Pattern::Node(node, this));
  }
  return results;
}

absl::flat_hash_set<Pattern::Node> Pattern::pattern_ins() {
  absl::flat_hash_set<Pattern::Node> ins;
  for (auto& node : nodes()) {
    if (node.is_input()) {
      ins.emplace(node);
    }
  }
  return ins;
}

absl::flat_hash_set<Pattern::Node> Pattern::pattern_outs() {
  /// TODO
}

Pattern CreatePattern(const char* model_pb_path) {
  Pattern pattern;
  pattern.LoadModel(model_pb_path);
  return pattern;
}

} /* namespace grappler */
} /* namespace tensorflow */
