#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/config.pb.h"


namespace tensorflow {
namespace grappler {

// this is an example custom grappler optimization using tensorflow
class TestOptimizer : public CustomGraphOptimizer {
 public:
  static void SetOptimized(const bool flag_value) { optimized_ = flag_value; }
  static bool IsOptimized() { return optimized_; }

  TestOptimizer() {}
  string name() const override { return "test_optimizer"; }
  bool UsesFunctionLibrary() const { return false; }

  Status Init(const tensorflow::RewriterConfig_CustomGraphOptimizer* config =
                  nullptr) override {
    return Status::OK();
  }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override {
    optimized_ = true;
    *optimized_graph = item.graph;
    return Status::OK();
  }

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override {}

 private:
  static bool optimized_;
};

bool TestOptimizer::optimized_;

REGISTER_GRAPH_OPTIMIZER(TestOptimizer);

} /* namespace grappler */

} /* namespace tensorflow */
