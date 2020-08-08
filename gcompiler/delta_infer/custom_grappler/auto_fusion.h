#ifndef _DELTA_INFER_AUTO_FUSION_H_
#define _DELTA_INFER_AUTO_FUSION_H_

#include "delta_infer/core/scatter_search.h"
#include "delta_infer/custom_grappler/local_optimizer.h"

namespace tensorflow {
namespace grappler {

/// help function to regist grpah pattern for automatic fusion
void RegisterFusionPattern(const std::string& name, const Pattern& graph,
                           const std::string& hint_op);

class AutoFusion : public LocalOptimizer {
 public:
  explicit AutoFusion(const string& name) : LocalOptimizer(name) {}

  virtual GraphDef* GenOptGraph(const GraphDef* original_graph) final;
};

} /* namespace grappler */
} /* namespace tensorflow */

#endif
