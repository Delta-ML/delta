#ifndef _DELTA_INFER_ALGORITHM_H_
#define _DELTA_INFER_ALGORITHM_H_

#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace grappler {

class Algorithm {
 public:
  Algorithm(std::string name) : _name(name) {}
  ~Algorithm() {}

  virtual GraphDef* GenOptGraph() = 0;

 private:
  std::string _name;
};

class AlgoManager {
 public:
  typedef std::function<Algorithm*(void)> creator;

 public:
  static AlgoManager& Instance() {
    static AlgoManager algo_manager;
    return algo_manager;
  }

  bool Set(const std::string& name, creator algo) {
    if (!contain(name)) {
      _algo_map[name] = algo;
    }
    return true;
  }

  bool contain(const std::string& name) {
    for (auto it = _algo_map.begin(); it != _algo_map.end(); ++it) {
      if (it->first == name) {
        return true;
      }
    }
    return false;
  }

  creator Get(const std::string& name) {
    if (!contain(name)) {
      return creator(nullptr);
    }
    return _algo_map[name];
  }

 private:
  AlgoManager(){};
  std::unordered_map<std::string, creator> _algo_map;
};

#define REGISTER_ALGO_WITH_NAME(ALGO, NAME)                        \
  static bool new_algo##__COUNTER__ = AlgoManager::Instance().Set( \
      #NAME, []() -> Algorithm* { return new ALGO(#NAME); })

#define REGISTER_ALGO(ALGO) REGISTER_ALGO_WITH_NAME(ALGO, ALGO)

} /* namespace grappler */
} /* namespace tensorflow */

#endif
