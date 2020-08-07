#ifndef _DELTA_INFER_LOCAL_OPTIMIZER_H_
#define _DELTA_INFER_LOCAL_OPTIMIZER_H_

#include "delta_infer/core/pattern.h"

namespace tensorflow {
namespace grappler {

class LocalOptimizer {
 public:
  LocalOptimizer(const string& name) : _name(name) {}

  virtual GraphDef* GenOptGraph(const GraphDef* original_graph) = 0;

  const std::string& GetName() const { return _name; }

 private:
  std::string _name;
};

class LocalOptimizerManager {
 public:
  typedef std::function<LocalOptimizer*(void)> creator;

 public:
  static LocalOptimizerManager& Instance() {
    static LocalOptimizerManager _ins;
    return _ins;
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

  std::vector<std::string> names() {
    std::vector<std::string> ret;
    for (auto it = _algo_map.begin(); it != _algo_map.end(); ++it) {
      ret.push_back(it->first);
    }
    return ret;
  }

 private:
  LocalOptimizerManager() {}
  std::unordered_map<std::string, creator> _algo_map;
};

#define REGISTER_LOCAL_OPTIMIZER_WITH_NAME(ALGO, NAME)                       \
  static bool new_algo##__COUNTER__ = LocalOptimizerManager::Instance().Set( \
      #NAME, []() -> LocalOptimizer* { return new ALGO(#NAME); })

#define REGISTER_LOCAL_OPTIMIZER(ALGO) \
  REGISTER_LOCAL_OPTIMIZER_WITH_NAME(ALGO, ALGO)

} /* namespace grappler */
} /* namespace tensorflow */

#endif
