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

void test_for_optimizer() {
  const std::vector<string> optimizers =
      CustomGraphOptimizerRegistry::GetRegisteredOptimizers();
  if (std::count(optimizers.begin(), optimizers.end(), "TestOptimizer") != 1) {
    std::cout << "get test_optimizer error!\n";
  }
  std::unique_ptr<const CustomGraphOptimizer> test_optimizer =
      CustomGraphOptimizerRegistry::CreateByNameOrNull("TestOptimizer");

  if (!test_optimizer) {
    std::cout << "get test_optimizer error!\n";
    exit(1);
  }
  std::cout << "Get registered optimizer: " << test_optimizer->name()
            << " suc!\n";
}

}  // namespace grappler
}  // namespace tensorflow

int main(int argc, const char** argv) {
  tensorflow::grappler::test_for_optimizer();
  return 1;
}
