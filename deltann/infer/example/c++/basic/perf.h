#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/util/stat_summarizer.h"

#include "tensorflow/core/graph/default_device.h"

namespace tensorflow {

struct InputLayerInfo {
  string name;
  DataType data_type;
  TensorShape shape;
  std::vector<float> initialization_values;
};

class Perf {
 public:
  typedef struct {
    string graph = "/path/to/graph.pb";
    string input_layer = "input:0";
    string input_layer_shape = "1,224,224,3";
    string input_layer_type = "float";
    string input_layer_values = "";
    string output_layer = "output:0";
    string target_layer = "";

    int num_threads = -1;
  } Config;

  typedef struct {
    bool show_run_order = true;
    int run_order_limit = 0;
    bool show_time = true;
    int time_limit = 10;
    bool show_memory = true;
    int memory_limit = 10;
    bool show_type = true;
    bool show_summary = true;
  } StatConfig;

 public:
  Perf(const Config* config);
  ~Perf() {}

  void with_stat(const StatConfig*);

  Status run(double& total_time_ms);

 private:
  const Config* cfg{nullptr};
  const StatConfig* stat_cfg{nullptr};
  // graph and session
  std::unique_ptr<Session> session{nullptr};
  std::unique_ptr<GraphDef> graph_def{nullptr};
  // extra execution info with stats
  std::unique_ptr<StatSummarizer> stats{nullptr};

  // session running config
  RunOptions run_options;
  RunMetadata run_metadata;

  // io of model
  std::vector<std::pair<string, tensorflow::Tensor> > input_tensors;
  std::vector<string> outputs;
  std::vector<string> targets;
};

}  // namespace tensorflow
