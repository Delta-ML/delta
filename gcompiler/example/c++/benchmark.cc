#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/protobuf.h"
//#include "tensorflow/core/protobuf/config.pb.h"

#include "example/c++/basic/perf.h"

namespace tensorflow {

void perf_of_inference() {
  Perf::Config cfg;
  //cfg.graph = "/tmp-data/test/self/delta_infer/delta_infer/example/python/result.pb";
  cfg.graph = "/tmp-data/zhanghui/delta/gcompiler/frozen_graph_30sec_e2e.pb";

  //cfg.input_layer = "Placeholder:0,Placeholder_1:0";
  cfg.input_layer = "inputs:0";

  //cfg.input_layer_shape = "1,100,768:1,100,100";
  cfg.input_layer_shape = "32,3000,40,3";
  //cfg.input_layer_type = "float,int32";
  cfg.input_layer_type = "float";
  //cfg.input_layer_values = "1,1";
  cfg.input_layer_values = "";
  //cfg.output_layer = "import/e2e_model/Model_1/attn_decoder/LASDecoder/decoding_output:0";
  cfg.output_layer =  "softmax_output:0";
  cfg.target_layer = "";
  cfg.num_threads = 1;
  Perf infer(&cfg);

  double warm_up_ms = 0.0;
  int warm_up = 10;
  for (int i = 0; i < warm_up; i++) {
    infer.run(warm_up_ms);
  }

  int iteration = 100;
  double total_time_ms = 0.0;

  for (int i = 0; i < iteration; i++) {
    infer.run(total_time_ms);
  }

  LOG(INFO) << "Average inference timings: "
            << "Warmup: " << warm_up_ms / warm_up << " ms, "
            << "with no stats: " << total_time_ms / iteration << " ms ";
}

}  // namespace tensorflow

int main(int argc, const char** argv) {
  tensorflow::perf_of_inference();
  return 0;
}
