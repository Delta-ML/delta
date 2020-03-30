#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/protobuf.h"
//#include "tensorflow/core/protobuf/config.pb.h"

#include "example/c++/basic/perf.h"

namespace tensorflow {

void perf_of_inference() {
    Perf::Config cfg;
    //cfg.graph = "/nfs/project/ccw/new/work_space/models/simplify/xingcheng_20190630.frozen_graph.pb";
    cfg.graph = "/tmp-data/test/self/delta_infer/delta_infer/example/python/result.pb";
    //cfg.graph = "/home/test/self/fastertransformer/DeepLearningExamples/FasterTransformer/build/nv_fasttransformer.pb";
        
    //cfg.input_layer = "import/e2e_model/Model_1/input_seq:0,import/e2e_model/Model_1/input_len:0,import/e2e_model/Model_1/input_mask:0";
    cfg.input_layer = "Placeholder:0,Placeholder_1:0";

    //cfg.input_layer_shape = "1000,1,40:1000:1000,1";
    cfg.input_layer_shape = "1,100,768:1,100,100";
    //cfg.input_layer_type = "float,int32,int32";
    cfg.input_layer_type = "float,int32";
    //cfg.input_layer_values = "0.42,1,1";
    cfg.input_layer_values = "1,1";
    //cfg.output_layer = "import/e2e_model/Model_1/attn_decoder/LASDecoder/decoding_output:0";
    cfg.output_layer = "TransformerCell0";
    //cfg.output_layer = "layer_0/output/LayerNorm/batchnorm/add_1";
    cfg.target_layer = "";
    cfg.num_threads = 1;
    Perf infer(&cfg);

    double warm_up_ms = 0.0;
    int warm_up = 10;
    for(int i=0; i<warm_up; i++) {
        infer.run(warm_up_ms);
    }

    int iteration=100;
    double total_time_ms=0.0;

    for(int i=0; i<iteration; i++) {
        infer.run(total_time_ms);
    }

    LOG(INFO) << "Average inference timings: "
              << "Warmup: " <<  warm_up_ms / warm_up << " ms, "
              << "with no stats: " << total_time_ms / iteration << " ms ";
}

void perf_of_inference_for_hy_bert() {
    Perf::Config cfg;
    cfg.graph = "/tmp-data/test/self/delta_infer/delta_infer/example/python/result.pb";
        
    //cfg.input_layer = "import/e2e_model/Model_1/input_seq:0,import/e2e_model/Model_1/input_len:0,import/e2e_model/Model_1/input_mask:0";
    cfg.input_layer = "input_x";

    //cfg.input_layer_shape = "1000,1,40:1000:1000,1";
    //cfg.input_layer_shape = "1,100,768:1,100,100";
    cfg.input_layer_shape = "1,512";
    //cfg.input_layer_type = "float,int32,int32";
    cfg.input_layer_type = "int32";
    //cfg.input_layer_values = "0.42,1,1";
    cfg.input_layer_values = "1";
    //cfg.output_layer = "import/e2e_model/Model_1/attn_decoder/LASDecoder/decoding_output:0";
    cfg.output_layer = "output/probs";
    //cfg.output_layer = "layer_0/output/LayerNorm/batchnorm/add_1";
    cfg.target_layer = "";
    cfg.num_threads = 1;
    Perf infer(&cfg);

    double warm_up_ms = 0.0;
    int warm_up = 10;
    for(int i=0; i<warm_up; i++) {
        infer.run(warm_up_ms);
    }

    int iteration=100;
    double total_time_ms=0.0;

    for(int i=0; i<iteration; i++) {
        infer.run(total_time_ms);
    }

    LOG(INFO) << "Average inference timings: "
              << "Warmup: " <<  warm_up_ms / warm_up << " ms, "
              << "with no stats: " << total_time_ms / iteration << " ms ";
}



void perf_with_stat() {
}

} // namepace tensorflow

int main(int argc, const char** argv) {
    //tensorflow::perf_of_inference();
    tensorflow::perf_of_inference_for_hy_bert();
    return 0;
}
