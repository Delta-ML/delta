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

void perf_of_transformercell() {
    Perf::Config cfg;
    cfg.graph = "/tmp-data/test/self/delta_infer/delta_infer/example/python/nlp_fasttransformer_cell.pb";
    //cfg.graph = "/tmp-data/test/self/delta_infer/delta_infer/example/python/result.pb";
        
    cfg.input_layer = "Placeholder:0,Placeholder_1:0,Placeholder_2:0,Placeholder_3:0,Placeholder_4:0,Placeholder_5:0";

    cfg.input_layer_shape = "100,4,256:100,1,256:6,64:6,64:100,116:16,4,256";
    cfg.input_layer_type = "float,float,float,float,float,float";
    cfg.input_layer_values = "1:1:1:1:1:1";
    //cfg.output_layer = "import/e2e_model/Model_1/attn_decoder/LASDecoder/decoding_output:0";
    cfg.output_layer = "y/Tensordot:0";//"TransformerCellNLP0:0";
    //cfg.output_layer = "layer_0/output/LayerNorm/batchnorm/add_1";
    cfg.target_layer = "";
    cfg.num_threads = 1;
    Perf infer(&cfg);

    double warm_up_ms = 0.0;
    int warm_up = 0;
    for(int i=0; i<warm_up; i++) {
        infer.run(warm_up_ms);
    }

    int iteration=1;
    double total_time_ms=0.0;

    for(int i=0; i<iteration; i++) {
        infer.run(total_time_ms);
    }

    LOG(INFO) << "Average inference timings: "
              << "Warmup: " <<  warm_up_ms / warm_up << " ms, "
              << "with no stats: " << total_time_ms / iteration << " ms ";
}

void perf_of_inference() {
    Perf::Config cfg;
    //cfg.graph = "/tmp-data/test/self/delta_infer/delta_infer/example/python/model.pb";
    cfg.graph = "/tmp-data/test/self/delta_infer/delta_infer/example/python/result.pb";
        
    cfg.input_layer = "input0:0,input1:0,input2:0,input3:0,input4:0,input5:0,input:0";

    cfg.input_layer_shape = "16,4,256:16,4,256:16,4,256:16,4,256:16,4,256:16,4,256:4,100";
    cfg.input_layer_type = "float,float,float,float,float,float,int32";
    cfg.input_layer_values = "2:2:2:2:2:2:2";
    //cfg.output_layer = "import/e2e_model/Model_1/attn_decoder/LASDecoder/decoding_output:0";
    cfg.output_layer = "frozen_output_:0";
    //cfg.output_layer = "layer_0/output/LayerNorm/batchnorm/add_1";
    cfg.target_layer = "";
    cfg.num_threads = 1;
    Perf infer(&cfg);

    double warm_up_ms = 0.0;
    int warm_up = 10;
    for(int i=0; i<warm_up; i++) {
        infer.run(warm_up_ms);
    }

    int iteration=1000;
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
    tensorflow::perf_of_inference();
    //tensorflow::perf_of_transformercell();
    return 0;
}
