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

void perf_of_inference(const char * graph, int batch, int  seq) {
    Perf::Config cfg;
    cfg.graph = graph;
        
    cfg.input_layer = "input0:0,input1:0,input2:0,input3:0,input4:0,input5:0,input:0";

    char buf[1000];
    auto ret = snprintf(buf, 1000, "16,%d,256:16,%d,256:16,%d,256:16,%d,256:16,%d,256:16,%d,256:%d,%d", batch, batch, batch, batch, batch, batch, batch, seq);
    if(ret>=0 && ret <1000) {
        cfg.input_layer_shape = std::string(buf);
        fprintf(stdout, "Get %s \n", cfg.input_layer_shape.c_str());
    } else {
        fprintf(stderr, "Error! construct input layer shape string error!\n");
        exit(1);
    }
    cfg.input_layer_type = "float,float,float,float,float,float,int32";
    cfg.input_layer_values = "2:2:2:2:2:2:2";
    cfg.output_layer = "frozen_output_:0";
    cfg.target_layer = "";
    cfg.num_threads = 1;
    Perf infer(&cfg);

    double warm_up_ms = 0.0;
    int warm_up = 10;
    for(int i=0; i<warm_up; i++) {
        infer.run(warm_up_ms);
    }

    int iteration=50;
    double total_time_ms=0.0;

    for(int i=0; i<iteration; i++) {
        infer.run(total_time_ms);
    }

    LOG(INFO) << graph << ","<<batch<<","<<seq
              <<", Average inference timings: "
              << " Warmup: " <<  warm_up_ms / warm_up << " ms, "
              << " with no stats: " << total_time_ms / iteration << " ms ";
}

} // namepace tensorflow

int main(int argc, const char** argv) {
    //int fd = open(argv[0], O_RDONLY);
    //tensorflow::test(fd);
    std::vector<int> batchs{4, 8, 16, 32, 64, 128};
    std::vector<int> seqs{50, 100, 200, 300, 400, 500, 600, 700, 800,900, 1000};
    const char * graph = argv[1];
    int batch = atoi(argv[2]);
    //for(auto btid : batchs) {
        for(auto seq : seqs) {
            tensorflow::perf_of_inference(graph, batch, seq);
        }
    //}
    return 0;
}
