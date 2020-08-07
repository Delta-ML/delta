#include "delta_infer/custom_grappler/auto_fusion.h"
#include <fstream>
#include <iostream>

namespace tensorflow {
namespace grappler {

void serialization(const GraphDef* graph, const char* path) {
  std::string out_graph;
  std::ofstream fout;
  graph->SerializeToString(&out_graph);
  fout.open(path, std::ios::binary | std::ios::out);
  fout << out_graph;
  fout.close();
}

void test_for_auto_fusion() {
  // load target original graph
  Pattern ori_graph;
  // ori_graph.LoadModel("/tmp-data/test/self/self_tf/my_test/transformer_pattern.pb");
  // ori_graph.LoadModel("/tmp-data/test/self/delta_infer/delta_infer/example/python/model.pb");
  ori_graph.LoadModel("/path/to/nv_fasttransformer.pb");
  // Register pattern
  Pattern pattern;
  // pattern.LoadModel("/home/test/work/delta_infer/delta_infer/test/subgraphs/TansformerCell.pb");
  // pattern.LoadModel("/tmp-data/test/self/delta_infer/delta_infer/example/python/TransformerCell_1.pb");
  pattern.LoadModel("/path/to/TansformerCell.pb");
  RegisterFusionPattern("TransformerCell", pattern, "BatchMatMulV2");
  // RegisterFusionPattern("TansformerCell", pattern, "BatchMatMul");

  // invocke auto fusion
  auto* local_optimizer = LocalOptimizerManager::Instance().Get("AutoFusion")();
  AutoFusion* auto_fusion_optimizer =
      dynamic_cast<AutoFusion*>(local_optimizer);
  if (auto_fusion_optimizer) {
    auto* optimized_graph_def =
        auto_fusion_optimizer->GenOptGraph(ori_graph.graph());
    serialization(optimized_graph_def, "./generation.pb");
  } else {
    printf("dynamic cast error!\n");
    exit(1);
  }
}

}  // namespace grappler
}  // namespace tensorflow

int main(int argc, const char** argv) {
  tensorflow::grappler::test_for_auto_fusion();
  return 1;
}
