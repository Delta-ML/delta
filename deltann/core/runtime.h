/* Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef DELTANN_CORE_RUNTIME_H_
#define DELTANN_CORE_RUNTIME_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "core/config.h"
#include "core/graph.h"
#include "core/io.h"
#include "core/misc.h"
#include "core/tflite_model.h"
#include "core/tfmodel.h"
#include "core/tfserving_model.h"

#ifdef USE_TF
#include "tensorflow/c/c_api.h"
#endif

namespace delta {

namespace core {

using std::string;
using std::vector;

struct In {
  // Input for Constant TensorShape
  In(string graph_name, string input_name, const void* ptr, std::size_t size)
      : _graph_name(graph_name),
        _input_name(input_name),
        _ptr(ptr),
        _size(size) {}

  // Input for PartialTensorShape
  In(string graph_name, string input_name, const int* shape, const int ndims,
     const void* ptr, std::size_t size)
      : _graph_name(graph_name),
        _input_name(input_name),
        _ptr(ptr),
        _size(size) {
    _shape = Shape(shape, ndims);
  }

  std::string _graph_name;
  std::string _input_name;
  const void* _ptr;
  std::size_t _size;
  Shape _shape;
};

// run graph
class Runtime {
 public:
  explicit Runtime(RuntimeConfig& rt_cfg)
      : _rt_cfg(rt_cfg), _graphs(rt_cfg.graphs()) {}

  virtual ~Runtime() {
    // unload_models();
  }

  // all graphs name
  const std::vector<std::string> graphs_name() {
    std::vector<string> names;
    for (auto& graph : _graphs) {
      names.push_back(graph.first);
    }
    return names;
  }

  // load custom ops
  DeltaStatus load_custom_ops();

  // load model
  DeltaStatus load_model();

  DeltaStatus set_inputs(const std::vector<In>& inputs);

  // get all out result
  DeltaStatus get_outputs(std::vector<string>* results);

  // get one out result
  DeltaStatus get_output(const std::string& out_name, string* output) {
    return DeltaStatus::STATUS_OK;
  }

  // run
  DeltaStatus run();

  // run one graph, with inputs to get outputs;
  // DeltaStatus run(const Graph& graph,
  DeltaStatus run(Graph* graph, const std::vector<InputData>& inputs,
                  std::vector<OutputData>* outputs) {
    return graph->run(inputs, outputs);
  }

  // load model if not loaded, then warmup model;
  DeltaStatus warmup();

  // reload model with new config, then warmup model;
  DeltaStatus warmup(bool reset) {}

  // number of output nodes
  int get_output_num() { return _outputs_data.size(); }

#define CHECK_OUTPUT_INDEX(index) \
  int num = _outputs_data.size(); \
  DELTA_CHECK_LT(index, num);     \
  DELTA_CHECK_GE(index, 0);

  // the rank of the `output_index`
  int get_output_ndim(int output_index) {
    CHECK_OUTPUT_INDEX(output_index);
    return _outputs_data[output_index].ndim();
  }

  // the `dim_index` number of the `output_index`
  int get_output_dim(int output_index, int dim_index) {
    CHECK_OUTPUT_INDEX(output_index);

    int ndim = _outputs_data[output_index].ndim();
    DELTA_CHECK_LT(dim_index, ndim);
    DELTA_CHECK_GE(dim_index, 0);

    auto& shape = _outputs_data[output_index].shape();

    return shape[dim_index];
  }

  int get_output_bytesize(int output_index) {
    CHECK_OUTPUT_INDEX(output_index);
    return _outputs_data[output_index].bytes();
  }

  int copy_to_buffer(int output_index, void* data, int data_byte_size) {
    CHECK_OUTPUT_INDEX(output_index);
    _outputs_data[output_index].copy_to(data, data_byte_size);
    return 0;
  }

#undef CHECK_OUTPUT_INDEX

 private:
  // graphs
  const RuntimeConfig& _rt_cfg;
  std::unordered_map<std::string, Graph>& _graphs;

  std::vector<InputData> _inputs_data;
  std::vector<OutputData> _outputs_data;
};

}  // namespace core

}  // namespace delta

#endif  // DELTANN_CORE_RUNTIME_H_
