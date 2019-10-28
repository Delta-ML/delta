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

#ifndef DELTANN_CORE_GRAPH_H_
#define DELTANN_CORE_GRAPH_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "yaml-cpp/yaml.h"

#include "core/base_model.h"
#include "core/io.h"
#include "core/misc.h"

namespace delta {
namespace core {

using std::string;
using std::vector;

// graphs with inputs/outputs and meta of graph
class Graph {
 private:
  ModelMeta _model_meta;
  std::string _name;
  std::string _engine_type;
  YAML::Node _cfg;
  std::unordered_map<std::string, Input> _inputs;
  std::unordered_map<std::string, Output> _outputs;

  BaseModel* _model = nullptr;

 private:
  DeltaStatus add_inputs();
  DeltaStatus add_outputs();

 public:
  /*
  Graph(DeltaMode mode, ModelMeta model_meta):
      _mode(mode), _model(model_meta)  {
  }
  */
  Graph();

  explicit Graph(const YAML::Node& cfg);

  ~Graph();

  std::string engine_type() const;
  ModelMeta model_meta() const;
  DeltaStatus set_model(BaseModel* model);

  DeltaStatus run(const std::vector<InputData>& inputs,
                  std::vector<OutputData>* outputs) {
    if (!_model) {
      LOG_FATAL << "Error, _model is nullptr!";
    }
    if (_model->run(inputs, outputs)) {
      return DeltaStatus::STATUS_ERROR;
    }
    return DeltaStatus::STATUS_OK;
  }

  std::unordered_map<std::string, Input>& get_inputs() { return _inputs; }

  std::unordered_map<std::string, Output>& get_outputs() { return _outputs; }

  // DeltaStatus load_model(RuntimeConfig* rt_cfg);

  // Status add_input(Input& in);
  // Status add_output(Output& out);

  // std::unordered_map<std::string, Input>& inputs() const;
  // std::unordered_map<std::string, Output>& outputs() const;
};

}  // namespace core
}  // namespace delta

#endif  // DELTANN_CORE_GRAPH_H_
