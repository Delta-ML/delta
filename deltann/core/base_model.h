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

#ifndef DELTANN_CORE_BASE_MODEL_H_
#define DELTANN_CORE_BASE_MODEL_H_

#include <string>
#include <utility>
#include <vector>

#include "core/io.h"
#include "core/misc.h"

namespace delta {
namespace core {

constexpr char kSavedModel[] = "saved_model";
constexpr char kFrozenGraphPb[] = "frozen_graph_pb";
constexpr char kFrozenGraphTxt[] = "frozen_graph_txt";

enum class ModelType {
  MODEL_SAVED_MODEL = 0,
  MODEL_FROZEN_PB,
  MODEL_FROZEN_PB_TEXT
};

// local or remote model meta
struct ModelMeta {
  EngineType engine;
  string server_type;
  int version;

  struct {
    ModelType model_type;    // saved_model, pb
    std::string model_path;  // model path
  } local;

  struct {
    // serving  model tag
    string model_name;
    // serving config
    string hostname;
    unsigned short port;
  } remote;
};

class BaseModel {
 public:
  BaseModel(ModelMeta model_meta, int num_threads);
  virtual ~BaseModel();
  virtual int run(const std::vector<InputData>& inputs,
                  std::vector<OutputData>* output) = 0;

 protected:
  ModelMeta _model_meta;
  const int _num_threads;
};

}  // namespace core
}  // namespace delta

#endif  // DELTANN_CORE_BASE_MODEL_H_
