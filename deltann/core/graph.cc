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

#include "core/graph.h"
#include "core/config.h"

namespace delta {
namespace core {

using std::pair;
using std::string;

Graph::Graph(const YAML::Node& cfg) : _cfg(cfg) {
  try {
    _name = cfg["name"].as<string>();
    string server_type = cfg["server_type"].as<string>();
    int version = cfg["version"].as<int>();
    _engine_type = cfg["engine"].as<string>();
    string model_path = cfg["local"]["path"].as<string>();
    DELTA_CHECK(version > 0);
    LOG_INFO << "graph name: [" << _name << "]";
    LOG_INFO << "server type: " << server_type;
    LOG_INFO << "engine: " << _engine_type;
    LOG_INFO << "version: " << version;

    _model_meta.server_type = server_type;
    _model_meta.version = version;
    if (server_type == "local") {
      LOG_INFO << "load local model";
      _model_meta.local.model_path = model_path + "/" + std::to_string(version);
      LOG_INFO << "model path: " << _model_meta.local.model_path;

      string model_type = cfg["local"]["model_type"].as<string>();
      if (model_type == kSavedModel) {
        _model_meta.local.model_type = ModelType::MODEL_SAVED_MODEL;
      } else if (model_type == kFrozenGraphPb) {
        _model_meta.local.model_type = ModelType::MODEL_FROZEN_PB;
      } else if (model_type == kFrozenGraphTxt) {
        _model_meta.local.model_type = ModelType::MODEL_FROZEN_PB_TEXT;
      } else {
        LOG_FATAL << "Error, not support model_type " << model_type;
      }

      LOG_INFO << "model type : "
               << static_cast<std::underlying_type<ModelType>::type>(
                      _model_meta.local.model_type);

    } else if (server_type == "remote") {
      LOG_INFO << "load remote model";
      _model_meta.remote.hostname = cfg["remote"]["host"].as<string>();
      _model_meta.remote.port = cfg["remote"]["port"].as<std::uint16_t>();
      _model_meta.remote.model_name = cfg["remote"]["model_name"].as<string>();

    } else {
      LOG_FATAL << "Error, not support server_type " << server_type;
    }
  } catch (const YAML::Exception& e) {
    LOG_FATAL << "Error, read config failed: " << e.what();
  }

  DELTA_ASSERT_OK(add_inputs());
  DELTA_ASSERT_OK(add_outputs());
}

Graph::~Graph() {
  if (_model) {
    delete _model;
  }
  _model = nullptr;
  _inputs.clear();
  _outputs.clear();
}

// inputs
DeltaStatus Graph::add_inputs() {
  int in_num = _cfg["inputs"].size();
  LOG_INFO << "inputs num is " << in_num;
  for (int i = 0; i < in_num; ++i) {
    LOG_INFO << "in name is " << _cfg["inputs"][i]["name"];
    LOG_INFO << "in dtype is " << _cfg["inputs"][i]["dtype"];

    string name = _cfg["inputs"][i]["name"].as<string>();
    string dtype = _cfg["inputs"][i]["dtype"].as<string>();
    int id = _cfg["inputs"][i]["id"].as<int>();

    if (_cfg["inputs"][i]["shape"]) {
      LOG_INFO << "in shape is " << _cfg["inputs"][i]["shape"];
      YAML::Node s = _cfg["inputs"][i]["shape"];
      std::vector<int> v;
      for (std::size_t i = 0; i < s.size(); i++) {
        v.push_back(s[i].as<int>());
      }
      _inputs.insert(pair<string, Input>(
          name, Input(name, id, Shape(v), delta_str_dtype(dtype))));
    } else {
      LOG_INFO << "graph " << _name << " shape is None"
               << _cfg["inputs"][i]["shape"];
      _inputs.insert(
          pair<string, Input>(name, Input(name, id, delta_str_dtype(dtype))));
    }
  }

  return DeltaStatus::STATUS_OK;
}

// outputs
DeltaStatus Graph::add_outputs() {
  int out_num = _cfg["outputs"].size();
  LOG_INFO << "output num is " << out_num;

  for (int i = 0; i < out_num; ++i) {
    LOG_INFO << "out name is " << _cfg["outputs"][i]["name"];
    LOG_INFO << "out dtype is " << _cfg["outputs"][i]["dtype"];
    string name = _cfg["outputs"][i]["name"].as<string>();
    string dtype = _cfg["outputs"][i]["dtype"].as<string>();

    int id = _cfg["outputs"][i]["id"].as<int>();
    _outputs.insert(
        pair<string, Output>(name, Output(name, id, delta_str_dtype(dtype))));
  }
  return DeltaStatus::STATUS_OK;
}

string Graph::engine_type() const { return _engine_type; }

ModelMeta Graph::model_meta() const { return _model_meta; }

DeltaStatus Graph::set_model(BaseModel* model) { _model = model; }

}  // namespace core
}  // namespace delta
