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
#include "core/runtime.h"

namespace delta {

namespace core {

namespace {

// load ops lib
DeltaStatus load_custom_ops_lib(const std::string& lib) {
  if (!lib.size()) {
    LOG_WARN << "no custom ops to load";
    return DeltaStatus::STATUS_OK;
  }

#ifdef USE_TF
  LOG_INFO << "custom op lib is: " << lib;
  TF_Status* status = TF_NewStatus();
  TF_Library* custom_op_lib = TF_LoadLibrary(lib.c_str(), status);

  TF_Code code = TF_GetCode(status);
  string status_msg(TF_Message(status));
  TF_DeleteStatus(status);
  if (TF_OK != code) {
    LOG_FATAL << status_msg;
  }
  LOG_INFO << "custom op lib load succesfully" << lib;

  TF_Buffer op_list_buf = TF_GetOpList(custom_op_lib);
  tensorflow::OpList op_list;
  DELTA_CHECK(op_list.ParseFromArray(op_list_buf.data, op_list_buf.length));
  for (int i = 0; i != op_list.op_size(); ++i) {
    LOG_INFO << "cutsom op: " << op_list.op(i).name();
  }

  TF_DeleteLibraryHandle(custom_op_lib);
#endif  // USE_TF
  return DeltaStatus::STATUS_OK;
}

// load all models
DeltaStatus load_models(const RuntimeConfig& rt_cfg,
                        std::unordered_map<std::string, Graph>* graphs) {
  const int num_threads = rt_cfg.num_threads();

  if (!graphs->size()) {
    LOG_WARN << "graphs size is empty, " << graphs->size();
    return DeltaStatus::STATUS_ERROR;
  }

  for (auto& iter : *graphs) {
    LOG_INFO << "Load model for graph " << iter.first;

    Graph& graph = iter.second;
    std::string engine_type = graph.engine_type();
    ModelMeta model_meta = graph.model_meta();
    auto search = _global_engines.find(engine_type);

    BaseModel* model = nullptr;

    if (search != _global_engines.end()) {
#ifdef USE_TF
      if (EngineType::DELTA_EIGINE_TF == _global_engines[engine_type]) {
        LOG_INFO << "User engine tf";
        model = new TFModel(model_meta, num_threads);
#endif

      } else if (EngineType::DELTA_EIGINE_TFTRT ==
                 _global_engines[engine_type]) {
        LOG_INFO << "User engine tftrt";

#ifdef USE_TFLITE
      } else if (EngineType::DELTA_EIGINE_TFLITE ==
                 _global_engines[engine_type]) {
        LOG_INFO << "User engine tf lite";
        model = new TFLiteModel(model_meta, num_threads);
#endif

#ifdef USE_TF_SERVING
      } else if (EngineType::DELTA_EIGINE_TFSERVING ==
                 _global_engines[engine_type]) {
        LOG_INFO << "User engine TFSERVING";
        model = new TFServingModel(model_meta, num_threads);
#endif

      } else if (EngineType::DELTA_EIGINE_SERVING ==
                 _global_engines[engine_type]) {
        LOG_FATAL << "Not support now SERVING";
      } else {
        LOG_FATAL << "Not support engine type: " << engine_type;
      }
    } else {
      LOG_FATAL << "Not support engine";
      return DeltaStatus::STATUS_ERROR;
    }

    graph.set_model(model);
    model = nullptr;
  }  // for

  return DeltaStatus::STATUS_OK;
}

}  // namespace

// load custom ops
DeltaStatus Runtime::load_custom_ops() {
  return load_custom_ops_lib(_rt_cfg.custom_ops_path());
}

// load model
DeltaStatus Runtime::load_model() {
  if (DeltaStatus::STATUS_OK != load_models(_rt_cfg, &_graphs)) {
    LOG_FATAL << "load model failed";
    return DeltaStatus::STATUS_ERROR;
  }
  return DeltaStatus::STATUS_OK;
}

DeltaStatus Runtime::set_inputs(const std::vector<In>& ins) {
  _inputs_data.clear();

  for (auto& in : ins) {
    LOG_INFO << "Graph name: " << in._graph_name;
    auto search = _graphs.find(in._graph_name);
    if (search != _graphs.end()) {
      Graph& graph = search->second;

      LOG_INFO << "input name: " << in._input_name;
      try {
        Input& input = graph.get_inputs().at(in._input_name);

        if (input.shape().is_partial()) {
          LOG_INFO << "Override partial shape: " << input.shape() << " to "
                   << in._shape;
          input.set_shape(in._shape);
        }
        DELTA_CHECK_EQ(in._size, input.size() * delta_dtype_size(input.dtype()))
            << in._size << ":"
            << input.size() * delta_dtype_size(input.dtype());

        InputData input_data(input);
        input_data.copy_from(in._ptr, in._size);
        _inputs_data.push_back(input_data);
      } catch (std::out_of_range& e) {
        LOG_FATAL << "Can not find [" << in._input_name << "] node in ["
                  << in._graph_name << "] graph: " << e.what();
      }
    } else {
      LOG_FATAL << "Error, Graph " << in._graph_name << " not exist!";
    }
  }
  return DeltaStatus::STATUS_OK;
}

// load model if not loaded, then warmup model;
DeltaStatus Runtime::warmup() {
  if (DeltaStatus::STATUS_OK != load_custom_ops()) {
    LOG_FATAL << "load custom ops failed";
    return DeltaStatus::STATUS_ERROR;
  }

  if (DeltaStatus::STATUS_OK != load_model()) {
    LOG_FATAL << "load model failed";
    return DeltaStatus::STATUS_ERROR;
  }

  LOG_INFO << "Warmup ...";
  for (auto& graph : _graphs) {
    std::vector<InputData> inputs_data;
    std::vector<OutputData> outputs_data;

    std::unordered_map<std::string, Input>& inputs = graph.second.get_inputs();
    for (auto& input : inputs) {
      Input& in(input.second);

      LOG_INFO << in;
      LOG_INFO << in.shape().ndim();

      if (in.shape().is_partial()) {
        in.shape().set_dim(0, 1);
      }

      LOG_INFO << in;
      LOG_INFO << in.size();

      InputData in_data(in);
      in_data.feed_random_data();
      inputs_data.push_back(in_data);
    }

    std::unordered_map<std::string, Output>& outputs =
        graph.second.get_outputs();
    for (auto& output : outputs) {
      LOG_INFO << output.second;
      outputs_data.push_back(OutputData(output.second));
    }

    if (DeltaStatus::STATUS_OK !=
        this->run(&(graph.second), inputs_data, &outputs_data)) {
      LOG_FATAL << "run failed";
      return DeltaStatus::STATUS_ERROR;
    }
  }

  LOG_INFO << "Warmup Done!";
  return DeltaStatus::STATUS_OK;
}

// run
// only support one Graph now!!!! how to support multi-graphs ???
DeltaStatus Runtime::run() {
  _outputs_data.clear();

  Graph& graph = _graphs.begin()->second;

  // out
  std::unordered_map<std::string, Output>& outputs = graph.get_outputs();
  for (auto& output : outputs) {
    _outputs_data.push_back(OutputData(output.second));
  }

  // run
  if (DeltaStatus::STATUS_OK !=
      this->run(&(graph), _inputs_data, &_outputs_data)) {
    LOG_FATAL << "run failed";
    return DeltaStatus::STATUS_ERROR;
  }
}

// get all out result
DeltaStatus Runtime::get_outputs(std::vector<string>* results) {
  // get resulst
  results->clear();
  for (auto& output_data : _outputs_data) {
    LOG_INFO << "out shape is " << output_data.shape();
    DataType dtype = output_data.dtype();
    int size = output_data.size();
    LOG_INFO << "out size is " << size;
    std::stringstream ss;
    switch (dtype) {
      case DataType::DELTA_FLOAT32: {
        float* ptr = static_cast<float*>(output_data.ptr());
        for (int i = 0; i < size; ++i) {
          ss << ptr[i] << ",";
        }
      } break;
      case DataType::DELTA_INT32: {
        int* ptr = static_cast<int*>(output_data.ptr());
        for (int i = 0; i < size; ++i) {
          ss << ptr[i] << ",";
        }
      } break;
      case DataType::DELTA_CHAR: {
        LOG_FATAL << "DataType::CHAR do nothing";
      }
      default:
        LOG_FATAL << "Error, not support dtype ";
    }
    results->push_back(ss.str());
    LOG_INFO << "results are " << *results;
  }

  return DeltaStatus::STATUS_OK;
}

}  // namespace core

}  // namespace delta
