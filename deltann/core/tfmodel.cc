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

#ifdef USE_TF
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/tfmodel.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/types.pb.h"

namespace delta {
namespace core {

namespace {
tensorflow::DataType tf_data_type(DataType type) {
  switch (type) {
    case DataType::DELTA_FLOAT32:
      return tensorflow::DataTypeToEnum<float>::v();
    case DataType::DELTA_INT32:
      return tensorflow::DataTypeToEnum<int>::v();
    case DataType::DELTA_CHAR:
      return tensorflow::DataTypeToEnum<string>::v();
    default:
      assert(false);
      return tensorflow::DataType::DT_INVALID;
  }
}

}  // namespace

using std::string;
using tensorflow::ConfigProto;
using tensorflow::GraphDef;
using tensorflow::NewSession;
using tensorflow::ReadBinaryProto;
using tensorflow::RunOptions;
using tensorflow::SessionOptions;

TFModel::TFModel(ModelMeta model_meta, int num_threads)
    : BaseModel(model_meta, num_threads) {
  load_from_saved_model();
  // TODO(gaoyonghu): load frozen graph
  // load_from_frozen_graph();
}

void TFModel::feed_tensor(Tensor* tensor, const InputData& input) {
  std::int64_t num_elements = tensor->NumElements();
  switch (input.dtype()) {
    case DataType::DELTA_FLOAT32:
      std::copy_n(static_cast<float*>(input.ptr()), num_elements,
                  tensor->flat<float>().data());
      break;
    case DataType::DELTA_INT32:
      std::copy_n(static_cast<int*>(input.ptr()), num_elements,
                  tensor->flat<int>().data());
      break;
    case DataType::DELTA_CHAR: {
      char* cstr = static_cast<char*>(input.ptr());
      std::string str = std::string(cstr);
      tensor->scalar<std::string>()() = str;
    } break;
    default:
      assert(false);
  }
}

void TFModel::fetch_tensor(const Tensor& tensor, OutputData* output) {
  // set shape
  const tensorflow::TensorShape& ts = tensor.shape();
  output->set_shape(ts);

  // copy data
  std::size_t num_elements = tensor.NumElements();

  switch (tensor.dtype()) {
    case tensorflow::DT_FLOAT: {
      output->set_dtype(DataType::DELTA_FLOAT32);
      output->resize(num_elements);
      auto c = tensor.flat<float>();
      float* ptr = static_cast<float*>(output->ptr());
      for (int i = 0; i < num_elements; i++) {
        ptr[i] = c(i);
      }
    } break;
    case tensorflow::DT_INT32: {
      output->set_dtype(DataType::DELTA_INT32);
      output->resize(num_elements);
      auto c = tensor.flat<int>();
      int* ptr = static_cast<int*>(output->ptr());
      for (int i = 0; i < num_elements; i++) {
        ptr[i] = c(i);
      }
    } break;
    default:
      LOG_FATAL << "not support tensorflow datatype" << tensor.dtype();
      break;
  }
}

int TFModel::set_feeds(std::vector<std::pair<string, Tensor>>* feeds,
                       const std::vector<InputData>& inputs) {
  // set input
  DELTA_CHECK(inputs.size()) << "inputs size is 0";
  for (auto& input : inputs) {
    feeds->emplace_back(std::pair<string, Tensor>(
        input.name(),
        std::move(Tensor(tf_data_type(input.dtype()), input.tensor_shape()))));
    feed_tensor(&(feeds->at(feeds->size() - 1).second), input);
  }

  return 0;
}

int TFModel::set_fetches(std::vector<string>* fetches,
                         const std::vector<OutputData>& outputs) {
  // set input
  DELTA_CHECK(outputs.size()) << "outputs size is 0";
  for (auto& output : outputs) {
    fetches->push_back(output.name());
  }
}

int TFModel::get_featches(const std::vector<Tensor>& output_tensors,
                          std::vector<OutputData>* outputs) {
  int size = output_tensors.size();
  DELTA_CHECK_EQ(outputs->size(), size);
  for (int i = 0; i < size; ++i) {
    fetch_tensor(output_tensors[i], &(outputs->at(i)));
  }
  return 0;
}

int TFModel::run(const std::vector<InputData>& inputs,
                 std::vector<OutputData>* output) {
  LOG_INFO << "TFModel run ...";

  std::vector<std::pair<std::string, Tensor>> feeds;
  std::vector<std::string> fetches;
  std::vector<Tensor> output_tensors;

  set_feeds(&feeds, inputs);
  set_fetches(&fetches, *output);

  // Session run
  RunOptions run_options;
  Status s = _bundle.session->Run(run_options, feeds, fetches, {},
                                  &(output_tensors), nullptr);
  if (!s.ok()) {
    LOG_FATAL << "Error, TF Model run failed " << s;
    exit(-1);
  }

  get_featches(output_tensors, output);

  return 0;
}

Status TFModel::load_from_frozen_graph() {
  Status s;
  std::unique_ptr<GraphDef> graph_def;
  std::string path = _model_meta.local.model_path;
  if (path.empty()) {
    LOG_FATAL << "model path is empty" << path;
    return s;
  }
  std::string model_name = path + "/frozen_graph.pb";

  graph_def.reset(new GraphDef());
  // Read graph from disk
  s = ReadBinaryProto(tensorflow::Env::Default(), model_name, graph_def.get());
  if (!s.ok()) {
    LOG_FATAL << "Could not create TensorFlow Graph: " << s;
    return s;
  }
  LOG_INFO << "Loaded Graph from " << model_name;

  // create session options
  SessionOptions options;
  ConfigProto& config = options.config;
  if (_num_threads > 0) {
    config.set_intra_op_parallelism_threads(1);
    config.set_inter_op_parallelism_threads(_num_threads);
    config.add_session_inter_op_thread_pool();
    auto pool = config.mutable_session_inter_op_thread_pool(0);
    pool->set_num_threads(_num_threads);
    LOG_INFO << "Set thread pool size: " << pool->num_threads();
  }
  LOG_INFO << "Got config, " << _num_threads << " threads";

  // create session
  _bundle.session.reset(tensorflow::NewSession(options));
  s = _bundle.session->Create(*(graph_def.get()));
  if (!s.ok()) {
    LOG_FATAL << "Could not create TensorFlow Session: " << s;
    return s;
  }
  LOG_INFO << "Created Session.";

  return Status::OK();
}

Status TFModel::load_from_saved_model() {
  // create session options
  SessionOptions options;
  ConfigProto& config = options.config;
  if (_num_threads > 0) {
    config.set_intra_op_parallelism_threads(1);
    config.set_inter_op_parallelism_threads(_num_threads);
    config.add_session_inter_op_thread_pool();
    auto pool = config.mutable_session_inter_op_thread_pool(0);
    pool->set_num_threads(_num_threads);
    LOG_INFO << "Set thread pool size: " << pool->num_threads();
  }

  RunOptions run_options;
  std::string path = _model_meta.local.model_path;
  LOG_INFO << "load saved model from path: " << path;

  Status s = LoadSavedModel(options, run_options, path,
                            {tensorflow::kSavedModelTagServe}, &_bundle);
  if (!s.ok()) {
    LOG_FATAL << "Failed Load model from saved_model.pb : " << s;
  }

  return Status::OK();
}

}  // namespace core
}  // namespace delta

#endif  // USE_TF
