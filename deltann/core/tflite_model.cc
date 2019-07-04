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

#ifdef USE_TFLITE
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/tflite_model.h"

namespace delta {
namespace core {

TFLiteModel::TFLiteModel(ModelMeta model_meta, int num_threads)
    : BaseModel(model_meta, num_threads), _interpreter(nullptr) {
  load_model();
}

int TFLiteModel::load_model() {
  // init interpreter
  std::string path = _model_meta.local._model_path;
  LOG_INFO << "load tf lite model from " << path;
  TFL_Model* model = TFL_NewModelFromFile(path.c_str());
  TFL_InterpreterOptions* options = TFL_NewInterpreterOptions();
  LOG_INFO << "tf lite num_threads is " << _num_threads;
  // TFL_InterpreterOptionsSetNumThreads(options, _num_threads);
  TFL_InterpreterOptionsSetNumThreads(options, 1);
  _interpreter = TFL_NewInterpreter(model, options);

  if (!_interpreter) {
    LOG_FATAL << "_interpreter is nullptr, load tflite model failed";
  }

  // The options/model can be deleted immediately after interpreter creation.
  TFL_DeleteInterpreterOptions(options);
  TFL_DeleteModel(model);

  return 0;
}

int TFLiteModel::feed_tensor(const InputData& input) {
  const int id = input.id();
  LOG_INFO << "input id is " << id << ", shape is " << input.shape();
  TFL_Tensor* input_tensor = TFL_InterpreterGetInputTensor(_interpreter, id);
  int in_ndim = TFL_TensorNumDims(input_tensor);
  std::vector<int> in_shape;
  int size = 1;
  for (int i = 0; i < in_ndim; ++i) {
    in_shape.push_back(TFL_TensorDim(input_tensor, i));
    size *= in_shape[i];
  }

  LOG_INFO << "input tensor shape is " << in_shape;

  // resize tensor
  TFL_InterpreterResizeInputTensor(_interpreter, id, in_shape.data(),
                                   in_shape.size());
  TFL_InterpreterAllocateTensors(_interpreter);

  // copy data into input Tensor
  switch (input.dtype()) {
    case DataType::DELTA_FLOAT32:
      TFL_TensorCopyFromBuffer(input_tensor, input.ptr(), size * sizeof(float));
      break;
    case DataType::DELTA_INT32:
      TFL_TensorCopyFromBuffer(input_tensor, input.ptr(), size * sizeof(int));
      break;
    default:
      LOG_FATAL << "not support data type";
  }
}

int TFLiteModel::fetch_tensor(OutputData* output) {
  int id = output->id();
  const TFL_Tensor* output_tensor =
      TFL_InterpreterGetOutputTensor(_interpreter, id);
  int out_ndim = TFL_TensorNumDims(output_tensor);
  std::vector<int> output_shape;
  int size = 1;
  for (int i = 0; i < out_ndim; ++i) {
    output_shape.push_back(TFL_TensorDim(output_tensor, i));
    size *= output_shape[i];
  }
  LOG_INFO << "ouputput tensor shape is " << output_shape << ", size is "
           << size;

  LOG_INFO << "copy result from tf-lite";
  output->set_shape(Shape(output_shape));
  switch (output->dtype()) {
    case DataType::DELTA_FLOAT32:
      output->set_dtype(DataType::DELTA_FLOAT32);
      output->resize(size);
      TFL_TensorCopyToBuffer(output_tensor, output->ptr(),
                             size * sizeof(float));
      break;
    case DataType::DELTA_INT32:
      output->set_dtype(DataType::DELTA_INT32);
      output->resize(size);
      TFL_TensorCopyToBuffer(output_tensor, output->ptr(), size * sizeof(int));
      break;
    default:
      LOG_FATAL << "not support data type";
  }
}

int TFLiteModel::run(const std::vector<InputData>& inputs,
                     std::vector<OutputData>* outputs) {
  LOG_INFO << "TFLiteModel run ...";
  // in
  for (auto& input : inputs) {
    feed_tensor(input);
  }

  // Inference
  LOG_INFO << "tf-lite inference";
  TFL_InterpreterInvoke(_interpreter);

  // get result
  for (auto& output : *outputs) {
    fetch_tensor(&output);
  }

  return 0;
}

}  // namespace core
}  // namespace delta
#endif  // USE_TFLITE
