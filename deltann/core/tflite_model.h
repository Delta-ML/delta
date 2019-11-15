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

#ifndef DELTANN_CORE_TFLITE_MODEL_H_
#define DELTANN_CORE_TFLITE_MODEL_H_

#ifdef USE_TFLITE

#include <string>
#include <utility>
#include <vector>

#include "core/base_model.h"
#include "core/data.h"
#include "core/logging.h"
#include "core/misc.h"

#include "tensorflow/lite/experimental/c/c_api.h"

namespace delta {
namespace core {

class TFLiteModel : public BaseModel {
 public:
  TFLiteModel(ModelMeta model_meta, int num_threads);
  int run(const std::vector<InputData>& inputs,
          std::vector<OutputData>* output) override;

 private:
  TFL_Interpreter* _interpreter;
  int load_model();
  int feed_tensor(const InputData& input);
  int fetch_tensor(OutputData* output);
};

}  // namespace core
}  // namespace delta

#endif  // USE_TFLITE

#endif  // DELTANN_CORE_TFLITE_MODEL_H_
