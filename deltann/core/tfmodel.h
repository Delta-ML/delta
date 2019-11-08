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

#ifndef DELTANN_CORE_TFMODEL_H_
#define DELTANN_CORE_TFMODEL_H_
#ifdef USE_TF

#include <string>
#include <utility>
#include <vector>

#include "core/base_model.h"
#include "core/io.h"
#include "core/logging.h"
#include "core/misc.h"

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/public/session.h"

namespace delta {
namespace core {

using tensorflow::SavedModelBundleLite;
using tensorflow::Status;
using tensorflow::Tensor;

class TFModel : public BaseModel {
 public:
  TFModel(ModelMeta model_meta, int num_threads);
  int run(const std::vector<InputData>& inputs,
          std::vector<OutputData>* output) override;

 private:
  Status load_from_saved_model();
  Status load_from_frozen_graph();
  void feed_tensor(Tensor* tensor, const InputData& input);
  void fetch_tensor(const Tensor& tensor, OutputData* output);
  int set_feeds(std::vector<std::pair<string, Tensor>>* feeds,
                const std::vector<InputData>& inputs);
  int set_fetches(std::vector<string>* fetches,
                  const std::vector<OutputData>& outputs);
  int get_featches(const std::vector<Tensor>& output_tensors,
                   std::vector<OutputData>* outputs);

 private:
  SavedModelBundleLite _bundle;
};

}  // namespace core
}  // namespace delta

#endif  // USE_TF
#endif  // DELTANN_CORE_TFMODEL_H_
