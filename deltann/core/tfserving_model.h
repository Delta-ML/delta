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

#ifndef DELTANN_CORE_TFSERVING_MODEL_H_
#define DELTANN_CORE_TFSERVING_MODEL_H_

#ifdef USE_TF_SERVING

#include <string>
#include <vector>

#include "core/base_model.h"

#if USE_CURL
#include "core/utils/curl_client.h"
#else
#include "core/utils/https_client.h"
#endif

namespace delta {
namespace core {

class TFServingModel : public BaseModel {
 public:
  TFServingModel(ModelMeta model_meta, int num_threads);
  ~TFServingModel();

  int run(const std::vector<InputData>& inputs,
          std::vector<OutputData>* output) override;

 private:
#if USE_CURL
  HttpCurlclient* _hcc;
#else
  Httpsclient* _hcc;
#endif

  const std::string _hostname;
  const uint16 _port;
  const std::string _model_name;
  const int _version;
  std::string _base_url;
};

}  // namespace core
}  // namespace delta

#endif  // USE_TF_SERVING

#endif  // DELTANN_CORE_TFSERVING_MODEL_H_
