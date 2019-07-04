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

#ifdef USE_TF_SERVING

#include <string>
#include <vector>

#include "core/tfserving_model.h"

namespace delta {
namespace core {

TFServingModel::TFServingModel(ModelMeta model_meta, int num_threads)
    : BaseModel(model_meta, num_threads),
      _hostname(model_meta.remote.hostname),
      _port(model_meta.remote.port),
      _model_name(model_meta.remote.model_name),
      _version(model_meta.version) {
  _base_url = _hostname + ":" + std::to_string(_port) + "/v" +
              std::to_string(_version) + "/models/" + _model_name +
              "/versions/" + std::to_string(_version);

  // const std::string url = _hostname + ":" + std::to_string(_port) + "/v" +
  // std::to_string(_version) + "/models/" + _model_name + "/verions/" +
  // std::to_string(_version) + ":predict";

  std::string url = _base_url + ":predict";
  LOG_INFO << " url = " << url;
#if USE_CURL
  _hcc = new HttpCurlclient(url);
#else
  _hcc = new Httpsclient(url);
#endif
}

TFServingModel::~TFServingModel() { delete _hcc; }

int TFServingModel::run(const std::vector<InputData>& inputs,
                        std::vector<OutputData>* output) {
  LOG_INFO << "tfserving run ...";
  _hcc->Post(inputs, output);
}

}  // namespace core
}  // namespace delta

#endif  // USE_TF_SERVING
