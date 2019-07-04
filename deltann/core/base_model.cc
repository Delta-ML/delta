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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/base_model.h"

namespace delta {
namespace core {

using std::string;

BaseModel::BaseModel(ModelMeta model_meta, int num_threads)
    : _model_meta(model_meta), _num_threads(num_threads) {}

BaseModel::~BaseModel() {}

}  // namespace core
}  // namespace delta
