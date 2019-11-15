
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

#ifndef DELTANN_CORE_UTILS_HTTPS_CLIENT_H_
#define DELTANN_CORE_UTILS_HTTPS_CLIENT_H_
#ifdef USE_TF_SERVING

#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "core/data.h"
#include "core/shape.h"
#include "core/utils/https.h"

#include "json/json.h"

namespace delta {

using delta::core::InputData;
using delta::core::OutputData;
using delta::core::Shape;
using std::string;
using std::vector;

class Httpsclient {
 public:
  explicit Httpsclient(string& url);
  ~Httpsclient();

  int Post(const vector<InputData>& inputs, vector<OutputData>* output);

  int Post(const std::map<string, Json::Value>& strs, string* result);

 private:
  std::string _url;
  HTTP_INFO _hi;

  template <typename T>
  Json::Value dump_to_json(const string& name, T* src, const Shape& shape);
};

}  // namespace delta

#endif  // USE_TF_SERVING

#endif  // DELTANN_CORE_UTILS_HTTPS_CLIENT_H_
