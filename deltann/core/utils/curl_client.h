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

#ifndef DELTANN_UTILS_CURL_CLIENT_H_
#define DELTANN_UTILS_CURL_CLIENT_H_
#ifdef USE_TF_SERVING
#ifdef USE_CURL

#include <curl/curl.h>
#include <map>
#include <string>
#include <vector>

#include "core/data.h"
#include "core/shape.h"

#include "dist/json/json-forwards.h"  //NOLINT
#include "dist/json/json.h"           //NOLINT

namespace delta {

using delta::core::InputData;
using delta::core::OutputData;
using delta::core::Shape;
using std::string;
using std::vector;

template <typename T>
Json::Value array_to_json(int nest, T*& ptr, const Shape& shape);

class HttpCurlclient {
 public:
  explicit HttpCurlclient(const string& url);
  ~HttpCurlclient();

  int Post(const vector<InputData>& inputs, vector<OutputData>* output);

  int Post(const std::map<string, Json::Value>& strs, string* result);

 private:
  const std::string _url;
  CURL* _curl;
  CURLcode _res;

  template <typename T>
  Json::Value dump_to_json(const string& name, T* src, const Shape& shape);
};

}  // namespace delta

#endif  // USE_CURL
#endif  // USE_TF_SERVING
#endif  // DELTANN_UTILS_CURL_CLIENT_H_
