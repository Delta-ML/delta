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
#ifdef USE_CURL

#include "core/utils/curl_client.h"

#include <iostream>
#include <utility>

#include "dist/jsoncpp.cpp"

namespace delta {

using std::pair;
using std::string;

namespace {
size_t write_data(void* ptr, size_t size, size_t nmemb, void* stream) {
  std::string* str = dynamic_cast<std::string*>((std::string*)stream);
  if (nullptr == str || nullptr == ptr) {
    return -1;
  }

  char* pData = reinterpret_cast<char*>(ptr);
  str->append(pData, size * nmemb);
  return nmemb;
}
}  // namespace

template <typename T>
Json::Value array_to_json(int nest, T*& ptr, const Shape& shape) {
  int ndim = shape.ndim();
  DELTA_CHECK_LT(nest, ndim);

  int len = shape[nest];
  Json::Value arr(Json::arrayValue);

  // inner loop
  if ((ndim - 1) == nest) {
    for (int i = 0; i < len; ++i) {
      arr[i] = ptr[i];
    }

    ptr += len;
  } else {
    for (int i = 0; i < len; ++i) {
      int n = nest + 1;
      arr[i] = array_to_json(n, ptr, shape);
    }
  }

  return arr;
}

HttpCurlclient::HttpCurlclient(const std::string& url) : _url(url) {
  /* In windows, this will init the winsock stuff */
  curl_global_init(CURL_GLOBAL_ALL);

  /* get a curl handle */
  _curl = curl_easy_init();
}

HttpCurlclient::~HttpCurlclient() { curl_global_cleanup(); }

int HttpCurlclient::Post(const vector<InputData>& inputs,
                         vector<OutputData>* output) {
  std::map<string, Json::Value> json_data;

  for (auto& input : inputs) {
    switch (input.dtype()) {
      case DataType::DELTA_FLOAT32: {
        LOG_INFO << "float input";
        float* ptr = reinterpret_cast<float*>(input.ptr());
        const Shape& shape = input.shape();
        json_data.insert(pair<string, Json::Value>(
            input.name(), array_to_json(0, ptr, shape)));
        break;
      }
      case DataType::DELTA_INT32: {
        LOG_INFO << "int32 input";
        int* ptr = reinterpret_cast<int*>(input.ptr());
        const Shape& shape = input.shape();
        json_data.insert(pair<string, Json::Value>(
            input.name(), array_to_json(0, ptr, shape)));
        break;
        LOG_FATAL << "Error, DataType::DELTA_INT32 failed not support now";
        break;
      }
      default:
        LOG_FATAL << "not support data type";
    }
  }
  string result;

  Post(json_data, &result);
  LOG_INFO << "post result is " << result;

  // parse json
  Json::Value value;
  Json::CharReaderBuilder builder;
  string errs;
  Json::CharReader* reader = builder.newCharReader();

  const char* p = result.data();
  if (!reader->parse(p, p + result.size(), &value, &errs)) {
    LOG_FATAL << "Error, parse responce error: " << errs;
    return -1;
  }

  Json::Value error = value["error"];
  if (!error.empty()) {
    LOG_INFO << "response error: " << error;
    return -1;
  }

  LOG_INFO << "json value " << value["predictions"];

  return 0;
}

int HttpCurlclient::Post(const std::map<string, Json::Value>& values,
                         string* result) {
  // url
  if (_curl) {
    curl_easy_setopt(_curl, CURLOPT_URL, _url.c_str());
  }
  Json::StreamWriterBuilder builder;
  Json::Value key_val;
  for (auto& value : values) {
    key_val[value.first] = value.second;
  }
  Json::Value val_arr(Json::arrayValue);
  val_arr[0] = key_val;

  string name = "instances";
  Json::Value data;
  data[name] = val_arr;
  string str = Json::writeString(builder, data);

  curl_easy_setopt(_curl, CURLOPT_POSTFIELDS, str.c_str());
  curl_easy_setopt(_curl, CURLOPT_WRITEFUNCTION, write_data);
  curl_easy_setopt(_curl, CURLOPT_WRITEDATA, reinterpret_cast<void*>(result));
  curl_easy_setopt(_curl, CURLOPT_NOSIGNAL, 1);
  curl_easy_setopt(_curl, CURLOPT_CONNECTTIMEOUT, 3);
  curl_easy_setopt(_curl, CURLOPT_TIMEOUT, 3);

  /* Perform the request, res will get the return code */
  _res = curl_easy_perform(_curl);

  if (_res != CURLE_OK)
    std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(_res)
              << std::endl;
}

}  // namespace delta

#endif  // USE_CURL
#endif  // USE_TF_SERVING
