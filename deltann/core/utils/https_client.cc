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

#include "core/utils/https_client.h"

#include <iostream>
#include <utility>

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

Httpsclient::Httpsclient(std::string& url) : _url(url) {}

Httpsclient::~Httpsclient() {}

int Httpsclient::Post(const vector<InputData>& inputs,
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

int Httpsclient::Post(const std::map<string, Json::Value>& values,
                      string* result) {
  // data
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

  // init
  char err[1024], response[4096];
  http_init(&_hi, FALSE);

  char* ptr = (char*)(_url.c_str());

  if (http_open(&_hi, ptr) < 0) {
    http_strerror(err, 1024);
    printf("socket error: %s \n", err);
    LOG_FATAL << "socket err or " << std::string(err);
  }

  // set method
  snprintf(_hi.request.method, 8, "POST");

  _hi.request.close = FALSE;
  _hi.request.chunked = FALSE;
  snprintf(_hi.request.content_type, 256, "application/json");
  _hi.request.content_length = str.size();

#define CHECK_HTTPS_WRITE(f)              \
  if (f) {                                \
    http_strerror(err, 1024);             \
    LOG_FATAL << "socket err or " << err; \
  }

  // write header
  CHECK_HTTPS_WRITE(http_write_header(&_hi) < 0);

  // write
  CHECK_HTTPS_WRITE(http_write(&_hi, (char*)(str.c_str()), str.size()) !=
                    str.size());

  // write end
  CHECK_HTTPS_WRITE(http_write_end(&_hi) < 0);

#undef CHECK_HTTPS_WRITE

  // read
  int ret = http_read_chunked(&_hi, response, sizeof(response));
  LOG_INFO << "return code: " << ret;
  LOG_INFO << "return body: " << response;
  *result = std::string(response);

  http_close(&_hi);
}

}  // namespace delta

#endif  // USE_TF_SERVING
