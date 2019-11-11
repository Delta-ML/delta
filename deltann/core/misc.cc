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

#include <assert.h>
#include <cstddef>
#include <random>

#include "core/logging.h"
#include "core/misc.h"

namespace delta {

// convert dtype to bytes
std::size_t delta_dtype_size(DataType type) {
  switch (type) {
    case DataType::DELTA_FLOAT32:
      return sizeof(float);
    case DataType::DELTA_FLOAT16:
      return sizeof(float) / 2;
    case DataType::DELTA_DOUBLE:
      return sizeof(double);
    case DataType::DELTA_INT32:
      return sizeof(std::int32_t);
    case DataType::DELTA_CHAR:
      return sizeof(char);
    default:
      LOG_FATAL << "Not support delta dtype";
      return 0;
  }
}

// convert dtype to string
std::string delta_dtype_str(DataType type) {
  switch (type) {
    case DataType::DELTA_FLOAT32:
      return "float";
    case DataType::DELTA_FLOAT16:
      return "float16";
    case DataType::DELTA_DOUBLE:
      return "double";
    case DataType::DELTA_INT32:
      return "int";
    case DataType::DELTA_CHAR:
      return "char";
    default:
      return "Not support delta dtype";
  }
}

// convert string to dtype
DataType delta_str_dtype(const std::string& type) {
  if (type == "float") {
    return DataType::DELTA_FLOAT32;
  } else if (type == "double") {
    return DataType::DELTA_DOUBLE;
  } else if (type == "int") {
    return DataType::DELTA_INT32;
  } else if (type == "char") {
    return DataType::DELTA_CHAR;
  } else {
    LOG_FATAL << "Unknown data type " << type;
    return DataType::DELTA_NONE;
  }
}

std::unordered_map<std::string, EngineType> _global_engines{
    {"TF", EngineType::DELTA_EIGINE_TF},
    {"TFTRT", EngineType::DELTA_EIGINE_TFTRT},
    {"TFLITE", EngineType::DELTA_EIGINE_TFLITE},
    {"TFSERVING", EngineType::DELTA_EIGINE_TFSERVING},
    {"SERVING", EngineType::DELTA_EIGINE_SERVING}};

void make_random(float* ptr, int len) {
  DELTA_CHECK(ptr);
  static std::mt19937 random_engine;
  std::uniform_real_distribution<float> distribution(-1, 1);
  for (int i = 0; i < len; ++i) {
    ptr[i] = distribution(random_engine);
  }
}

char gen_random_char() {
  // return (char)(random(127-33) + 33);
  char c = 'A' + rand_r(&RANDOM_SEED) % 24;
  return c;
}

}  // namespace delta
